"""Materialize independent raw-input modifications from a paired RGB manifest.

Only measurement files are generated. Original measurements and targets are
opened read-only and checked against their recorded hashes. Each output manifest
works with dataset.ImagePairs; it retains the existing split order and IDs.
No model, calibration, training, or GPU allocation is involved.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import datetime
import fcntl
import hashlib
from io import BytesIO
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import PIL
import torch

from .dead_pixels import apply_dead_pixels, sample_dead_pixel_mask
from .fixed_pattern import apply_fixed_pattern, sample_fixed_pattern
from .presets import PRESETS, RANDOM_PER_IMAGE, TRANSFORMS

PROTOCOL = "mirflickr_raw_input_modifications_v1"
ROLES = ("train", "val", "calibration", "test")
_WORKER = None


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def seed_for(base_seed, *parts):
    """Stable named streams, independent of worker count, ordering, and Python hash."""
    text = json.dumps([PROTOCOL, base_seed, *parts], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % (2**63)


def code_fingerprint():
    files = sorted(Path(__file__).parent.glob("*.py"))
    return digest(b"".join(p.name.encode() + b"\0" + p.read_bytes() for p in files))


def resolve_record(record, manifest_folder):
    result = dict(record)
    path = Path(record["path"])
    if not path.is_absolute():
        path = manifest_folder / path
    result["path"] = str(path.resolve(strict=True))
    return result


def read_rgb(record):
    """Validate the actual source bytes and decoded pixels without normalizing them."""
    path = Path(record["path"])
    blob = path.read_bytes()
    file_hash = digest(blob)
    expected = record.get("file_sha256", record.get("sha256"))
    if expected is not None and file_hash != expected:
        raise ValueError(f"Source file hash mismatch: {path}")
    with Image.open(BytesIO(blob)) as image:
        if image.mode != "RGB":
            raise ValueError(f"Expected prepared 8-bit RGB, not {image.mode}: {path}")
        pixels = np.array(image, copy=True)
    pixel_hash = digest(pixels.tobytes())
    if record.get("pixel_sha256", pixel_hash) != pixel_hash:
        raise ValueError(f"Source pixel hash mismatch: {path}")
    result = dict(record, file_sha256=file_hash, pixel_sha256=pixel_hash)
    result.update(mode="RGB", shape=list(pixels.shape))
    return pixels, result


def as_tensor(pixels):
    return torch.from_numpy(pixels.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255


def sensor_state(shape, base_seed):
    """Every worker reconstructs the same persistent sensor, across every split."""
    template = torch.zeros(1, *shape)
    mask_seed = seed_for(base_seed, "sensor", "dead_pixels", list(shape))
    pattern_seed = seed_for(base_seed, "sensor", "fixed_pattern", list(shape))
    mask = sample_dead_pixel_mask(
        template,
        **PRESETS["dead_pixels"],
        generator=torch.Generator().manual_seed(mask_seed),
    )
    pattern = sample_fixed_pattern(
        template,
        **PRESETS["fixed_pattern"],
        generator=torch.Generator().manual_seed(pattern_seed),
    )
    return mask, pattern


def initialize_worker(output, base_seed, shape, variants):
    """Initialize process-local state once, with the same simulated sensor per worker.

    Image noise gets a separate named seed in build_pair. The persistent sensor
    mask/gain/offset must instead match across every worker, split and image.
    """
    global _WORKER
    torch.set_num_threads(1)
    mask, pattern = sensor_state(shape, base_seed)
    _WORKER = dict(
        output=Path(output),
        seed=base_seed,
        shape=shape,
        variants=variants,
        mask=mask,
        pattern=pattern,
    )


def transform(y, variant, image_seed, state):
    """Apply one measurement-only variant; reuse persistent sensor defects."""
    if variant == "dead_pixels":
        return apply_dead_pixels(y, state["mask"])
    if variant == "fixed_pattern":
        return apply_fixed_pattern(y, state["pattern"])
    kwargs = {}
    if variant in RANDOM_PER_IMAGE:
        kwargs["generator"] = torch.Generator().manual_seed(image_seed)
    return TRANSFORMS[variant](y, **kwargs)


def save_measurement(path, pixels):
    """Atomically save lossless RGB and verify decoded pixels, including on resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(".tmp")
        Image.fromarray(pixels).save(temporary, format="PNG", compress_level=1)
        temporary.replace(path)
    blob = path.read_bytes()
    with Image.open(BytesIO(blob)) as image:
        actual = np.array(image)
        if image.mode != "RGB" or not np.array_equal(actual, pixels):
            raise ValueError(f"Output readback/reproducibility mismatch: {path}")
    return dict(
        path=str(path.resolve()),
        mode="RGB",
        shape=list(pixels.shape),
        file_sha256=digest(blob),
        pixel_sha256=digest(pixels.tobytes()),
    )


def build_pair(item):
    """Validate one original pair and export each requested measurement variant.

    Targets are only read/hashed and referenced in manifests. Their pixel data
    are never passed to transforms. Output hashes describe decoded uint8 pixels
    and PNG bytes separately, so quantization and file encoding remain traceable.
    """
    role, index, row = item
    state = _WORKER
    original, original_record = read_rgb(row["measurement"])
    target, target_record = read_rgb(row["target"])
    if original.shape != target.shape or tuple(original.shape) != tuple(
        state["shape"][1:]
    ) + (3,):
        raise ValueError(f"Unexpected measurement/target shape for {row['id']}")
    y = as_tensor(original)
    outputs = {}
    for variant in state["variants"]:
        image_seed = seed_for(state["seed"], role, str(row["id"]), variant)
        modified = transform(y, variant, image_seed, state)
        if modified.shape != y.shape or not torch.isfinite(modified).all():
            raise ValueError(f"Invalid modification: {variant}/{row['id']}")
        pixels = (
            modified[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
        )
        path = state["output"] / variant / "measurements" / role / f"{index:05d}.png"
        measurement = save_measurement(path, pixels)
        delta = (pixels.astype(np.float64) - original) / 255
        outputs[variant] = dict(
            row,
            measurement=measurement,
            target=target_record,
            original_measurement=original_record,
            modification=variant,
            parameters=PRESETS[variant],
            seed=image_seed,
            input_change=dict(
                mae=float(np.abs(delta).mean()),
                mse=float(np.square(delta).mean()),
                changed_fraction=float(np.mean(pixels != original)),
                zero_fraction=float(np.mean(pixels == 0)),
                saturated_fraction=float(np.mean(pixels == 255)),
            ),
        )
    clean = dict(row, measurement=original_record, target=target_record)
    return role, index, clean, outputs


def validate_source(source, manifest_folder, output, limit):
    if set(source["roles"]) != set(ROLES):
        raise ValueError(
            "Manifest must contain train, val, calibration, and test roles."
        )
    selected, seen_ids, seen_measurements = {}, set(), set()
    for role in ROLES:
        rows = source["roles"][role]
        if source.get("counts", {}).get(role, len(rows)) != len(rows):
            raise ValueError(f"Source count mismatch: {role}")
        selected[role] = []
        for index, row in enumerate(rows):
            identifier = str(row["id"])
            if identifier in seen_ids:
                raise ValueError(
                    f"ID appears in multiple source rows/splits: {identifier}"
                )
            seen_ids.add(identifier)
            resolved = dict(row)
            for field in ("measurement", "target"):
                resolved[field] = resolve_record(row[field], manifest_folder)
                path = Path(resolved[field]["path"])
                if path.is_relative_to(output) or output.is_relative_to(path.parent):
                    raise ValueError(
                        "Output must be separate from every original image folder."
                    )
            measurement = resolved["measurement"]["path"]
            if measurement in seen_measurements:
                raise ValueError(
                    "A measurement path appears in multiple source rows/splits."
                )
            seen_measurements.add(measurement)
            if limit is None or index < limit:
                selected[role].append(resolved)
        if not selected[role]:
            raise ValueError(f"Empty source split: {role}")
    return selected


def save_sensor_files(output, shape, base_seed):
    mask, pattern = sensor_state(shape, base_seed)
    folder = output / "sensor"
    folder.mkdir(exist_ok=True)
    entries = {}
    for name, tensor in [
        ("dead_mask", mask),
        ("gain", pattern.gain),
        ("offset", pattern.offset),
    ]:
        path = folder / f"{name}.npy"
        values = tensor.numpy()
        if path.exists():
            if not np.array_equal(np.load(path, allow_pickle=False), values):
                raise ValueError(f"Persistent sensor state changed: {path}")
        else:
            with path.with_suffix(".tmp").open("wb") as stream:
                np.save(stream, values, allow_pickle=False)
            path.with_suffix(".tmp").replace(path)
        entries[name] = dict(path=str(path), sha256=digest(path.read_bytes()))
    entries["dead_pixel_count"] = int(mask.sum())
    entries["spatial_pixels"] = int(mask.numel())
    return entries


def make_preview(output, roles, variants):
    """One example from each split, showing the original, every variant, and target."""
    columns = ["original", *variants, "target"]
    tile_width, tile_height, label_height = 144, 81, 30
    canvas = Image.new(
        "RGB",
        (len(columns) * tile_width, len(ROLES) * (tile_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, role in enumerate(ROLES):
        row = roles[role][0]
        for column_index, name in enumerate(columns):
            if name == "original":
                path = row["measurement"]["path"]
            elif name == "target":
                path = row["target"]["path"]
            else:
                path = output / name / "measurements" / role / "00000.png"
            with Image.open(path) as image:
                thumbnail = image.convert("RGB").resize((tile_width, tile_height))
            x, top = column_index * tile_width, row_index * (tile_height + label_height)
            draw.text((x + 3, top + 2), f"{role}\n{name}", fill="black")
            canvas.paste(thumbnail, (x, top + label_height))
    canvas.save(output / "preview.jpg", quality=90, subsampling=0)


def prepare(
    manifest, output, *, workers=4, seed=20261030, variants=None, limit_per_split=None
):
    """Validate source -> generate variants -> verify manifests -> write READY.json.

    A matching protocol permits exact resumption; changed source/settings require
    a new output directory. READY.json is published only after PNG readback,
    original-target identities and split ordering all pass verification.
    """
    if workers < 1 or (limit_per_split is not None and limit_per_split < 1):
        raise ValueError("workers and limit_per_split must be positive.")
    variants = list(PRESETS) if variants is None else list(variants)
    if (
        not variants
        or len(set(variants)) != len(variants)
        or set(variants) - set(PRESETS)
    ):
        raise ValueError("Choose unique, supported variants.")
    manifest, output = Path(manifest).resolve(strict=True), Path(output).resolve()
    if output == manifest.parent or manifest.is_relative_to(output):
        raise ValueError("Output must not contain or replace the source manifest.")
    # 1. Resolve and audit the original pairs before creating output images.
    source_blob = manifest.read_bytes()
    source = json.loads(source_blob)
    roles = validate_source(source, manifest.parent, output, limit_per_split)
    first_pixels, _ = read_rgb(roles[ROLES[0]][0]["measurement"])
    shape = (3, *first_pixels.shape[:2])
    counts = {role: len(rows) for role, rows in roles.items()}
    identity = dict(
        protocol=PROTOCOL,
        source_manifest=str(manifest),
        source_sha256=digest(source_blob),
        code_sha256=code_fingerprint(),
        seed=seed,
        shape=list(shape),
        counts=counts,
        variants={name: PRESETS[name] for name in variants},
        smoke=limit_per_split is not None,
        software=dict(
            torch=torch.__version__, numpy=np.__version__, pillow=PIL.__version__
        ),
        input_format="prepared uint8 RGB / 255; not native sensor-bit-depth arrays",
        output_format="clipped, rounded uint8 RGB PNG; JPEG variant decoded once before PNG storage",
        target_policy="Original targets referenced read-only; no resampling, new targets, or split changes",
        sensor_policy="One dead-pixel mask and one gain/offset pattern shared across all splits/frames",
        noise_policy="Independent named image streams; Poisson SD varies with signal; parameters are pre-clipping/quantization",
        training_submitted=False,
    )
    output.mkdir(parents=True, exist_ok=True)
    # 2. Lock this build and reject an incompatible partial output directory.
    with (output / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol_path = output / "protocol.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != identity:
            raise ValueError(
                "Existing output has different source/code/settings; choose a new output folder."
            )
        write_json(protocol_path, identity)
        (output / "READY.json").unlink(missing_ok=True)
        sensor = save_sensor_files(output, shape, seed)
        items = [
            (role, index, row)
            for role in ROLES
            for index, row in enumerate(roles[role])
        ]
        built = {name: {role: [] for role in ROLES} for name in ["clean", *variants]}
        # 3. Generate in source order; all workers reconstruct the same sensor.
        started = time.monotonic()
        initializer_args = (str(output), seed, shape, variants)
        if workers == 1:
            initialize_worker(*initializer_args)
        context = (
            ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_worker,
                initargs=initializer_args,
            )
            if workers > 1
            else nullcontext()
        )
        with context as pool:
            results = (
                pool.map(build_pair, items, chunksize=4)
                if pool
                else map(build_pair, items)
            )
            for processed, (role, index, clean, outputs) in enumerate(results, 1):
                if index != len(built["clean"][role]):
                    raise ValueError("Worker results changed the source order.")
                built["clean"][role].append(clean)
                for name, row in outputs.items():
                    built[name][role].append(row)
                if processed % 250 == 0 or processed == len(items):
                    progress = dict(
                        processed=processed,
                        total=len(items),
                        generated=processed * len(variants),
                        elapsed_seconds=time.monotonic() - started,
                    )
                    write_json(output / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
        # 4. Write one loader-compatible manifest per variant, preserving targets.
        summaries = {}
        for name, selected_roles in built.items():
            for role in ROLES:
                if [r["id"] for r in selected_roles[role]] != [
                    r["id"] for r in roles[role]
                ]:
                    raise ValueError("Output split identities/order changed.")
            relative_roles = {}
            for role, rows in selected_roles.items():
                relative_roles[role] = []
                for row in rows:
                    entry = dict(row)
                    if name != "clean":
                        entry["measurement"] = dict(
                            row["measurement"],
                            path=str(
                                Path(row["measurement"]["path"]).relative_to(
                                    output / name
                                )
                            ),
                        )
                    relative_roles[role].append(entry)
            payload = dict(
                protocol=PROTOCOL,
                modification=name,
                parameters=PRESETS.get(name, {}),
                source_manifest_sha256=identity["source_sha256"],
                counts=counts,
                roles=relative_roles,
                sensor=sensor,
                smoke=identity["smoke"],
                training_submitted=False,
            )
            path = output / name / "manifest.json"
            write_json(path, payload)
            summary = dict(
                manifest=str(path),
                manifest_sha256=digest(path.read_bytes()),
                count=sum(counts.values()),
            )
            if name != "clean":
                changes = [
                    r["input_change"] for rows in selected_roles.values() for r in rows
                ]
                summary["mean_input_change"] = {
                    key: float(np.mean([r[key] for r in changes])) for key in changes[0]
                }
                summary["input_rmse"] = summary["mean_input_change"]["mse"] ** 0.5
            summaries[name] = summary
        if digest(manifest.read_bytes()) != identity["source_sha256"]:
            raise ValueError("Source manifest changed during preparation.")
        # 5. Mark readiness only after every variant and source check completes.
        make_preview(output, roles, variants)
        ready = dict(
            passed=True,
            smoke=identity["smoke"],
            protocol=PROTOCOL,
            completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            counts=counts,
            source_pairs=sum(counts.values()),
            generated_pngs=len(items) * len(variants),
            verification="All source file/pixel hashes and every generated PNG readback verified; split IDs/order preserved",
            source_sha256=identity["source_sha256"],
            code_sha256=identity["code_sha256"],
            sensor=sensor,
            datasets=summaries,
            training_submitted=False,
            elapsed_seconds=time.monotonic() - started,
        )
        write_json(output / "READY.json", ready)
        print(
            json.dumps(
                dict(
                    passed=True,
                    generated_pngs=ready["generated_pngs"],
                    output=str(output),
                )
            ),
            flush=True,
        )
        return ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--variants", nargs="+", choices=list(PRESETS))
    parser.add_argument(
        "--limit-per-split",
        type=int,
        help="Small smoke build, explicitly marked in every output manifest",
    )
    args = parser.parse_args()
    prepare(
        args.manifest,
        args.output,
        workers=args.workers,
        seed=args.seed,
        variants=args.variants,
        limit_per_split=args.limit_per_split,
    )


if __name__ == "__main__":
    main()
