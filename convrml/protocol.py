"""Recorded experiment settings, model construction and dataset identity checks."""

from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path

import torch

from dataset import ImagePairs, NumpyPairs, read_split_file

SETTINGS_PATH = Path(__file__).resolve().parents[1] / "configs/experiment_settings.json"


def settings():
    """Read recorded settings as a fresh dictionary, without changing the JSON file."""
    with SETTINGS_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def sha256(path):
    """Hash file bytes in chunks, without retaining a checkpoint in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """Atomically replace complete JSON; reject NaN/Inf instead of hiding them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def profile(experiment, dataset, method, member=None):
    """Build the run dictionary consumed by training, checkpoint loading and inference.

    Dataset/method select a recorded optimizer profile and shuffle policy.
    Separate seeds control initialization, training randomness, data order and q
    draws. ``member`` is required only for an individual ensemble member (0-9).
    """
    config = settings()
    data = config["datasets"][dataset]
    if data["experiment"] != experiment:
        raise ValueError("Dataset does not belong to the requested experiment")
    if method not in config["methods"][f"experiment{experiment}"]:
        raise ValueError("Method is not part of this experiment")
    if method == "ensemble" and member is None:
        raise ValueError("Choose --member 0..9 for ensemble training/loading")
    if method != "ensemble" and member is not None:
        raise ValueError("Only ensemble models use --member")
    if member is not None and member not in range(10):
        raise ValueError("Ensemble member index must be 0..9")
    # Select a recorded optimizer profile, not a different model architecture.
    if dataset == "raw_corrupted":
        family = "adaptation"
    elif method == "qutcc" or experiment == 3:
        family = "paper"
    else:
        family = "baseline"
    result = dict(config["profiles"][family])
    if data["loader"] == "images":
        result["microbatch"] = 1
    # Ensemble members use independent initialization AND image-order streams.
    policy = config["seed_policy"]
    if family == "baseline":
        index = member if method == "ensemble" else policy["arm_index"][method]
        seeds = {
            name: policy["baseline_" + name] + index
            for name in ("initialization", "training", "data", "quantile")
        }
        shuffle = "persistent_generator"
    else:
        seed = policy["paper_base"] + data["camera_index"]
        seeds = dict(
            initialization=seed, training=seed, data=seed, quantile=seed + 1000
        )
        shuffle = "per_epoch_permutation"
    return dict(
        result,
        schema=config["schema"],
        experiment=experiment,
        dataset=dataset,
        method=method,
        member=member,
        seeds=seeds,
        shuffle=shuffle,
        dropout=config["dropout_probability"],
        family=family,
    )


def make_model(run):
    """Construct dataset-independent model classes with recorded initialization seeds."""
    torch.manual_seed(run["seeds"]["initialization"])
    from models import Im2im, QUTCC, MCDropout, EnsembleMember

    if run["method"] == "qutcc":
        return QUTCC()
    if run["method"] == "im2im":
        return Im2im()
    if run["method"] == "mc_dropout":
        return MCDropout(run["dropout"])
    if run["method"] == "ensemble":
        return EnsembleMember()
    raise ValueError("Unknown model method: " + run["method"])


def dataset_for(dataset, manifest, pairs, split):
    """Choose the loader; split is train/val/test/calibration, never a new partition."""
    if settings()["datasets"][dataset]["loader"] == "numpy":
        if pairs is None:
            raise ValueError("PLD experiment requires --pairs for its NumPy cache")
        return NumpyPairs(pairs, manifest, split)
    if pairs is not None:
        raise ValueError("DLMD experiment reads paired image paths, not a NumPy cache")
    return ImagePairs(manifest, split)


def id_digest(ids):
    """Hash ordered image IDs; both membership and order are experiment identity."""
    return hashlib.sha256(
        json.dumps(list(map(str, ids)), separators=(",", ":")).encode()
    ).hexdigest()


def audit_manifest(manifest):
    """Check nonempty, disjoint split IDs and record the supplied data identity.

    Accept new datasets and image sizes. Published split counts/hashes remain
    reference records, rather than restrictions on the research prototype.
    """
    splits = read_split_file(manifest)
    identities, seen = {}, set()
    for role in ("train", "val", "test", "calibration"):
        if role not in splits:
            continue
        ids = [str(row["id"]) for row in splits[role]]
        if not ids or len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError("Empty, duplicated or overlapping split IDs")
        seen.update(ids)
        identities[role] = dict(count=len(ids), ids_sha256=id_digest(ids))
    if not identities:
        raise ValueError("Manifest has no recognized splits")
    return dict(manifest_sha256=sha256(manifest), splits=identities)


def autocast(run, device):
    """Use the recorded BF16 context, or preserve normal FP32 computation."""
    if run["precision"] == "bfloat16":
        return torch.autocast(device.type, dtype=torch.bfloat16)
    return nullcontext()


def learning_rate(step, total_steps, run):
    """Recorded constant or warmup/cosine schedule; resume keeps its full horizon."""
    peak = run["peak_lr"]
    if run["schedule"] == "constant":
        return peak
    # The saved profile warms up for 30% of all optimizer updates, then decays
    # over the remaining 70%. --epochs changes when to stop, not this horizon.
    progress = step / total_steps
    initial_lr = peak / 10
    warmup_fraction = 0.3
    if progress <= warmup_fraction:
        return initial_lr + progress / warmup_fraction * (peak - initial_lr)

    minimum_lr = initial_lr / 1e4
    # Keep the recorded operation order for numerical parity with saved runs.
    cosine_angle = math.pi * (progress - warmup_fraction) / 0.7
    cosine = math.cos(cosine_angle)
    return minimum_lr + (peak - minimum_lr) / 2 * (cosine + 1)


def center(model, method, y):
    """One-pass validation prediction, shape (N,3,H,W).

    With model.eval(), MC dropout is deterministic here; final inference instead
    averages stochastic passes. An ensemble training run validates its member,
    not an aggregate. Im2im returns its center head; QUTCC queries q=0.5.
    """
    if method == "ensemble":
        return model(y)[0]
    if method == "mc_dropout":
        return model(y)
    return model.center(y)
