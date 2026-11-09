"""Calibrate the frozen Gaussian specialists, then evaluate held-out scenes.

The comparison source is the immutable Exp 4 snapshot. Reusing its corruption
seeds and calibration functions makes these inputs identical to the demo's
Gaussian inputs. Each specialist is calibrated on Gaussian inputs only.
"""

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time

import torch


def source_digest(source):
    """Record every Python dependency in the frozen comparison snapshot."""
    digest = hashlib.sha256()
    for name in ("dataset.py", "models", "convrml", "input_modifications", "experiments/experiment4_demo"):
        root = source / name
        paths = [root] if root.is_file() else sorted(root.rglob("*.py"))
        if not paths:
            raise ValueError(f"Missing comparison dependency: {name}")
        for path in paths:
            digest.update(str(path.relative_to(source)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def top_error_overlap(lower, upper, prediction, target):
    """Share of the most uncertain 10% of spatial pixels also in top 10% error.

    RGB values are averaged at each pixel. Constant maps are undefined, rather
    than assigning arbitrary tied pixel positions a meaningful overlap.
    """
    width = (upper - lower).mean(1).flatten(1)
    error = (prediction - target).abs().mean(1).flatten(1)
    rows = []
    for uncertainty, residual in zip(width, error):
        if uncertainty.max() == uncertainty.min() or residual.max() == residual.min():
            rows.append(None)
            continue
        count = max(1, math.ceil(0.1 * uncertainty.numel()))
        # Stable ordering makes ties reproducible across restarts.
        selected = torch.argsort(uncertainty, descending=True, stable=True)[:count]
        largest_errors = torch.argsort(residual, descending=True, stable=True)[:count]
        mask = torch.zeros_like(uncertainty, dtype=torch.bool)
        mask[largest_errors] = True
        rows.append(float(mask[selected].float().mean()))
    return rows


def load_specialist(e, root, method):
    """Require the completed, from-scratch Gaussian epoch-50 checkpoint."""
    folder = root / "results" / method
    completed = json.loads((folder / "COMPLETE.json").read_text())
    if not (completed["passed"] and completed["epochs"] == 50 and completed["method"] == method):
        raise ValueError("Specialist has not completed 50 epochs")
    path = folder / "checkpoints/epoch_050.pt"
    checkpoint, digest = e.read_checkpoint(path)
    config = checkpoint["config"]
    expected = dict(
        method=method,
        protocol="raw_online_gaussian_sigma001_scratch50_20261101_v1",
        manifest_sha256=e.MANIFEST_SHA,
        training_condition="gaussian",
        gaussian_sigma=0.01,
        pretrained=False,
        initial_trained_steps=0,
        initial_weights_sha256=e.INITIAL_SHA[method],
    )
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Wrong Gaussian checkpoint identity or initialization")
    if checkpoint["epoch"] != 50 or checkpoint.get("smoke", False):
        raise ValueError("Expected the full epoch-50 checkpoint")
    if [row["epoch"] for row in checkpoint["history"]] != list(range(1, 51)):
        raise ValueError("Incomplete training history")
    model = e.QUTCC() if method == "qutcc" else e.Im2im()
    e.load_model_weights(model, checkpoint["model"])
    return model.cuda().eval().requires_grad_(False), dict(
        checkpoint=str(path), checkpoint_sha256=digest, training_config=config
    )


@torch.inference_mode()
def run(args):
    sys.path.insert(0, str(args.comparison_source.resolve()))
    from experiments.experiment4_demo import evaluate as e
    from convrml.metrics import interval_metrics, finite_mean

    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "evaluation.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGUSR1, e.stop)
    signal.signal(signal.SIGTERM, e.stop)
    torch.set_num_threads(2)
    torch.manual_seed(20261101)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    e.audit_manifest(args.manifest)
    model, provenance = load_specialist(e, args.training_root, args.method)
    request = dict(
        schema="gaussian_specialist_calibration_test_v1",
        method=args.method, epoch=50, sigma=0.01,
        manifest_sha256=e.sha(args.manifest),
        checkpoint_sha256=provenance["checkpoint_sha256"],
        evaluator_sha256=e.sha(Path(__file__)),
        comparison_source_sha256=source_digest(args.comparison_source),
        calibration_condition="gaussian", calibration_count=3000,
        test_condition="gaussian", test_count=2000,
        target_coverage=0.9, calibration="tight empirical; not RCPS",
        test_used_for_calibration=False, examples=args.examples,
    )
    request_path = args.output / "request.json"
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise ValueError("Refusing to reuse outputs from a different evaluation")
    e.write(request_path, request)
    if (args.output / "READY.json").exists():
        ready = json.loads((args.output / "READY.json").read_text())
        if ready["report_sha256"] != e.sha(args.output / "report.json"):
            raise ValueError("Completed report changed")
        print("Already complete", flush=True)
        return

    # Only calibration targets are opened until this fit has been committed.
    started = time.monotonic()
    e.write(args.output / "progress.json", dict(stage="calibration", processed=0, total=3000))
    calibration = e.ModifiedPairs(args.manifest, "calibration", "gaussian")
    arm = f"{args.method}_gaussian50"
    fit = e.calibrate(model, arm, calibration, args.output)
    fit["calibrated_on"] = "3000 held-out Gaussian-corrupted calibration scenes only"
    e.write(args.output / "calibration.json", fit)
    calibration_seconds = time.monotonic() - started
    print("CALIBRATED", json.dumps(fit), flush=True)

    test = e.ModifiedPairs(args.manifest, "test", "gaussian")
    if len(calibration) != 3000 or len(test) != 2000:
        raise ValueError("Unexpected calibration or test count")
    digest, rows = hashlib.sha256(), []
    started = time.monotonic()
    for index, batch in enumerate(e.loader(test)):
        e.interrupted()
        digest.update(batch[0].numpy().tobytes())
        measurement, target = e.move(batch)
        prediction, lower, upper = e.predict(model, args.method, measurement, fit)
        e.finite(prediction, lower, upper, target)
        row = dict(index=index, id=test.ids[index])
        for space in ("raw", "clipped"):
            values = interval_metrics(prediction, lower, upper, target, clipped=space == "clipped")
            row[space] = {
                name: float(value[0]) if torch.isfinite(value[0]) else None
                for name, value in values.items()
            }
        p, lo, hi = (value.clamp(0, 1) for value in (prediction, lower, upper))
        row["clipped"]["top10_error_overlap"] = top_error_overlap(lo, hi, p, target)[0]
        rows.append(row)
        if index < args.examples:
            folder = args.output / f"examples/{index:03d}"
            for name, image in dict(original=batch[2][0], input=measurement[0], target=target[0], reconstruction=p[0]).items():
                e.save_rgb(folder / f"{name}.png", image)
            e.save_width(folder / "uncertainty.png", lo[0], hi[0])
            error = (p[0] - target[0]).abs()
            e.save_width(folder / "absolute_error.png", torch.zeros_like(error), error)
        if (index + 1) % 100 == 0:
            e.write(args.output / "progress.json", dict(stage="test", processed=index + 1, total=2000))
            print("TEST", index + 1, "/2000", flush=True)
    metrics, valid_counts = {}, {}
    for space in ("raw", "clipped"):
        metrics[space], valid_counts[space] = {}, {}
        for name in rows[0][space]:
            mean, count = finite_mean(row[space][name] for row in rows)
            metrics[space][name], valid_counts[space][name] = mean, count
    if e.sha(provenance["checkpoint"]) != provenance["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed during evaluation")
    report = dict(
        request=request, provenance=provenance, calibration=fit,
        test_count=len(rows), metrics=metrics, metric_valid_counts=valid_counts,
        modified_input_float32_sha256=digest.hexdigest(), per_image=rows,
        timing=dict(calibration_seconds=calibration_seconds, test_seconds=time.monotonic() - started),
        gpu=torch.cuda.get_device_name(),
        uncertainty="Calibrated upper minus lower; normalized intensity units.",
        correlation="Per-image Pearson/Spearman between interval width and absolute reconstruction error across RGB values.",
        top10_error_overlap="Intersection of most uncertain 10% spatial pixels and largest-error 10%, divided by selection size; RGB averaged per pixel; chance baseline 0.10.",
        calibration_target_met=fit["target_met"],
    )
    e.write(args.output / "report.json", report)
    e.write(args.output / "READY.json", dict(completed=True, report_sha256=e.sha(args.output / "report.json")))
    e.write(args.output / "progress.json", dict(stage="complete", test_count=len(rows)))
    print("COMPLETE", json.dumps(metrics["clipped"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("qutcc", "im2im"), required=True)
    for name in ("comparison-source", "training-root", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--examples", type=int, default=50)
    args = parser.parse_args()
    if not 0 <= args.examples <= 2000:
        parser.error("examples must be between 0 and 2000")
    run(args)
