"""Held-out test metrics and exports for already calibrated, frozen models.

This module does not fit calibration parameters. The top-level calibration.py
workflow fits them on the separate calibration split before calling these helpers.
"""

import csv
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import finite_mean, interval_metrics
from .protocol import write_json


def summarize_metrics(rows):
    """Average each metric over defined per-image values and retain valid counts."""
    means, counts = {}, {}
    for space in ("raw", "clipped"):
        means[space], counts[space] = {}, {}
        for name in rows[0][space]:
            mean, count = finite_mean(row[space][name] for row in rows)
            means[space][name] = mean
            counts[space][name] = count
    return means, counts


def evaluate_test_set(
    dataset,
    ids,
    predict,
    *,
    batch_size,
    device,
    workers,
    example_count=0,
    perceptual=None,
):
    """Predict test images, measure metrics, and retain the first requested examples.

    ``ids`` follows dataset order; ``predict`` returns calibrated center/lower/upper.
    The result contains per-image rows, metric means/valid counts, timings and
    example arrays. Predictions/bounds in examples are clipped for display;
    reported metrics retain both raw and clipped views.
    """
    rows, offset = [], 0
    examples = {
        name: [] for name in ("measurement", "target", "prediction", "lower", "upper")
    }
    started = time.monotonic()
    prediction_seconds, metric_seconds = 0.0, 0.0
    for batch in DataLoader(dataset, batch_size=batch_size, num_workers=workers):
        # 1. Predict with calibration already frozen; synchronize for GPU timing.
        measurement, target = batch[:, 0].to(device), batch[:, 1].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        prediction_started = time.monotonic()
        prediction, lower, upper = predict(measurement)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        prediction_seconds += time.monotonic() - prediction_started
        if not all(torch.isfinite(value).all() for value in (prediction, lower, upper)):
            raise ValueError("Nonfinite calibrated predictions")

        # 2. Measure raw/clipped views without altering the saved calibration fit.
        metric_started = time.monotonic()
        measured = {}
        for space in ("raw", "clipped"):
            measured[space] = interval_metrics(
                prediction, lower, upper, target, clipped=space == "clipped"
            )
            for name, values in measured[space].items():
                if (
                    name not in ("pearson", "spearman")
                    and not torch.isfinite(values).all()
                ):
                    raise ValueError("Nonfinite " + name + " scores")
        lpips = perceptual(prediction, target) if perceptual is not None else None
        if lpips is not None and not torch.isfinite(lpips).all():
            raise ValueError("Nonfinite LPIPS scores")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        metric_seconds += time.monotonic() - metric_started

        # 3. Preserve manifest order and nulls for undefined per-image metrics.
        for index in range(len(batch)):
            row = dict(index=offset + index, id=ids[offset + index])
            for space, values in measured.items():
                row[space] = {}
                for name, batch_values in values.items():
                    value = float(batch_values[index])
                    row[space][name] = value if math.isfinite(value) else None
            # LPIPS is defined on clipped images, not unbounded raw predictions.
            row["raw"]["lpips"] = None
            row["clipped"]["lpips"] = float(lpips[index]) if lpips is not None else None
            rows.append(row)
            if offset + index < example_count:
                images = dict(
                    measurement=measurement,
                    target=target,
                    prediction=prediction.clamp(0, 1),
                    lower=lower.clamp(0, 1),
                    upper=upper.clamp(0, 1),
                )
                for name, values in images.items():
                    examples[name].append(values[index].cpu().numpy())
        offset += len(batch)
    metrics, valid_counts = summarize_metrics(rows)
    timing = dict(
        test_seconds=time.monotonic() - started,
        prediction_seconds=prediction_seconds,
        prediction_seconds_per_image=prediction_seconds / len(rows),
        metric_seconds=metric_seconds,
    )
    return dict(
        rows=rows,
        examples=examples,
        metrics=metrics,
        valid_counts=valid_counts,
        timing=timing,
    )


def make_report(result, *, metadata, lpips_enabled):
    """Combine measured scores with the workflow's provenance and timing records.

    ``metadata`` supplies epoch, camera, method, request, calibration, provenance,
    data identity, timing and example count.
    This only assembles JSON fields: it does not predict, fit or recompute metrics.
    """
    return dict(
        metadata,
        test_count=len(result["rows"]),
        metrics=result["metrics"],
        metric_valid_counts=result["valid_counts"],
        metric_protocol=dict(
            lpips=dict(
                enabled=lpips_enabled,
                backbone="alex",
                inputs="clipped RGB [0,1]",
                raw="unavailable: unbounded outputs outside supported intensity range",
            ),
            correlations="Per-image Pearson/Spearman: UB-LB versus absolute target-prediction error; flattened RGB; no smoothing/regions",
            undefined="null; means exclude undefined images and record valid counts",
        ),
        timing={**metadata["timing"], **result["timing"]},
        per_image=result["rows"],
    )


def save_results(output, report, examples):
    """Write the aggregate report, per-image CSV and optional example arrays."""
    write_json(output / "report.json", report)
    columns = ["id", "index"]
    for space, metrics in report["metrics"].items():
        columns.extend(f"{space}_{name}" for name in metrics)
    with (output / "per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in report["per_image"]:
            flattened = dict(id=row["id"], index=row["index"])
            for space in ("raw", "clipped"):
                for name, value in row[space].items():
                    flattened[f"{space}_{name}"] = value
            writer.writerow(flattened)
    if report["example_count"]:
        np.savez_compressed(
            output / "examples.npz",
            **{name: np.stack(values) for name, values in examples.items()},
        )
