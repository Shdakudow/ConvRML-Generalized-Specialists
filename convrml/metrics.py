"""Reconstruction quality, interval quality, and uncertainty/error correlations."""

import numpy as np
import torch
from scipy.stats import rankdata
from torchmetrics.functional.image import structural_similarity_index_measure


class PerceptualMetric:
    """Frozen LPIPS-AlexNet on clipped RGB [0,1], with one score per image.

    Construction preserves the model's random stream, including MC dropout.
    Official pretrained weights may download into the standard torch cache.
    """

    def __init__(self, device):
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        with torch.random.fork_rng(devices=[]):
            self.metric = (
                LearnedPerceptualImagePatchSimilarity(
                    net_type="alex", normalize=True, reduction="none"
                )
                .to(device)
                .eval()
                .requires_grad_(False)
            )

    @torch.inference_mode()
    def __call__(self, prediction, target):
        if min(prediction.shape[-2:]) < 64:
            raise ValueError(
                "LPIPS-AlexNet requires images at least 64x64; use --no-lpips for tiny fixtures"
            )
        try:
            return self.metric(prediction.float().clamp(0, 1), target.float()).reshape(
                -1
            )
        finally:
            self.metric.reset()


def _pearson(first, second):
    first, second = first - first.mean(), second - second.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return (
        float(np.clip(np.dot(first, second) / denominator, -1, 1))
        if denominator > 0
        else np.nan
    )


def uncertainty_error_correlations(width, error):
    """Per-image Pearson/Spearman across flattened RGB values, with average ties.

    Constant maps yield NaN internally; evaluation serializes them as null and
    reports valid-image counts. No smoothing, regional aggregation or p-values.
    """
    if width.shape != error.shape or width.ndim != 4:
        raise ValueError("Expected matching (N,C,H,W) width and absolute-error maps")
    result = {"pearson": [], "spearman": []}
    for uncertainty_map, error_map in zip(
        width.detach().double().cpu().numpy(), error.detach().double().cpu().numpy()
    ):
        uncertainty_values = uncertainty_map.reshape(-1)
        error_values = error_map.reshape(-1)
        valid = np.isfinite(uncertainty_values) & np.isfinite(error_values)
        uncertainty_values, error_values = (
            uncertainty_values[valid],
            error_values[valid],
        )
        pearson = spearman = np.nan
        if len(uncertainty_values) >= 2:
            pearson = _pearson(uncertainty_values, error_values)
            spearman = _pearson(
                rankdata(uncertainty_values, method="average"),
                rankdata(error_values, method="average"),
            )
        result["pearson"].append(pearson)
        result["spearman"].append(spearman)
    return {
        key: torch.tensor(values, dtype=torch.float64, device=width.device)
        for key, values in result.items()
    }


def finite_mean(values):
    """A JSON-safe mean and valid count; undefined statistics never become zero."""
    finite = [
        float(value) for value in values if value is not None and np.isfinite(value)
    ]
    return (sum(finite) / len(finite) if finite else None), len(finite)


def reconstruction_metrics(prediction, target):
    """Return (N,) PSNR/SSIM/MAE from matching (N,3,H,W) images.

    Targets are in [0,1], so PSNR/SSIM use intensity range 1 even for unbounded
    predictions. Clipping, when desired, is done by the caller.
    """
    prediction, target = prediction.float(), target.float()
    mse = (prediction - target).square().mean((1, 2, 3))
    return dict(
        psnr=-10 * mse.clamp_min(1e-12).log10(),
        ssim=structural_similarity_index_measure(
            prediction, target, data_range=1.0, reduction="none"
        ).reshape(-1),
        mae=(prediction - target).abs().mean((1, 2, 3)),
    )


def interval_metrics(prediction, lower, upper, target, *, clipped):
    """Return per-image (N,) scores from matching (N,3,H,W) tensors.

    Coverage is the fraction of RGB values inside inclusive bounds; width is
    mean UB-LB in normalized-intensity units. ``clipped`` clamps predictions and
    both endpoints to [0,1] for this metric view, without changing calibration.
    Crossed bounds are reported and remain unsorted.
    """
    if clipped:
        prediction, lower, upper = [v.clamp(0, 1) for v in (prediction, lower, upper)]
    result = reconstruction_metrics(prediction, target)
    result.update(
        coverage=((target >= lower) & (target <= upper)).float().mean((1, 2, 3)),
        width=(upper - lower).float().mean((1, 2, 3)),
        crossing=(lower > upper).float().mean((1, 2, 3)),
    )
    result.update(
        uncertainty_error_correlations(upper - lower, (prediction - target).abs())
    )
    return result
