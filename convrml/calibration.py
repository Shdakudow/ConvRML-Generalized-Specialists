"""Paper-based calibration of frozen models on a separate held-out split.

Baselines use image-level Hoeffding--Bentkus RCPS with separate endpoint scales.
QUTCC uses Algorithm 1 to select lower/upper quantile inputs. Neither procedure
updates model weights or forces an exact coverage fraction on future test images.
"""

import math
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader

from scipy.special import xlogy
from scipy.stats import binom

from .protocol import settings


class CalibrationFailure(ValueError):
    """A frozen model or calibration set cannot satisfy the requested procedure."""


def hb_pvalue(risk, n, alpha):
    """HB p-value for mean image-level miss fraction, using n independent images."""
    if not 0 <= risk <= 1 or n < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid risk, sample count or target")
    if risk >= alpha:
        return 1.0
    kl = xlogy(risk, risk / alpha) + xlogy(1 - risk, (1 - risk) / (1 - alpha))
    hoeffding = math.exp(-n * float(kl))
    bentkus = math.e * float(binom.cdf(math.ceil(n * risk), n, alpha))
    return min(1.0, hoeffding, bentkus)


def risk_threshold(n, alpha=0.1, delta=0.05):
    """Largest empirical tail risk accepted by the image-level HB test."""
    if not 0 < delta < 1:
        raise ValueError("Invalid confidence failure probability")
    if hb_pvalue(0.0, n, alpha) >= delta:
        raise CalibrationFailure(
            "Insufficient calibration images for a finite HB certificate"
        )
    accepted, rejected = 0.0, alpha
    for _ in range(80):
        candidate = (accepted + rejected) / 2
        if hb_pvalue(candidate, n, alpha) < delta:
            accepted = candidate
        else:
            rejected = candidate
    return max(0.0, accepted - 1e-10)


def rcps_scales(lower_scores, upper_scores, alpha=0.1, delta=0.05):
    """Fit separate lower/upper scales with image-level RCPS-HB certificates.

    Scores are dimensionless arrays (N, ...), one row per image. The sample count is the
    number of images, not pixels. Alpha and delta are split equally between tails.
    Scales multiply native distances from the center; they do not rescale images.
    """
    if lower_scores.shape != upper_scores.shape or lower_scores.ndim < 2:
        raise ValueError("Expected matching image-by-pixel lower/upper scores")
    if not 0 < alpha < 1 or not 0 < delta < 1:
        raise ValueError("Invalid risk or confidence budget")
    n = len(lower_scores)
    tail_alpha, tail_delta = alpha / 2, delta / 2
    threshold = risk_threshold(n, tail_alpha, tail_delta)
    scales, losses = {}, {}
    for side, scores in (("lower", lower_scores), ("upper", upper_scores)):
        if not all(np.isfinite(image).all() for image in scores):
            raise ValueError("Nonfinite calibration scores")
        selected = np.quantile(scores, 1 - threshold, method="higher")
        scales[side] = math.ceil(max(0.0, float(selected)) * 1e6) / 1e6
        losses[side] = np.array([np.mean(image > scales[side]) for image in scores])
        if hb_pvalue(float(losses[side].mean()), n, tail_alpha) >= tail_delta:
            raise CalibrationFailure("Failed " + side + " RCPS certificate")
    total_risk = float(
        np.mean(
            [
                np.mean((lower > scales["lower"]) | (upper > scales["upper"]))
                for lower, upper in zip(lower_scores, upper_scores)
            ]
        )
    )
    certificate = dict(
        n_images=n,
        alpha=alpha,
        delta=delta,
        tail_alpha=tail_alpha,
        tail_delta=tail_delta,
        maximum_accepted_tail_empirical_risk=threshold,
        tail_risks={side: float(value.mean()) for side, value in losses.items()},
        total_risk=total_risk,
        confidence_policy="Bonferroni: delta/2 per tail, joint failure <= delta",
        tail_hb_pvalues={
            side: hb_pvalue(float(value.mean()), n, tail_alpha)
            for side, value in losses.items()
        },
    )
    return dict(
        scales=scales,
        certificate=certificate,
        delta=delta,
        procedure="asymmetric RCPS-HB",
    )


def tail_budget(alpha, n):
    """Per-tail miss budget from QUTCC v2 Algorithm 1, lines 1-2.

    Use (alpha - (1-alpha)/n)/2 as written in the pseudocode. The adjacent
    paper prose gives a different correction; this implementation follows Algorithm 1.
    """
    if n < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid calibration size or alpha")
    budget = (alpha - (1 - alpha) / n) / 2
    if not 0 < budget < 0.5:
        raise ValueError("Calibration set too small for alpha")
    return budget


def paper_quantiles(risk, n, alpha=0.1, tol=1e-3, max_iterations=20):
    """QUTCC Algorithm 1: binary-search global q endpoints with finite-N correction.

    ``risk(lower_q, upper_q)`` returns lower/upper miss fractions in [0,1].
    These q values are shared by all images; the predicted images vary spatially.
    The search assumes monotone quantile responses, as in the original algorithm.
    """
    if tol <= 0 or max_iterations < 1:
        raise ValueError("Quantile tolerance and iteration count must be positive")
    target = tail_budget(alpha, n)
    trace = []

    def check(lower_q, upper_q):
        lower_risk, upper_risk = risk(lower_q, upper_q)
        if not 0 <= lower_risk <= 1 or not 0 <= upper_risk <= 1:
            raise ValueError("Invalid calibration risk")
        trace.append(
            dict(
                lower_q=lower_q,
                upper_q=upper_q,
                lower_risk=lower_risk,
                upper_risk=upper_risk,
            )
        )
        return lower_risk, upper_risk

    lower_risk, upper_risk = check(0.0, 1.0)
    if lower_risk > target or upper_risk > target:
        raise CalibrationFailure(
            "No feasible extreme quantile bounds; inspect model fit/crossing"
        )
    lower_q_min, lower_q_max = 0.0, 1.0
    upper_q_min, upper_q_max = 0.0, 1.0
    valid_lower_q, valid_upper_q = 0.0, 1.0
    for _ in range(max_iterations):
        lower_q = (lower_q_min + lower_q_max) / 2
        upper_q = (upper_q_min + upper_q_max) / 2
        lower_risk, upper_risk = check(lower_q, upper_q)
        # Increasing lower_q tightens the lower endpoint; decreasing upper_q
        # tightens the upper endpoint. Retain the last feasible value on each side.
        if lower_risk > target:
            lower_q_max = lower_q
        else:
            lower_q_min = lower_q
            valid_lower_q = lower_q
        if upper_risk > target:
            upper_q_min = upper_q
        else:
            upper_q_max = upper_q
            valid_upper_q = upper_q
        if lower_q_max - lower_q_min < tol and upper_q_max - upper_q_min < tol:
            break
    if valid_lower_q > valid_upper_q:
        raise CalibrationFailure("Calibrated quantile levels reversed")
    lower_risk, upper_risk = check(valid_lower_q, valid_upper_q)
    if lower_risk > target or upper_risk > target:
        raise CalibrationFailure("Final bounds failed calibration recheck")
    return dict(
        lower_q=valid_lower_q,
        upper_q=valid_upper_q,
        alpha=alpha,
        per_tail_target=target,
        n_images=n,
        tolerance=tol,
        trace=trace,
        procedure="QUTCC paper Algorithm 1 calibrated quantile search",
        assumption="Quantile response should be monotone; shared weights do not strictly enforce it",
        calibrated_on="calibration split only",
    )


def capture_rng(device):
    return torch.get_rng_state(), (
        torch.cuda.get_rng_state_all() if device.type == "cuda" else []
    )


def restore_rng(state):
    torch.set_rng_state(state[0])
    if state[1]:
        torch.cuda.set_rng_state_all(state[1])


def fit_calibration(calibration, predict, q_model, run, device, workers, output):
    """Fit on calibration pairs, then audit the fitted intervals on the same pairs.

    DataLoader batches are float (B,2,3,H,W): measurement, then target in [0,1].
    Native predict() returns center/lower/upper in unbounded intensity units.
    Baselines fit two dimensionless scales; QUTCC fits two scalar q inputs.
    Model weights remain frozen and test images are not read. Endpoints are
    neither clipped nor sorted by this fitting/audit workflow.
    """
    loader = DataLoader(calibration, batch_size=run["microbatch"], num_workers=workers)
    target_risk = settings()["alpha"]
    rng_before, rng_after = capture_rng(device), None

    # 1. Fit the method's interval parameters on calibration pairs only.
    if run["method"] == "qutcc":
        fitted = _fit_quantile_inputs(
            loader, q_model, device, len(calibration), target_risk
        )
    else:
        fitted = _fit_baseline_scales(
            calibration, loader, predict, device, output, target_risk
        )
        rng_after = capture_rng(device)
        # The audit must replay the same MC masks used to fit the scales.
        restore_rng(rng_before)

    # 2. Measure coverage/crossings, then preserve the post-fit inference RNG.
    covered, total, crossing = _audit_calibration(
        loader, predict, q_model, run["method"], fitted, device
    )
    if rng_after is not None:
        restore_rng(rng_after)
    coverage = covered / total
    fitted.update(
        calibration_coverage=coverage,
        calibration_crossing=crossing / total,
        target_coverage=1 - target_risk,
        n_images=len(calibration),
        calibrated_on="calibration split only",
        endpoint_clipping=None,
        has_hb_certificate="certificate" in fitted,
        calibration_mode="paper",
    )
    return fitted


def _fit_quantile_inputs(loader, q_model, device, n_images, target_risk):
    """Evaluate tail misses for Algorithm 1; cache each side/q result separately.

    The callback returns lower/upper miss fractions across all RGB values.
    Caching avoids repeating a full calibration pass for an already tested q.
    """
    cache = {"lower": {}, "upper": {}}

    def risk(lower_q, upper_q):
        quantiles = {"lower": lower_q, "upper": upper_q}
        missing = [side for side in quantiles if quantiles[side] not in cache[side]]
        misses, pixels = {"lower": 0, "upper": 0}, 0
        if missing:
            for batch in loader:
                measurement, target = batch[:, 0].to(device), batch[:, 1].to(device)
                for side in missing:
                    q = measurement.new_full((len(measurement),), quantiles[side])
                    endpoint = q_model(measurement, q)
                    if not torch.isfinite(endpoint).all():
                        raise ValueError("Nonfinite quantile prediction")
                    if side == "lower":
                        misses[side] += int((target < endpoint).sum())
                    else:
                        misses[side] += int((target > endpoint).sum())
                pixels += target.numel()
            for side in missing:
                cache[side][quantiles[side]] = misses[side] / pixels
        return cache["lower"][lower_q], cache["upper"][upper_q]

    return paper_quantiles(risk, n_images, alpha=target_risk)


def _fit_baseline_scales(calibration, loader, predict, device, output, target_risk):
    """Collect normalized residuals on disk and fit separate RCPS-HB tail scales.

    Memmaps retain one row per image without keeping the prediction set on the
    GPU. Temporary score files are closed and removed before returning the fit.
    """
    shape = (len(calibration), *calibration[0][0].shape)
    # Disk-backed scores keep full-resolution calibration off the GPU and
    # avoid retaining every calibration prediction in RAM.
    with tempfile.TemporaryDirectory(prefix="calibration-", dir=output) as temporary:
        scores = {
            side: np.lib.format.open_memmap(
                Path(temporary) / f"tail_{index}.npy",
                mode="w+",
                dtype=np.float32,
                shape=shape,
            )
            for index, side in enumerate(("lower", "upper"))
        }
        offset = 0
        try:
            for batch in loader:
                measurement, target = batch[:, 0].to(device), batch[:, 1].to(device)
                center, lower, upper = predict(measurement)
                # A target misses a calibrated endpoint exactly when this
                # normalized residual exceeds that endpoint's scale factor.
                lower_score = (center - target) / (center - lower).clamp_min(1e-6)
                upper_score = (target - center) / (upper - center).clamp_min(1e-6)
                scores["lower"][
                    offset : offset + len(batch)
                ] = lower_score.cpu().numpy()
                scores["upper"][
                    offset : offset + len(batch)
                ] = upper_score.cpu().numpy()
                offset += len(batch)
            fitted = rcps_scales(
                scores["lower"],
                scores["upper"],
                alpha=target_risk,
                delta=settings()["calibration"]["delta"],
            )
        finally:
            for array in scores.values():
                array._mmap.close()
    return fitted


def _audit_calibration(loader, predict, q_model, method, fitted, device):
    """Return counts of covered RGB values, all RGB values and crossed bounds."""
    covered, total, crossing = 0, 0, 0
    for batch in loader:
        measurement, target = batch[:, 0].to(device), batch[:, 1].to(device)
        _, lower, upper = apply_calibration(
            measurement, predict, q_model, method, fitted
        )
        covered += int(((target >= lower) & (target <= upper)).sum())
        crossing += int((lower > upper).sum())
        total += target.numel()
    return covered, total, crossing


def apply_calibration(y, predict, q_model, method, fitted):
    """Return calibrated (center, lower, upper), each (N,3,H,W), without clipping.

    ``y`` is a measurement batch in [0,1]. Frozen q inputs change QUTCC endpoints;
    baseline scales multiply distances from the unchanged center image.
    """
    if method == "qutcc" and "lower_q" in fitted:
        return q_model.bounds(y, fitted["lower_q"], fitted["upper_q"])
    center, lower, upper = predict(y)
    if "scales" in fitted:
        lower = center - fitted["scales"]["lower"] * (center - lower).clamp_min(1e-6)
        upper = center + fitted["scales"]["upper"] * (upper - center).clamp_min(1e-6)
    return center, lower, upper
