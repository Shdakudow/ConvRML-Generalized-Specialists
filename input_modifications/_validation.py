"""Shared tensor and scalar checks for the individual modification modules."""

import math

import torch


def check_image(y: torch.Tensor) -> None:
    """Require a nonempty, finite BCHW floating tensor with intensities in [0, 1]."""
    if not isinstance(y, torch.Tensor) or y.ndim != 4:
        raise ValueError("Expected a torch.Tensor with shape (B, C, H, W).")
    if not y.is_floating_point() or any(size < 1 for size in y.shape):
        raise ValueError("Images must be nonempty floating point tensors.")
    if not torch.isfinite(y).all() or torch.any(y < 0) or torch.any(y > 1):
        raise ValueError("Image intensities must be finite and in [0, 1].")


def check_nonnegative(value: float, name: str) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
