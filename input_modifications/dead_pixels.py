"""Persistent missing sensor locations, shared across frames and RGB channels."""

from typing import Optional

import torch

from ._validation import check_image, check_nonnegative


def sample_dead_pixel_mask(
    y: torch.Tensor,
    drop_rate: float = 0.10,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Choose round(drop_rate*H*W) dead locations; return bool shape (1,1,H,W).

    True means dead. Sample once for a simulated sensor and reuse across frames.
    Ten percent is a strong missing-data stress test, not a modest defect rate.
    """
    check_image(y)
    check_nonnegative(drop_rate, "drop_rate")
    if drop_rate > 1:
        raise ValueError("drop_rate must be in [0, 1].")
    height, width = y.shape[-2:]
    count = round(drop_rate * height * width)
    positions = torch.randperm(height * width, device=y.device, generator=generator)
    mask = torch.zeros(height * width, dtype=torch.bool, device=y.device)
    mask[positions[:count]] = True
    return mask.reshape(1, 1, height, width)


def apply_dead_pixels(y: torch.Tensor, dead_mask: torch.Tensor) -> torch.Tensor:
    """Zero the selected RGB locations; preserve every unselected pixel exactly."""
    check_image(y)
    if (
        dead_mask.dtype != torch.bool
        or dead_mask.shape != (1, 1, *y.shape[-2:])
        or dead_mask.device != y.device
    ):
        raise ValueError("dead_mask must be bool (1,1,H,W) on the image device.")
    return y.masked_fill(dead_mask, 0)
