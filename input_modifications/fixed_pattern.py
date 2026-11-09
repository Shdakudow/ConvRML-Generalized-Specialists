"""Persistent sensor gain and offset variations, separate from temporal noise."""

from dataclasses import dataclass
from typing import Optional

import torch

from ._validation import check_image, check_nonnegative


@dataclass(frozen=True)
class FixedPattern:
    """Per-channel pixel gains and offsets, each with shape (1,C,H,W)."""

    gain: torch.Tensor
    offset: torch.Tensor


def sample_fixed_pattern(
    y: torch.Tensor,
    gain_std: float = 0.01,
    offset_std: float = 0.003,
    *,
    generator: Optional[torch.Generator] = None,
) -> FixedPattern:
    """Sample one sensor: gain~N(1,gain_std**2), offset~N(0,offset_std**2).

    Gain SD is relative (0.01 means 1%); offset SD uses the [0,1] intensity scale.
    Reuse this pattern across the sensor's images rather than resampling it.
    """
    check_image(y)
    check_nonnegative(gain_std, "gain_std")
    check_nonnegative(offset_std, "offset_std")
    shape = (1, *y.shape[1:])
    gain = torch.randn(shape, dtype=y.dtype, device=y.device, generator=generator)
    offset = torch.randn(shape, dtype=y.dtype, device=y.device, generator=generator)
    return FixedPattern((1 + gain_std * gain).clamp_min(0), offset_std * offset)


def apply_fixed_pattern(y: torch.Tensor, pattern: FixedPattern) -> torch.Tensor:
    """Return clip(y*gain + offset) using a previously sampled sensor pattern."""
    check_image(y)
    for field in (pattern.gain, pattern.offset):
        if (
            field.shape != (1, *y.shape[1:])
            or field.dtype != y.dtype
            or field.device != y.device
            or not torch.isfinite(field).all()
        ):
            raise ValueError(
                "Pattern fields must be finite matching (1,C,H,W) tensors."
            )
    if torch.any(pattern.gain < 0):
        raise ValueError("Sensor gains must be nonnegative.")
    return (y * pattern.gain + pattern.offset).clamp(0, 1)
