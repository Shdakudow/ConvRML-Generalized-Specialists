"""Deterministic scaling of an existing sensor measurement."""

import torch

from ._validation import check_image, check_nonnegative


def change_exposure(y: torch.Tensor, gain: float = 0.50) -> torch.Tensor:
    """Return clip(gain*y) without changing the paired scene target.

    One global gain applies to all pixels and channels in an image. This tests
    brightness/gain mismatch. Scaling an already recorded measurement
    also scales its existing noise; it does not simulate a fresh acquisition at
    a different exposure time with newly sampled photon and read noise.
    """
    check_image(y)
    check_nonnegative(gain, "gain")
    return (y * gain).clamp(0, 1)
