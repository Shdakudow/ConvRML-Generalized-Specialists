"""Simultaneous shot and read noise, with one final sensor-range clipping step."""

from typing import Optional

import torch

from ._validation import check_image, check_nonnegative


def poisson_gaussian_noise(
    y: torch.Tensor,
    gaussian_sigma: float = 0.01,
    peak_photons: float = 1330.0,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Return clip(Poisson(P*y)/P + N(0, gaussian_sigma**2)).

    The pre-clipping conditional variance is y/P + gaussian_sigma**2.
    Do not call the two individually clipped noise helpers in sequence: clipping
    between the draws would change the combined noise model near the boundaries.
    """
    check_image(y)
    check_nonnegative(gaussian_sigma, "gaussian_sigma")
    check_nonnegative(peak_photons, "peak_photons")
    if peak_photons == 0:
        raise ValueError("peak_photons must be positive.")
    shot = torch.poisson(y * peak_photons, generator=generator) / peak_photons
    read = torch.randn(y.shape, dtype=y.dtype, device=y.device, generator=generator)
    return (shot + gaussian_sigma * read).clamp(0, 1)
