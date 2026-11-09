"""Signal-dependent Poisson shot-noise approximation for prepared measurements."""

from typing import Optional

import torch

from ._validation import check_image, check_nonnegative


def poisson_noise(
    y: torch.Tensor,
    peak_photons: float = 1330.0,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Return clip(Poisson(P*y) / P), with P=peak_photons.

    Before clipping, E[output]=y and Var[output]=y/P. P=1330 gives approximately
    0.01 RMS noise when mean(y)=0.133; it does not give constant per-pixel SD.
    P is an effective count scale, not a calibrated photon measurement.
    """
    check_image(y)
    check_nonnegative(peak_photons, "peak_photons")
    if peak_photons == 0:
        raise ValueError("peak_photons must be positive.")
    counts = torch.poisson(y * peak_photons, generator=generator)
    return (counts / peak_photons).clamp(0, 1)
