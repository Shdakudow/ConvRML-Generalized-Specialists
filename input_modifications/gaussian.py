"""Signal-independent additive Gaussian measurement noise."""

from typing import Optional

import torch

from ._validation import check_image, check_nonnegative


def gaussian_noise(
    y: torch.Tensor,
    sigma: float = 0.01,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Return clip(y + N(0, sigma**2)); sigma is measured on the [0, 1] scale.

    Clipping changes noise statistics near zero and one. The source is not edited.
    An explicit generator makes a measurement reproducible without global seeds.
    """
    check_image(y)
    check_nonnegative(sigma, "sigma")
    if sigma == 0:
        return y.clone()
    noise = torch.randn(y.shape, dtype=y.dtype, device=y.device, generator=generator)
    return (y + sigma * noise).clamp(0, 1)
