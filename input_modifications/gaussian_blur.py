"""Gaussian filtering of the recorded sensor plane, independently per channel."""

import torch
import torch.nn.functional as F

from ._validation import check_image, check_nonnegative


def gaussian_sensor_blur(
    y: torch.Tensor, sigma: float = 0.5, radius: int = 2
) -> torch.Tensor:
    """Apply a normalized (2*radius+1)-square Gaussian with replicated borders.

    Sigma is in sensor-image pixels. This models measurement filtering, not a
    newly simulated optical PSF, diffuser displacement, or object-plane defocus.
    """
    check_image(y)
    check_nonnegative(sigma, "sigma")
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 1:
        raise ValueError("radius must be a positive integer.")
    if sigma == 0:
        return y.clone()
    # Float32 accumulation also supports low-precision input tensors on CPU.
    work = y.float() if y.dtype in (torch.float16, torch.bfloat16) else y
    positions = torch.arange(-radius, radius + 1, dtype=work.dtype, device=y.device)
    weights = torch.exp(-0.5 * (positions / sigma).square())
    weights = weights / weights.sum()
    kernel_2d = torch.outer(weights, weights)
    channels = y.shape[1]
    kernel = kernel_2d[None, None].expand(channels, 1, -1, -1).contiguous()
    padded = F.pad(work, (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(padded, kernel, groups=channels).clamp(0, 1).to(y.dtype)
