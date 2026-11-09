"""Explicit JPEG encode/decode of prepared grayscale or RGB sensor images."""

from io import BytesIO

import numpy as np
from PIL import Image
import torch

from ._validation import check_image


def jpeg_compression(
    y: torch.Tensor, quality: int = 50, subsampling: int = 0
) -> torch.Tensor:
    """Round-trip through 8-bit JPEG; default 4:4:4 avoids chroma downsampling.

    This CPU/Pillow data-preparation operation is not differentiable. JPEG quality
    is an encoder setting, not a retained-information percentage. Output tensors
    retain the input shape, device and floating dtype. Store decoded outputs as PNG
    to avoid an accidental second JPEG compression during dataset materialization.
    """
    check_image(y)
    if y.shape[1] not in (1, 3):
        raise ValueError("JPEG supports only grayscale (C=1) or RGB (C=3).")
    if (
        isinstance(quality, bool)
        or not isinstance(quality, int)
        or not 1 <= quality <= 95
    ):
        raise ValueError("quality must be an integer from 1 to 95.")
    if (
        isinstance(subsampling, bool)
        or not isinstance(subsampling, int)
        or subsampling not in (0, 1, 2)
    ):
        raise ValueError("subsampling must be 0 (4:4:4), 1 (4:2:2), or 2 (4:2:0).")
    mode = "L" if y.shape[1] == 1 else "RGB"
    outputs = []
    for sample in y.detach().cpu():
        pixels = sample.mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
        if mode == "L":
            pixels = pixels[:, :, 0]
        with BytesIO() as buffer:
            Image.fromarray(pixels).save(
                buffer, format="JPEG", quality=quality, subsampling=subsampling
            )
            buffer.seek(0)
            with Image.open(buffer) as encoded:
                decoded = np.array(encoded.convert(mode), copy=True)
        if decoded.ndim == 2:
            decoded = decoded[:, :, None]
        result = torch.from_numpy(decoded).permute(2, 0, 1)
        outputs.append(result.to(device=y.device, dtype=y.dtype) / 255)
    return torch.stack(outputs)
