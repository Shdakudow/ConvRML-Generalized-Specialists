"""QUTCC on the ConvNeXt/FPN backbone, shared by PLD and DLMD.

The model accepts RGB (N,3,H,W); image loading and normalization are separate.
Existing PLD runs use 256x256 and DLMD runs use 270x480, both divided by 255.
Inputs are replicate-padded to multiples of 32; outputs are cropped to input size.
One scalar q per image conditions 36 encoder and 12 decoder sites through FiLM.
Only sampled-q pinball is optimized; there is no additional MSE or q selector.
Measurements/targets are floating tensors in [0,1]. Predictions are unbounded
normalized intensities; increasing q is not guaranteed to increase every pixel.
"""

import torch
from torch import nn
from torch.nn import functional as F

from convrml.backbone import ConvNeXtFPN
from convrml.film import FiLMBackbone


def pinball(prediction, target, q):
    """Return a scalar from (N,3,H,W) predictions/targets and q of shape (N,).

    The same q conditions the network and weights its asymmetric prediction error.
    """
    error = target - prediction
    q = q.reshape(-1, 1, 1, 1)
    return torch.maximum(q * error, (q - 1) * error).mean()


class QUTCC(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = FiLMBackbone(ConvNeXtFPN())

    def forward(self, y, q=None):
        """Map y (N,3,H,W), q (N,) in [0,1] to an unbounded RGB quantile image.

        Output shape matches y. Default q=0.5 estimates the conditional median.
        """
        h, w = y.shape[-2:]
        padded = F.pad(y, (0, (-w) % 32, 0, (-h) % 32), mode="replicate")
        if q is None:
            q = y.new_full((len(y),), 0.5)
        return self.net(padded, q)[..., :h, :w]

    def center(self, y):
        return self(y)

    def bounds(self, y, lower_q=0.05, upper_q=0.95):
        """Return (median, lower, upper), each (N,3,H,W), without clipping.

        Scalar endpoints are shared across images; calibration selects new q inputs.
        """
        if not 0 <= lower_q <= upper_q <= 1:
            raise ValueError("Expected ordered q values in [0, 1]")
        return (
            self(y),
            self(y, y.new_full((len(y),), lower_q)),
            self(y, y.new_full((len(y),), upper_q)),
        )

    def loss(self, y, target, *, generator=None):
        """Sample q ~ Uniform(0,1) independently per image and minimize pinball."""
        q = torch.rand(
            len(y), device=y.device, dtype=torch.float32, generator=generator
        )
        return pinball(self(y, q), target, q)
