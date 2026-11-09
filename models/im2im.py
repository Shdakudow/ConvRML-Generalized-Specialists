"""One end-to-end Im2im model for prepared PLD and DLMD RGB measurements.

Inputs: floating RGB measurements (N,3,H,W) in [0,1]; target has the same shape.
Outputs: (N, 3 heads, 3 RGB channels, H, W), ordered lower / center / upper.
Predictions are unbounded normalized intensities; endpoint ordering is not enforced.
Difference from QUTCC: three fixed heads; no quantile input or FiLM modulation.
All datasets use the same ``backbone.*`` parameter names. The checkpoint loader
translates historical ``base.*`` and ``net.*`` names before strict weight loading.
"""

import torch
from torch import nn

from convrml.backbone import ConvNeXtFPN


def pinball(prediction, target, q):
    """Scalar mean loss for matching (N,3,H,W) tensors and q of shape (N,).

    Positive target-minus-prediction error is weighted by q; negative error by
    1-q. Thus the two endpoint heads estimate different conditional quantiles.
    """
    error = target - prediction
    q = q.reshape(-1, 1, 1, 1)
    return torch.maximum(q * error, (q - 1) * error).mean()


class Im2im(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ConvNeXtFPN(output_heads=3)

    def forward(self, y):
        """Return lower/center/upper RGB heads as (N,3,3,H,W) in one pass."""
        return self.backbone(y).reshape(len(y), 3, 3, *y.shape[-2:])

    def center(self, y):
        return self(y)[:, 1]

    def bounds(self, y):
        """Return (center, lower, upper), each unbounded (N,3,H,W)."""
        outputs = self(y)
        # Training stores heads as lower/center/upper; callers receive center first.
        return outputs[:, 1], outputs[:, 0], outputs[:, 2]

    def loss(self, y, target):
        """Train all parameters jointly: center MSE plus endpoint pinball losses."""
        # The center estimates a conditional mean; endpoints estimate the native
        # 90% interval. Equal coefficients are the recorded experiment setting.
        center, lower, upper = self.bounds(y)
        return (
            (center.float() - target).square().mean()
            + pinball(lower.float(), target, target.new_full((len(target),), 0.05))
            + pinball(upper.float(), target, target.new_full((len(target),), 0.95))
        )
