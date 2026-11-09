"""Independent Gaussian-NLL ensemble for RGB measurements of variable size.

The model accepts floating (N,3,H,W) measurements in [0,1], with a matching target.
The loader handles file format and intensity conversion. Default members pad to
multiples of 32 internally and crop both mean and variance back to H/W.
Each member has the shared ConvNeXt/FPN, RGB mean and RGB variance heads.
All members see the entire training split with independent initialization and
shuffle seeds. Default comparison uses five members; ten is a separate reference.
Checkpoint keys retain ``baseModel.*`` for each independently trained member.
Existing PLD runs use 256x256 images; that is an experiment setting.
Means/bounds are unbounded normalized intensities; variances have squared units.
"""

import torch
from torch import nn
from torch.nn import functional as F

from convrml.backbone import ConvNeXtFPN
from convrml.intervals import gaussian_interval_from_mean_variance


class _GaussianBackbone(ConvNeXtFPN):
    def __init__(self):
        super().__init__()
        self.variance_head = nn.Conv2d(256, 3, 3, padding=1)
        nn.init.zeros_(self.variance_head.weight)
        nn.init.constant_(self.variance_head.bias, -4.6)

    def forward(self, y):
        height, width = y.shape[-2:]
        if height % 32 or width % 32:
            y = F.pad(y, (0, (-width) % 32, 0, (-height) % 32), mode="replicate")
        features = self.up2(self.features(y))
        mean = y + self.head(features)
        return torch.cat((mean, self.variance_head(features)), 1)[..., :height, :width]


class EnsembleMember(nn.Module):
    def __init__(self, base_model=None, *, channels=3, min_variance=1e-6):
        super().__init__()
        self.baseModel = _GaussianBackbone() if base_model is None else base_model
        self.channels = channels
        self.min_variance = min_variance
        if min_variance <= 0:
            raise ValueError("min_variance must be positive")

    def forward(self, x):
        """Return (mean, variance), each (N,3,H,W); variance is strictly positive."""
        raw = self.baseModel(x)
        if raw.shape[1] != 2 * self.channels:
            raise ValueError("Member must output C means and C raw variances")
        return (
            raw[:, : self.channels],
            F.softplus(raw[:, self.channels :]) + self.min_variance,
        )

    def loss(self, x, target):
        """Mean Gaussian NLL, omitting the constant 0.5*log(2*pi)."""
        mean, variance = self(x)
        mean, variance = mean.float(), variance.float()
        return 0.5 * (variance.log() + (target - mean).square() / variance).mean()


class DeepEnsemble(nn.Module):
    def __init__(self, members, *, base_interval_alpha=0.1):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.base_interval_alpha = base_interval_alpha
        if not len(self.members):
            raise ValueError("Ensemble cannot be empty")

    def moments(self, x):
        """Return mixture mean and total variance, each (N,3,H,W)."""
        # E[variance + mean^2] - E[mean]^2 includes both within-member noise
        # estimates and between-member disagreement. Keep this operation order
        # to preserve the saved evaluations' floating-point behavior.
        mean_sum = second_moment_sum = None
        for member in self.members:
            mean, variance = member(x)
            mean, variance = mean.float(), variance.float()
            mean_sum = mean if mean_sum is None else mean_sum + mean
            second_moment_sum = (
                variance + mean.square()
                if second_moment_sum is None
                else second_moment_sum + variance + mean.square()
            )
        mean = mean_sum / len(self.members)
        variance = (second_moment_sum / len(self.members) - mean.square()).clamp_min(0)
        return mean, variance

    def forward(self, x):
        """Return concatenated lower/mean/upper RGB images, shape (N,9,H,W)."""
        mean, variance = self.moments(x)
        lo, hi = gaussian_interval_from_mean_variance(
            mean, variance, self.base_interval_alpha
        )
        return torch.cat([lo, mean, hi], dim=1)

    def bounds(self, y):
        """Return (mean, lower, upper), each (N,3,H,W), before calibration.

        Bounds use a moment-matched Gaussian, not exact Gaussian-mixture quantiles.
        """
        mean, variance = self.moments(y)
        lower, upper = gaussian_interval_from_mean_variance(
            mean, variance, self.base_interval_alpha
        )
        return mean, lower, upper
