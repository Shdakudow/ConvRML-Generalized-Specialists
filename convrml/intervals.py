"""Native Gaussian-moment intervals, before fitting calibration scales."""

import torch


def gaussian_interval_from_mean_variance(mean, variance, alpha=0.1):
    """Return two tensors shaped like mean/variance, normally (N,3,H,W).

    Mean has normalized-intensity units; nonnegative variance has squared units.
    Alpha=0.1 uses the saved 90% multiplier. This is a Gaussian approximation,
    not calibration, and neither bounds nor means are clipped to [0,1].
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if alpha == 0.1:
        z = 1.6448536269514722
    else:
        z = torch.distributions.Normal(mean.new_tensor(0.0), mean.new_tensor(1.0)).icdf(
            mean.new_tensor(1 - alpha / 2)
        )
    radius = z * variance.sqrt()
    return mean - radius, mean + radius
