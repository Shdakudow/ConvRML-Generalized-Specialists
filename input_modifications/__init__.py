"""Independent modifications of raw measurement tensors, never scene targets.

Inputs and outputs use floating point (B, C, H, W) tensors in [0, 1].
See README.md for the physical interpretation and dataset preparation command.
"""

from .dead_pixels import apply_dead_pixels, sample_dead_pixel_mask
from .exposure import change_exposure
from .fixed_pattern import FixedPattern, apply_fixed_pattern, sample_fixed_pattern
from .gaussian import gaussian_noise
from .gaussian_blur import gaussian_sensor_blur
from .jpeg import jpeg_compression
from .poisson import poisson_noise
from .poisson_gaussian import poisson_gaussian_noise
from .resolution import reduce_sensor_resolution
from .online import apply_condition, sample_training_modification

__all__ = [
    "gaussian_noise",
    "poisson_noise",
    "poisson_gaussian_noise",
    "change_exposure",
    "sample_dead_pixel_mask",
    "apply_dead_pixels",
    "FixedPattern",
    "sample_fixed_pattern",
    "apply_fixed_pattern",
    "gaussian_sensor_blur",
    "reduce_sensor_resolution",
    "jpeg_compression",
    "sample_training_modification",
    "apply_condition",
]
