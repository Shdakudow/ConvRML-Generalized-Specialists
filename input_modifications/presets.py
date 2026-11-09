"""Shared settings for prepared corruptions and online training conditions.

PRESETS records parameter values; each named module implements one transform.
TRANSFORMS binds stateless presets for dataset preparation. Persistent sensor
masks/patterns are handled explicitly, and online.py exposes apply_condition.
Protocol snapshots record these values separately as immutable experiment evidence.
"""

from functools import partial

from .exposure import change_exposure
from .gaussian import gaussian_noise
from .gaussian_blur import gaussian_sensor_blur
from .jpeg import jpeg_compression
from .poisson import poisson_noise
from .poisson_gaussian import poisson_gaussian_noise
from .resolution import reduce_sensor_resolution

PRESETS = {
    "gaussian": {"sigma": 0.01},
    "poisson": {"peak_photons": 1330.0},
    "poisson_gaussian": {"gaussian_sigma": 0.01, "peak_photons": 1330.0},
    "exposure_050": {"gain": 0.50},
    "exposure_150": {"gain": 1.50},
    "dead_pixels": {"drop_rate": 0.10},
    "fixed_pattern": {"gain_std": 0.01, "offset_std": 0.003},
    "gaussian_blur": {"sigma": 0.5, "radius": 2},
    "resolution": {"factor": 64},
    "jpeg": {"quality": 50, "subsampling": 0},
}

# Persistent masks/patterns are initialized once per sensor by the dataset builder.
TRANSFORMS = {
    "gaussian": partial(gaussian_noise, **PRESETS["gaussian"]),
    "poisson": partial(poisson_noise, **PRESETS["poisson"]),
    "poisson_gaussian": partial(poisson_gaussian_noise, **PRESETS["poisson_gaussian"]),
    "exposure_050": partial(change_exposure, **PRESETS["exposure_050"]),
    "exposure_150": partial(change_exposure, **PRESETS["exposure_150"]),
    "gaussian_blur": partial(gaussian_sensor_blur, **PRESETS["gaussian_blur"]),
    "resolution": partial(reduce_sensor_resolution, **PRESETS["resolution"]),
    "jpeg": partial(jpeg_compression, **PRESETS["jpeg"]),
}

RANDOM_PER_IMAGE = frozenset({"gaussian", "poisson", "poisson_gaussian"})
