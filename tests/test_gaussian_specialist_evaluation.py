"""Check spatial overlap behavior for informative and uninformative maps."""

import torch

from experiments.gaussian_specialist.evaluate import top_error_overlap


def test_top_error_overlap_distinguishes_matching_and_disjoint_pixels():
    width = torch.arange(100, dtype=torch.float32).reshape(1, 1, 10, 10)
    lower = torch.zeros_like(width)
    assert top_error_overlap(lower, width, width, lower) == [1.0]
    assert top_error_overlap(lower, width, width.flip((2, 3)), lower) == [0.0]


def test_top_error_overlap_rejects_constant_maps_and_averages_rgb():
    zero = torch.zeros(1, 3, 10, 10)
    ramp = torch.arange(100, dtype=torch.float32).reshape(1, 1, 10, 10)
    rgb = ramp.expand(1, 3, 10, 10)
    assert top_error_overlap(zero, zero, rgb, zero) == [None]
    assert top_error_overlap(zero, rgb, zero, zero) == [None]
    assert top_error_overlap(zero, rgb, rgb, zero) == [1.0]
