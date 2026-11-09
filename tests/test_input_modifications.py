"""Check sensor statistics, persistent defects, edge geometry and paired-data safety."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from input_modifications import (
    apply_dead_pixels,
    apply_fixed_pattern,
    change_exposure,
    gaussian_noise,
    gaussian_sensor_blur,
    jpeg_compression,
    poisson_gaussian_noise,
    poisson_noise,
    reduce_sensor_resolution,
    sample_dead_pixel_mask,
    sample_fixed_pattern,
)
from input_modifications.prepare_dataset import PRESETS, ROLES, prepare


def generator(seed=42):
    return torch.Generator().manual_seed(seed)


class SensorModifications(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_gaussian_sd_and_reproducibility_without_source_or_global_rng_changes(self):
        y = torch.full((1, 1, 512, 512), 0.5)
        original, rng = y.clone(), torch.get_rng_state().clone()
        a = gaussian_noise(y, generator=generator())
        b = gaussian_noise(y, generator=generator())
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(y, original, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertLess(abs(float((a - y).mean())), 0.0001)
        self.assertAlmostEqual(float((a - y).std()), 0.01, delta=0.0001)

    def test_poisson_sd_depends_on_signal_and_matches_the_count_scale(self):
        for intensity in (0.05, 0.2, 0.7):
            y = torch.full((1, 1, 512, 512), intensity)
            noise = poisson_noise(y, generator=generator()) - y
            self.assertLess(abs(float(noise.mean())), 0.0002)
            expected_variance = intensity / 1330
            self.assertAlmostEqual(
                float(noise.var()), expected_variance, delta=expected_variance * 0.025
            )

    def test_combined_noise_variance_and_single_final_clipping(self):
        y = torch.full((1, 1, 512, 512), 0.2)
        z = poisson_gaussian_noise(y, generator=generator())
        variance = 0.2 / 1330 + 0.01**2
        self.assertAlmostEqual(float((z - y).var()), variance, delta=variance * 0.025)
        # At zero, half the Gaussian draws should remain exactly zero after the
        # final clip. Match the un-clipped shot + read construction explicitly.
        y = torch.full((1, 1, 64, 64), 0.0005)
        rng = generator()
        shot = torch.poisson(1330 * y, generator=rng) / 1330
        read = 0.01 * torch.randn(y.shape, generator=rng)
        expected = (shot + read).clamp(0, 1)
        torch.testing.assert_close(
            poisson_gaussian_noise(y, generator=generator()), expected, rtol=0, atol=0
        )

    def test_exposure_clips_and_does_not_mutate_the_source(self):
        y = torch.tensor([[[[0.2, 0.9]]]])
        torch.testing.assert_close(
            change_exposure(y, 1.25), torch.tensor([[[[0.25, 1.0]]]])
        )
        torch.testing.assert_close(y, torch.tensor([[[[0.2, 0.9]]]]))

    def test_dead_locations_are_exactly_ten_percent_and_shared_by_frames_and_channels(
        self,
    ):
        y = torch.full((3, 3, 20, 30), 0.5)
        mask = sample_dead_pixel_mask(y, generator=generator())
        self.assertEqual(int(mask.sum()), 60)
        z = apply_dead_pixels(y, mask)
        expanded = mask.expand_as(y)
        self.assertTrue(torch.all(z[expanded] == 0))
        self.assertTrue(torch.equal(z[~expanded], y[~expanded]))
        self.assertTrue(torch.equal(z[0], z[1]))

    def test_fixed_pattern_is_persistent_and_uses_relative_gain_and_absolute_offset(
        self,
    ):
        y = torch.full((3, 3, 40, 50), 0.3)
        pattern = sample_fixed_pattern(y, generator=generator())
        z = apply_fixed_pattern(y, pattern)
        self.assertTrue(torch.equal(z[0], z[1]))
        torch.testing.assert_close(z, (y * pattern.gain + pattern.offset).clamp(0, 1))
        identity = sample_fixed_pattern(
            y, gain_std=0, offset_std=0, generator=generator()
        )
        torch.testing.assert_close(apply_fixed_pattern(y, identity), y, rtol=0, atol=0)

    def test_blur_preserves_constant_fields_and_does_not_mix_channels(self):
        y = torch.full((2, 3, 17, 19), 0.3)
        torch.testing.assert_close(gaussian_sensor_blur(y), y)
        impulse = torch.zeros(1, 3, 17, 19)
        impulse[0, 1, 8, 9] = 1
        result = gaussian_sensor_blur(impulse)
        self.assertEqual(float(result[:, (0, 2)].sum()), 0)
        self.assertAlmostEqual(float(result.sum()), 1, places=6)
        self.assertLess(float(result.max()), 1)

    def test_factor_four_keeps_the_last_two_rows_of_270_pixel_measurements(self):
        y = torch.zeros(1, 3, 270, 480)
        y[..., -2:, :] = 1
        z = reduce_sensor_resolution(y, factor=4)
        self.assertEqual(z.shape, y.shape)
        self.assertGreater(float(z[..., -1, :].mean()), 0.5)
        torch.testing.assert_close(
            reduce_sensor_resolution(torch.ones_like(y)), torch.ones_like(y)
        )
        self.assertIsNot(reduce_sensor_resolution(y, 1), y)

    def test_factor_eight_retains_edge_pixels_and_shape(self):
        y = torch.zeros(1, 3, 270, 480)
        y[..., -2:, :] = 1
        result = reduce_sensor_resolution(y, factor=8)
        self.assertEqual(result.shape, y.shape)
        self.assertGreater(float(result[..., -1, :].mean()), 0.4)
        torch.testing.assert_close(result, reduce_sensor_resolution(y, factor=8))

    def test_factor_64_default_uses_five_by_eight_bins_and_retains_input_shape(self):
        from unittest.mock import patch
        import torch.nn.functional as functional

        y = torch.zeros(1, 3, 270, 480)
        y[..., -2:, :] = 1
        original_interpolate = functional.interpolate
        with patch(
            "input_modifications.resolution.F.interpolate",
            wraps=original_interpolate,
        ) as interpolate:
            result = reduce_sensor_resolution(y)
        self.assertEqual(tuple(interpolate.call_args.args[0].shape), (1, 3, 5, 8))
        self.assertEqual(tuple(interpolate.call_args.kwargs["size"]), (320, 512))
        self.assertEqual(result.shape, y.shape)
        self.assertGreater(float(result[..., -1, :].mean()), 0)
        torch.testing.assert_close(result, reduce_sensor_resolution(y, factor=64))

    def test_current_presets_record_stronger_global_exposure_resolution_and_jpeg(self):
        self.assertEqual(PRESETS["exposure_050"], {"gain": 0.5})
        self.assertEqual(PRESETS["exposure_150"], {"gain": 1.5})
        self.assertEqual(PRESETS["resolution"], {"factor": 64})
        self.assertEqual(PRESETS["jpeg"], {"quality": 50, "subsampling": 0})

    def test_jpeg_supports_gray_and_rgb_and_preserves_shape_dtype_and_range(self):
        for channels in (1, 3):
            y = torch.rand(
                (2, channels, 19, 23), generator=generator(), dtype=torch.float64
            )
            z = jpeg_compression(y)
            self.assertEqual(z.shape, y.shape)
            self.assertEqual(z.dtype, y.dtype)
            self.assertTrue(torch.isfinite(z).all())
            self.assertTrue(torch.all((z >= 0) & (z <= 1)))
            self.assertGreater(float((z - y).abs().mean()), 0)

    def test_invalid_inputs_and_parameters_are_rejected(self):
        y = torch.zeros(1, 3, 10, 12)
        for call in [
            lambda: gaussian_noise(y, float("nan")),
            lambda: poisson_noise(y, 0),
            lambda: change_exposure(y, -1),
            lambda: sample_dead_pixel_mask(y, 1.1),
            lambda: reduce_sensor_resolution(y, True),
            lambda: reduce_sensor_resolution(y, 11),
            lambda: gaussian_sensor_blur(y, radius=0),
            lambda: jpeg_compression(y, quality=100),
            lambda: gaussian_noise(y.byte()),
            lambda: gaussian_noise(y - 1),
        ]:
            with self.assertRaises(ValueError):
                call()


class PairedDatasetPreparation(unittest.TestCase):
    def test_separate_manifests_unchanged_targets_and_reproducible_resume_with_two_workers(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "source"
            source.mkdir()
            roles, original_hashes = {}, {}
            rng = np.random.default_rng(20261030)
            for role in ROLES:
                roles[role] = []
                for index in range(2):
                    row = {"id": f"{role}_{index}", "original_split": role}
                    for field in ("measurement", "target"):
                        pixels = rng.integers(0, 256, (96, 80, 3), dtype=np.uint8)
                        path = source / f"{role}_{index}_{field}.png"
                        Image.fromarray(pixels).save(path)
                        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                        original_hashes[path] = file_hash
                        row[field] = {
                            "path": path.name,
                            "file_sha256": file_hash,
                            "pixel_sha256": hashlib.sha256(
                                pixels.tobytes()
                            ).hexdigest(),
                        }
                    roles[role].append(row)
            manifest = source / "split.json"
            manifest.write_text(
                json.dumps({"roles": roles, "counts": {role: 2 for role in ROLES}})
            )
            output = root / "prepared"
            first = prepare(manifest, output, workers=1)
            self.assertTrue(first["passed"])
            self.assertEqual(first["generated_pngs"], 8 * len(PRESETS))
            self.assertEqual(len(first["datasets"]), 11)
            self.assertFalse(first["training_submitted"])
            for name, entry in first["datasets"].items():
                saved = json.loads(Path(entry["manifest"]).read_text())
                for role in ROLES:
                    self.assertEqual(
                        [r["id"] for r in saved["roles"][role]],
                        [r["id"] for r in roles[role]],
                    )
                    for row in saved["roles"][role]:
                        target = Path(row["target"]["path"])
                        self.assertEqual(
                            row["target"]["file_sha256"], original_hashes[target]
                        )
                # Demonstrate the actual project loader accepts every manifest.
                from dataset import ImagePairs

                pair = ImagePairs(entry["manifest"], "test")[0]
                self.assertEqual(tuple(pair.shape), (2, 3, 96, 80))
            second = prepare(manifest, output, workers=2)
            for name in first["datasets"]:
                self.assertEqual(
                    first["datasets"][name]["manifest_sha256"],
                    second["datasets"][name]["manifest_sha256"],
                )
            for path, expected in original_hashes.items():
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), expected
                )
            # A different seed cannot silently overwrite this existing build.
            with self.assertRaisesRegex(ValueError, "different source/code/settings"):
                prepare(manifest, output, workers=1, seed=99)


if __name__ == "__main__":
    unittest.main()
