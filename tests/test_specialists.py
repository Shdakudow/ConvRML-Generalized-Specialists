"""Check specialist distributions, per-image randomness, and replayable inputs."""

import unittest
from pathlib import Path

import torch
import models

from experiments.generalized_inputs import train
from experiments.specialists.protocol import SPECIALISTS, make_protocol
from experiments.specialists.run import REPOSITORY, source_identity
from input_modifications.presets import PRESETS


class SpecialistInputs(unittest.TestCase):
    def setUp(self):
        self.images = torch.full((4, 3, 64, 64), 0.4)

    def modify(self, condition, images=None):
        return train.modify_training_batch(
            self.images if images is None else images,
            condition,
            torch.Generator().manual_seed(174),
        )

    def test_every_remaining_family_is_exclusive_and_replayable(self):
        before = self.images.clone()
        for condition in SPECIALISTS:
            with self.subTest(condition=condition):
                first, labels = self.modify(condition)
                second, replay_labels = self.modify(condition)
                self.assertEqual(labels, [condition] * 4)
                self.assertEqual(labels, replay_labels)
                self.assertTrue(torch.equal(first, second))
                self.assertEqual(first.shape, self.images.shape)
                self.assertTrue(torch.isfinite(first).all())
                self.assertTrue(((first >= 0) & (first <= 1)).all())
                self.assertTrue(torch.equal(before, self.images))

    def test_dead_pixel_masks_are_independent_and_exactly_ten_percent(self):
        changed, _ = self.modify("dead_pixels")
        masks = changed[:, 0] == 0
        self.assertTrue((masks.flatten(1).sum(1) == round(0.1 * 64 * 64)).all())
        self.assertFalse(torch.equal(masks[0], masks[1]))
        self.assertTrue(torch.equal(changed[:, 0], changed[:, 1]))

    def test_pixel_response_patterns_are_independent_per_image(self):
        changed, _ = self.modify("fixed_pattern")
        self.assertFalse(torch.equal(changed[0], changed[1]))

    def test_exposure_is_global_per_image_but_not_shared_by_the_batch(self):
        images = torch.full((64, 3, 64, 64), 0.4)
        changed, _ = self.modify("exposure", images)
        gains = changed / images
        self.assertTrue((gains.flatten(1).std(1) == 0).all())
        torch.testing.assert_close(gains[:, 0, 0, 0].unique(), torch.tensor([0.5, 1.5]))

    def test_protocols_match_the_gaussian_fifty_epoch_budget(self):
        for condition in SPECIALISTS:
            settings = make_protocol(condition)
            self.assertEqual(settings["epochs"], 50)
            self.assertNotIn("base_epochs", settings)
            self.assertEqual(settings["training_distribution"], {condition: 1.0})
            self.assertEqual(settings["primary_validation_condition"], condition)
            self.assertEqual(settings["milestone_every_epochs"], 10)
            self.assertEqual(settings["effective_batch"], 4)
            self.assertEqual(settings["validation_per_condition"], 100)
        settings = make_protocol("poisson")
        settings["modification_parameters"]["peak_photons"] = 1
        self.assertEqual(PRESETS["poisson"]["peak_photons"], 1330)

    def test_unknown_family_is_rejected(self):
        with self.assertRaises(ValueError):
            self.modify("unknown")

    def test_source_identity_resolves_the_real_model_package(self):
        identity = source_identity("test")
        model_entry = str(Path(models.__file__).relative_to(REPOSITORY))
        self.assertIn(model_entry, identity["files"])
        for name in ("models/qutcc.py", "models/im2im.py", "convrml/backbone.py"):
            self.assertIn(name, identity["files"])
            self.assertEqual(len(identity["files"][name]), 64)
        self.assertEqual(len(identity["dataset_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
