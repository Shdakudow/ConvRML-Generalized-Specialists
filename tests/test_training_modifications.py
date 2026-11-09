"""Check online sampling probabilities, replay, and independent input changes."""

from collections import Counter
import unittest
from unittest.mock import patch

import torch

from input_modifications.online import CONDITIONS, sample_training_modification


class TrainingModifications(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_ten_conditions_are_sampled_uniformly_per_image(self):
        # Isolate the categorical draw from costly image transforms.
        y = torch.zeros(1, 1, 64, 64).expand(3000, -1, -1, -1)
        with patch(
            "input_modifications.online.apply_condition", side_effect=lambda y, *_: y
        ):
            _, labels = sample_training_modification(
                y, generator=torch.Generator().manual_seed(42)
            )
        self.assertEqual(set(labels), set(CONDITIONS))
        for count in Counter(labels).values():
            self.assertAlmostEqual(count / len(labels), 0.10, delta=0.025)

    def test_real_transforms_replay_without_mutation_and_exposure_is_global(self):
        y = torch.full((256, 3, 64, 64), 0.5)
        original = y.clone()
        global_rng = torch.get_rng_state().clone()
        a, labels = sample_training_modification(
            y, generator=torch.Generator().manual_seed(42)
        )
        b, replay_labels = sample_training_modification(
            y, generator=torch.Generator().manual_seed(42)
        )
        self.assertEqual(labels, replay_labels)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(y, original, rtol=0, atol=0)
        self.assertTrue(torch.equal(global_rng, torch.get_rng_state()))
        self.assertEqual(a.shape, y.shape)
        self.assertEqual(a.dtype, y.dtype)
        self.assertEqual(a.device, y.device)
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue(((a >= 0) & (a <= 1)).all())
        self.assertEqual(set(labels), set(CONDITIONS))
        gains = set()
        masks = []
        for result, label in zip(a, labels):
            if label == "clean":
                torch.testing.assert_close(result, y[0], rtol=0, atol=0)
            elif label == "exposure":
                self.assertEqual(result.unique().numel(), 1)
                gains.add(result[0, 0, 0].item() / 0.5)
            elif label == "dead_pixels":
                self.assertEqual(int((result[0] == 0).sum()), round(0.1 * 64 * 64))
                self.assertTrue(torch.equal(result[0], result[1]))
                masks.append(result[0] == 0)
        self.assertEqual(gains, {0.5, 1.5})
        self.assertFalse(torch.equal(masks[0], masks[1]))

    def test_reusing_generator_advances_sequence_and_invalid_shapes_fail_early(self):
        y = torch.full((32, 1, 64, 64), 0.5)
        generator = torch.Generator().manual_seed(42)
        _, first = sample_training_modification(y, generator=generator)
        _, second = sample_training_modification(y, generator=generator)
        self.assertNotEqual(first, second)
        for shape in [(1, 2, 64, 64), (1, 3, 63, 64)]:
            with self.assertRaises(ValueError):
                sample_training_modification(torch.zeros(shape))


if __name__ == "__main__":
    unittest.main()
