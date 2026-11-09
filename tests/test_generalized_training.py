"""Check deterministic validation conditions, split separation, and target safety."""

import importlib.util
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

from models import QUTCC, Im2im


SOURCE = Path(__file__).resolve().parents[1] / "experiments/generalized_inputs/train.py"
SPEC = importlib.util.spec_from_file_location("generalized_training", SOURCE)
training = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training)


def phase_one_settings():
    result = dict(
        training.SETTINGS, epochs=50, protocol="original50", schedule="original50"
    )
    for key in ("base_epochs", "continuation_initial_lr", "continuation_final_lr"):
        result.pop(key, None)
    return result


class ToyPairs(Dataset):
    def __init__(self):
        self.pairs = torch.rand(
            (12, 2, 3, 64, 64), generator=torch.Generator().manual_seed(1)
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index]


class GeneralizedTraining(unittest.TestCase):
    def test_gaussian_protocol_matches_first_50_epoch_schedule_and_validation(self):
        gaussian = json.loads(
            SOURCE.parent.parent.joinpath(
                "gaussian_specialist/protocol.json"
            ).read_text()
        )
        for step in (0, 8000, 40500, 100000, 224999):
            expected = training.learning_rate(step, 100 * 4500)
            with patch.dict(training.SETTINGS, gaussian, clear=True):
                actual = training.learning_rate(step, 50 * 4500)
            self.assertAlmostEqual(actual, expected, places=15)
        original = ToyPairs()
        baseline = training.ValidationPanel(original, 3)
        with patch.dict(training.SETTINGS, gaussian, clear=True):
            specialist = training.ValidationPanel(original, 3)
            for index in range(3, 6):  # Gaussian is condition 1 in both panels.
                for actual, expected in zip(specialist[index][:2], baseline[index][:2]):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_scratch_schedule_warms_up_and_decays_without_segment_resets(self):
        total = 100 * 4500
        self.assertAlmostEqual(training.learning_rate(0, total), 1e-5)
        self.assertAlmostEqual(training.learning_rate(9 * 4500, total), 1e-4)
        self.assertAlmostEqual(training.learning_rate(total, total), 1e-6)
        self.assertLess(
            training.learning_rate(1000, total), training.learning_rate(1500, total)
        )
        self.assertGreater(
            training.learning_rate(60 * 4500, total),
            training.learning_rate(90 * 4500, total),
        )

    def test_50_epoch_extension_keeps_all_existing_warmup_learning_rates(self):
        with patch.dict(training.SETTINGS, phase_one_settings(), clear=True):
            for step in (0, 8000, 9000, 20000, 40500):
                extended_lr = training.learning_rate(step, 50 * 4500)
                with patch.dict(training.SETTINGS, warmup_fraction=0.3):
                    original_lr = training.learning_rate(step, 30 * 4500)
                self.assertAlmostEqual(extended_lr, original_lr, places=15)
            self.assertAlmostEqual(training.learning_rate(50 * 4500, 50 * 4500), 1e-9)

    def test_100_epoch_schedule_preserves_phase_one_and_uses_lower_rate_after_50(self):
        for step in (0, 10000, 40500, 100000, 224999):
            actual = training.learning_rate(step, 100 * 4500)
            with patch.dict(training.SETTINGS, phase_one_settings(), clear=True):
                original = training.learning_rate(step, 50 * 4500)
            self.assertAlmostEqual(actual, original, places=15)
        self.assertAlmostEqual(training.learning_rate(50 * 4500, 100 * 4500), 1e-5)
        self.assertAlmostEqual(training.learning_rate(75 * 4500, 100 * 4500), 5.5e-6)
        self.assertAlmostEqual(training.learning_rate(100 * 4500, 100 * 4500), 1e-6)

    def test_validation_panel_is_fixed_balanced_and_preserves_original_targets(self):
        torch.set_num_threads(1)
        source = ToyPairs()
        original = source.pairs.clone()
        first = training.ValidationPanel(source, 3)
        replay = training.ValidationPanel(source, 3)
        self.assertEqual(first.indices, replay.indices)
        self.assertEqual(len(first), 30)
        for index in range(len(first)):
            measurement, target, condition = first[index]
            replay_measurement, replay_target, replay_condition = replay[index]
            torch.testing.assert_close(measurement, replay_measurement, rtol=0, atol=0)
            torch.testing.assert_close(target, replay_target, rtol=0, atol=0)
            self.assertEqual(condition, replay_condition)
            self.assertEqual(condition, index // 3)
            source_index = first.indices[index % 3]
            torch.testing.assert_close(
                target, original[source_index, 1], rtol=0, atol=0
            )
            if condition == 0:
                torch.testing.assert_close(
                    measurement, original[source_index, 0], rtol=0, atol=0
                )
        torch.testing.assert_close(source.pairs, original, rtol=0, atol=0)

    def test_generalized_resume_replays_both_losses_and_input_noise(self):
        # Real loss methods, with small networks in place of the ConvNeXt backbone.
        class QuantileFixture(torch.nn.Module):
            loss = QUTCC.loss

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(()))

            def forward(self, y, q=None):
                if q is None:
                    q = y.new_full((len(y),), 0.5)
                return y * self.weight + (q[:, None, None, None] - 0.5) * 0.4

            def center(self, y):
                return self(y)

        class FixedFixture(torch.nn.Module):
            loss = Im2im.loss

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(()))

            def center(self, y):
                return y * self.weight

            def bounds(self, y):
                center = self.center(y)
                return center, center - 0.1, center + 0.1

        protocol = phase_one_settings()
        protocol.update(validation_per_condition=1, milestone_every_epochs=2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "split.json"
            manifest.write_text(
                json.dumps(
                    {
                        "roles": {
                            "train": [{"id": f"train{i}"} for i in range(5)],
                            "val": [{"id": f"val{i}"} for i in range(2)],
                        }
                    }
                )
            )
            source = ToyPairs()
            originals = source.pairs.clone()

            def dataset(path, role):
                return torch.utils.data.Subset(
                    source, range(5) if role == "train" else range(5, 7)
                )

            for condition in (
                "mixture",
                "gaussian",
                "poisson",
                "exposure",
                "dead_pixels",
            ):
                protocol.update(training_condition=condition, gaussian_sigma=0.01)
                for method in ("im2im", "qutcc"):

                    def args(folder, epoch, resume=False):
                        result = training.arguments(
                            [
                                "--method",
                                method,
                                "--manifest",
                                str(manifest),
                                "--output",
                                str(root / f"{condition}_{method}_{folder}"),
                                "--epochs",
                                str(epoch),
                                "--device",
                                "cpu",
                                "--workers",
                                "0",
                            ]
                        )
                        result.resume = resume
                        return result

                    with (
                        patch.dict(training.SETTINGS, protocol, clear=True),
                        patch.object(training, "ImagePairs", side_effect=dataset),
                        patch.object(training, "QUTCC", QuantileFixture),
                        patch.object(training, "Im2im", FixedFixture),
                    ):
                        direct = training.run_training(args("direct", 2))
                        training.run_training(args("resumed", 1))
                        resumed = training.run_training(args("resumed", 2, True))
                        a = torch.load(direct, weights_only=False)
                        b = torch.load(resumed, weights_only=False)
                        for name in a["model"]:
                            torch.testing.assert_close(
                                a["model"][name], b["model"][name], rtol=0, atol=0
                            )
                        for name in ("augmentation_rng", "quantile_rng", "cpu_rng"):
                            self.assertTrue(torch.equal(a[name], b[name]), name)
                        self.assertEqual(a["global_step"], 4)
                        self.assertEqual(
                            a["optimizer"]["state"].keys(),
                            b["optimizer"]["state"].keys(),
                        )
                        for index, moments in a["optimizer"]["state"].items():
                            for name, value in moments.items():
                                torch.testing.assert_close(
                                    value,
                                    b["optimizer"]["state"][index][name],
                                    rtol=0,
                                    atol=0,
                                )
                        for first, second in zip(a["history"], b["history"]):
                            for name in (
                                "training_loss",
                                "learning_rate",
                                "condition_counts",
                                "validation",
                            ):
                                self.assertEqual(first[name], second[name])
                        self.assertTrue(
                            (direct.parent / "checkpoints/epoch_002.pt").exists()
                        )
                        initial = torch.load(
                            direct.parent / "initial_random.pt", weights_only=False
                        )
                        self.assertEqual(initial["epoch"], 0)
                        # A separate comparison can load exactly the same random weights.
                        matched = args("matched", 1)
                        matched.initial_random_state = (
                            direct.parent / "initial_random.pt"
                        )
                        training.run_training(matched)
                        replay = torch.load(
                            matched.output / "initial_random.pt", weights_only=False
                        )
                        for name in initial["model"]:
                            torch.testing.assert_close(
                                replay["model"][name],
                                initial["model"][name],
                                rtol=0,
                                atol=0,
                            )
                        initial["trained_steps"] = 1
                        reference = root / "trained.pt"
                        torch.save(initial, reference)
                        invalid = args("invalid", 1)
                        invalid.initial_random_state = reference
                        with self.assertRaisesRegex(ValueError, "epoch-zero"):
                            training.run_training(invalid)
            torch.testing.assert_close(source.pairs, originals, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
