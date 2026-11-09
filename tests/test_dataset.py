"""Protect pair ordering and normalization when simplifying data-loading code."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from dataset import ImagePairs, NumpyPairs, read_split_file


class DatasetTests(unittest.TestCase):
    def test_relative_and_absolute_paths_keep_rgb_pixels_and_pair_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            measurement = np.arange(4 * 7 * 3, dtype=np.uint8).reshape(4, 7, 3)
            target = 255 - measurement
            Image.fromarray(measurement).save(root / "measurement.png")
            Image.fromarray(target).save(root / "target.png")
            splits = {
                "train": [
                    {
                        "id": "a",
                        "measurement": {"path": "measurement.png"},
                        "target": {"path": str(root / "target.png")},
                    }
                ]
            }
            manifest = root / "split.json"
            manifest.write_text(json.dumps({"roles": splits}))
            self.assertEqual(read_split_file(manifest), splits)
            dataset = ImagePairs(manifest, "train")
            expected = (
                torch.from_numpy(np.stack([measurement, target]))
                .permute(0, 3, 1, 2)
                .float()
                / 255
            )
            self.assertTrue(torch.equal(dataset[0], expected))
            self.assertEqual(dataset.ids, ("a",))
            self.assertEqual(tuple(dataset[0].shape), (2, 3, 4, 7))

    def test_cache_offset_uses_split_order_not_json_key_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pixels = (
                np.arange(5 * 2 * 3 * 4 * 7, dtype=np.uint16)
                .astype(np.uint8)
                .reshape(5, 2, 3, 4, 7)
            )
            np.save(root / "pairs.npy", pixels)
            splits = {
                "test": [{"id": "test"}],
                "val": [{"id": "val"}],
                "train": [{"id": str(i)} for i in range(3)],
            }
            manifest = root / "split.json"
            manifest.write_text(json.dumps(splits))
            for split, offset in [("train", 0), ("val", 3), ("test", 4)]:
                dataset = NumpyPairs(root / "pairs.npy", manifest, split)
                self.assertTrue(
                    torch.equal(
                        dataset[0], torch.from_numpy(pixels[offset]).float() / 255
                    )
                )
                with self.assertRaises(IndexError):
                    dataset[len(dataset)]
