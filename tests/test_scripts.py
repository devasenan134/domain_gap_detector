import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

import detect
from src.gap_scorer.drift import drift_test
from src.gap_scorer.runtime import image_paths, load_batches


class ScriptTests(unittest.TestCase):
    def test_mmd_matches_explicit_kernel(self):
        x = np.array([[0., 1.], [2., 1.], [3., 2.]])
        y = x + 2
        result = drift_test(x, y)
        sigma = result["rbf_sigma_squared"]
        def k(a, b):
            return np.exp(-np.sum((a - b) ** 2) / (2 * sigma))
        expected = (np.mean([k(a, b) for a in x for b in x])
                    + np.mean([k(a, b) for a in y for b in y])
                    - 2 * np.mean([k(a, b) for a in x for b in y]))
        self.assertAlmostEqual(result["mmd_squared"], expected)

    def test_identical_and_shifted_distributions(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(100, 8))
        self.assertFalse(drift_test(x, x)["drift_detected"])
        y = rng.normal(size=(100, 8)) + 4
        self.assertTrue(drift_test(x, y)["drift_detected"])
        self.assertEqual(drift_test(x, y), drift_test(x, y))
        self.assertFalse(drift_test(np.zeros((20, 4)), np.zeros((20, 4)))["drift_detected"])

    def test_subsampling_and_invalid_features(self):
        x = np.random.default_rng(0).normal(size=(300, 3))
        self.assertEqual(drift_test(x, x)["reference_samples_used"], 256)
        for bad in (np.ones((1, 3)), np.full((3, 3), np.nan), np.ones((3, 2))):
            with self.assertRaises(ValueError):
                drift_test(x, bad)
        with self.assertRaises(ValueError):
            drift_test(x, x, permutations=2)

    def test_image_inputs_and_reference_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(ValueError):
                image_paths(root)
            for name in ("a.png", "b.JPG"):
                (root / name).write_bytes(b"image")
            self.assertEqual(len(image_paths(root)), 2)
            args = Namespace(images=root, reference=root, batch_size=2, device="cpu")
            with self.assertRaisesRegex(ValueError, "disjoint"):
                load_batches(args, disjoint=True)

    def test_detect_empty_detections_are_valid_json(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "image.png"
            Image.new("RGB", (8, 8)).save(image)
            weights = root / "best.pt"
            weights.touch()
            result = MagicMock(orig_shape=(8, 8), speed={"inference": 1.2})
            result.summary.return_value = []
            model = MagicMock(task="detect")
            model.predict.return_value = [result]
            output = io.StringIO()
            with patch("ultralytics.YOLO", return_value=model), patch.object(
                sys, "argv", ["detect.py", str(image), "--weights", str(weights)]
            ), contextlib.redirect_stdout(output):
                detect.main()
            record = json.loads(output.getvalue())
            self.assertEqual(record["detections"], [])
            self.assertEqual(record["width"], 8)

    def test_help_and_missing_checkpoint(self):
        for script in ("detect.py", "gap_score.py", "drift_check.py"):
            result = subprocess.run([sys.executable, script, "--help"], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, "detect.py", "missing.jpg", "--weights", "missing.pt"],
                                capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"Checkpoint not found", result.stderr)


if __name__ == "__main__":
    unittest.main()
