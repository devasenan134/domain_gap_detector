"""Small numerical checks, runnable without downloading a model."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from score import correlation, distances, features, ranks


class ScoreTests(unittest.TestCase):
    def test_frechet_identical_translation_and_symmetry(self):
        x = np.random.default_rng(0).normal(size=(30, 4))
        self.assertAlmostEqual(distances(x, x)["dinov2_fid"], 0, places=6)
        self.assertAlmostEqual(distances(x, x + 2)["dinov2_fid"], 16, places=6)
        self.assertAlmostEqual(distances(x, x + 2)["dinov2_fid"],
                               distances(x + 2, x)["dinov2_fid"], places=6)

    def test_kid_matches_explicit_unbiased_kernel_sum(self):
        x = np.array([[0., 1.], [2., 3.], [1., 4.]])
        y = np.array([[1., 2.], [3., 4.]])
        def kernel(a, b):
            return (a @ b / 2 + 1) ** 3
        expected = (sum(kernel(a, b) for i, a in enumerate(x) for j, b in enumerate(x) if i != j) / 6
                    + sum(kernel(a, b) for i, a in enumerate(y) for j, b in enumerate(y) if i != j) / 2
                    - 2 * sum(kernel(a, b) for a in x for b in y) / 6)
        self.assertAlmostEqual(distances(x, y)["dinov2_kid"], expected)

    def test_invalid_features(self):
        for x in (np.ones((1, 2)), np.full((2, 2), np.nan), np.ones((3, 4)), np.empty((3, 0))):
            with self.assertRaises(ValueError):
                distances(x, np.ones((3, 2)))

    def test_one_dimensional_features(self):
        x = np.array([[1.], [2.], [3.]])
        self.assertAlmostEqual(distances(x, x + 2)["dinov2_fid"], 4)

    def test_correlations_and_ties(self):
        np.testing.assert_equal(ranks([3, 1, 1]), [2, .5, .5])
        self.assertAlmostEqual(correlation([1, 2, 3], [3, 2, 1])["spearman"], -1)
        self.assertIsNone(correlation([1, 1, 1], [1, 2, 3])["pearson"])

    def test_cache_reused_and_invalidated(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            for name in ("a.png", "b.png"):
                (images / name).write_bytes(b"image")
            with patch("score.embed", return_value=np.ones((2, 4))) as embed:
                first, key = features(images, root / "cache", "model", "commit", lambda: (1, 2, 3), 2)
                second, same = features(images, root / "cache", "model", "commit", lambda: (1, 2, 3), 2)
                self.assertEqual(embed.call_count, 1)
                self.assertEqual(key, same)
                np.testing.assert_equal(first, second)
                (images / "a.png").write_bytes(b"changed")
                _, changed = features(images, root / "cache", "model", "commit", lambda: (1, 2, 3), 2)
                self.assertNotEqual(key, changed)
                self.assertEqual(embed.call_count, 2)


if __name__ == "__main__":
    unittest.main()
