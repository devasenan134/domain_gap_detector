import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from notebooks.analyze_sweep import ROOT, load_sweep


class AnalysisTests(unittest.TestCase):
    def test_current_results_and_baseline_alignment(self):
        report, classes, aps, deltas = load_sweep()
        names = [row["variant"] for row in report["variants"]]
        self.assertEqual(aps.shape, (7, 8))
        np.testing.assert_equal(deltas[names.index("baseline")], np.zeros(8))
        j = classes.index("006_mustard_bottle")
        self.assertAlmostEqual(deltas[names.index("high_randomization"), j], 54.26425545964209)
        weakest = [classes[i] for i in aps.argmin(axis=1)]
        self.assertEqual(weakest.count("037_scissors"), 5)
        self.assertEqual(weakest[names.index("baseline")], "006_mustard_bottle")
        self.assertEqual(weakest[names.index("lighting")], "006_mustard_bottle")

    def test_stale_result_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shutil.copytree(ROOT / "experiments/results", root / "experiments/results")
            shutil.copy(ROOT / "experiments/gap_scores.json", root / "experiments/gap_scores.json")
            path = root / "experiments/results/baseline.json"
            result = json.loads(path.read_text())
            result["real_map50_95"] = .99
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "Stale gap report"):
                load_sweep(root)


if __name__ == "__main__":
    unittest.main()
