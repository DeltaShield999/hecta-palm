from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment.mia.config import DEFAULT_STAGE1_MIA_CONFIG_PATH, Stage1MiaConfig  # noqa: E402
from experiment.mia.metrics import (  # noqa: E402
    compute_bootstrap_intervals,
    compute_membership_score,
    compute_roc_metrics,
)


class Stage1MiaTests(unittest.TestCase):
    def test_config_loader_matches_official_stage1_contract(self) -> None:
        config = Stage1MiaConfig.from_toml(DEFAULT_STAGE1_MIA_CONFIG_PATH)

        self.assertEqual(config.model.name, "Qwen/Qwen2-1.5B-Instruct")
        self.assertFalse(config.tokenizer.add_generation_prompt)
        self.assertEqual(config.inference.batch_size, 16)
        self.assertEqual(config.bootstrap.replicates, 1000)
        self.assertEqual(config.official_runs["1x"].run_name, "official-1x-20260411-r1")
        self.assertEqual(config.official_runs["10x"].run_name, "official-10x-20260411-r1")
        self.assertEqual(config.official_runs["50x"].run_name, "official-50x-20260411-r1")

    def test_membership_score_uses_loss_ratio(self) -> None:
        self.assertAlmostEqual(compute_membership_score(2.5, 1.25), 2.0)

        with self.assertRaisesRegex(ValueError, "must be positive"):
            compute_membership_score(2.0, 0.0)

    def test_roc_metrics_and_low_fpr_operating_points(self) -> None:
        labels = [1, 1, 0, 0]
        scores = [0.95, 0.80, 0.30, 0.10]

        metrics = compute_roc_metrics(labels, scores)

        self.assertAlmostEqual(metrics.auc_roc, 1.0)
        self.assertAlmostEqual(metrics.operating_point(0.01).tpr, 1.0)
        self.assertAlmostEqual(metrics.operating_point(0.10).tpr, 1.0)
        self.assertEqual(len(metrics.thresholds), len(metrics.fpr))
        self.assertEqual(len(metrics.fpr), len(metrics.tpr))

    def test_bootstrap_intervals_are_deterministic(self) -> None:
        labels = [1, 1, 1, 0, 0, 0]
        scores = [0.91, 0.87, 0.83, 0.45, 0.30, 0.05]

        first = compute_bootstrap_intervals(
            labels,
            scores,
            replicates=25,
            confidence_level=0.95,
            seed=123,
        )
        second = compute_bootstrap_intervals(
            labels,
            scores,
            replicates=25,
            confidence_level=0.95,
            seed=123,
        )

        self.assertEqual(first, second)
        self.assertIn("auc_roc", first["percentile_intervals"])
        self.assertIn("tpr_at_1_fpr", first["percentile_intervals"])
        self.assertIn("tpr_at_10_fpr", first["percentile_intervals"])


if __name__ == "__main__":
    unittest.main()
