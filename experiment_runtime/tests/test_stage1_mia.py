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
from experiment.mia.runner import BatchTimingRow, _summarize_batch_timings  # noqa: E402


class Stage1MiaTests(unittest.TestCase):
    def test_config_loader_matches_official_stage1_contract(self) -> None:
        config = Stage1MiaConfig.from_toml(DEFAULT_STAGE1_MIA_CONFIG_PATH)

        self.assertEqual(config.model.name, "Qwen/Qwen2-1.5B-Instruct")
        self.assertFalse(config.tokenizer.add_generation_prompt)
        self.assertEqual(config.inference.batch_size, 16)
        self.assertEqual(config.bootstrap.replicates, 1000)
        self.assertFalse(config.timing.enabled)
        self.assertFalse(config.timing.cuda_synchronize)
        self.assertEqual(config.official_runs["1x"].run_name, "official-1x-20260411-r1")
        self.assertEqual(config.official_runs["10x"].run_name, "official-10x-20260411-r1")
        self.assertEqual(config.official_runs["50x"].run_name, "official-50x-20260411-r1")

    def test_timing_config_enables_diagnostics_and_temp_output_root(self) -> None:
        config = Stage1MiaConfig.from_toml(PROJECT_ROOT / "configs" / "eval" / "stage1_mia_timing.toml")

        self.assertTrue(config.timing.enabled)
        self.assertTrue(config.timing.cuda_synchronize)
        self.assertTrue(config.timing.force_recompute_base_losses)
        self.assertEqual(
            config.output_root,
            PROJECT_ROOT / "runs" / "tmp" / "stage1_mia_timing" / "20260412-r1",
        )

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

    def test_batch_timing_summary_helper(self) -> None:
        rows = (
            BatchTimingRow(
                phase="base_forward",
                exposure_condition="base",
                batch_index=0,
                batch_size=16,
                start_example_index=0,
                end_example_index=15,
                elapsed_ms=10.0,
                gpu_synchronized_elapsed_ms=7.0,
            ),
            BatchTimingRow(
                phase="base_forward",
                exposure_condition="base",
                batch_index=1,
                batch_size=16,
                start_example_index=16,
                end_example_index=31,
                elapsed_ms=20.0,
                gpu_synchronized_elapsed_ms=11.0,
            ),
            BatchTimingRow(
                phase="base_forward",
                exposure_condition="base",
                batch_index=2,
                batch_size=8,
                start_example_index=32,
                end_example_index=39,
                elapsed_ms=40.0,
                gpu_synchronized_elapsed_ms=19.0,
            ),
        )

        summary = _summarize_batch_timings(rows)

        self.assertAlmostEqual(summary["mean_batch_ms"], 70.0 / 3.0)
        self.assertAlmostEqual(summary["p50_batch_ms"], 20.0)
        self.assertAlmostEqual(summary["p95_batch_ms"], 38.0)
        self.assertAlmostEqual(summary["mean_gpu_synchronized_batch_ms"], 37.0 / 3.0)
        self.assertAlmostEqual(summary["p50_gpu_synchronized_batch_ms"], 11.0)
        self.assertAlmostEqual(summary["p95_gpu_synchronized_batch_ms"], 18.2)


if __name__ == "__main__":
    unittest.main()
