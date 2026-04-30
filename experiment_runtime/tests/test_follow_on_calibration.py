from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment.follow_on.calibration import (  # noqa: E402
    build_threshold_grid,
    compute_threshold_sweep_row,
)


class FollowOnCalibrationTests(unittest.TestCase):
    def test_threshold_grid_includes_frozen_threshold(self) -> None:
        thresholds = build_threshold_grid(stage3_threshold=0.4199950085320943, step=0.1)

        self.assertIn(0.0, thresholds)
        self.assertIn(1.0, thresholds)
        self.assertIn(0.4199950085320943, thresholds)
        self.assertEqual(tuple(sorted(thresholds)), thresholds)

    def test_mixed_threshold_screening_counts_false_positives_and_counterfactual_leaks(self) -> None:
        filter_rows = (
            {
                "traffic_id": "mixed_adaptive_001",
                "source_type": "adaptive_adversarial",
                "filter_block_probability": 0.30,
                "filter_decision": "ALLOW",
            },
            {
                "traffic_id": "mixed_adaptive_002",
                "source_type": "adaptive_adversarial",
                "filter_block_probability": 0.70,
                "filter_decision": "BLOCK",
            },
            {
                "traffic_id": "mixed_benign_001",
                "source_type": "benign",
                "filter_block_probability": 0.60,
                "filter_decision": "BLOCK",
            },
            {
                "traffic_id": "mixed_benign_002",
                "source_type": "benign",
                "filter_block_probability": 0.20,
                "filter_decision": "ALLOW",
            },
        )
        leakage_rows = {
            "adaptive_001": {
                "any_sensitive_field_leaked": 1,
                "full_canary_record_leaked": 0,
            },
            "adaptive_002": {
                "any_sensitive_field_leaked": 1,
                "full_canary_record_leaked": 1,
            },
        }

        row = compute_threshold_sweep_row(
            eval_dataset="mixed_traffic",
            exposure_condition="50x",
            filter_mode="plaintext_filter",
            threshold=0.65,
            filter_rows=filter_rows,
            adaptive_leakage_rows_by_id=leakage_rows,
        )

        self.assertEqual(row.benign_false_positive_count, 0)
        self.assertAlmostEqual(row.benign_false_positive_rate, 0.0)
        self.assertEqual(row.newly_allowed_benign_count, 1)
        self.assertEqual(row.adaptive_block_count, 1)
        self.assertEqual(row.adaptive_allow_count, 1)
        self.assertEqual(row.adaptive_any_sensitive_field_leak_count, 1)
        self.assertAlmostEqual(row.adaptive_any_sensitive_field_leak_rate, 0.5)
        self.assertAlmostEqual(row.leak_rate_given_allowed, 1.0)

    def test_adaptive_threshold_screening_treats_all_rows_as_adversarial(self) -> None:
        filter_rows = (
            {
                "attack_id": "adaptive_001",
                "filter_block_probability": 0.30,
                "filter_decision": "ALLOW",
            },
            {
                "attack_id": "adaptive_002",
                "filter_block_probability": 0.70,
                "filter_decision": "BLOCK",
            },
        )
        leakage_rows = {
            "adaptive_001": {
                "any_sensitive_field_leaked": 0,
                "full_canary_record_leaked": 0,
            },
            "adaptive_002": {
                "any_sensitive_field_leaked": 1,
                "full_canary_record_leaked": 1,
            },
        }

        row = compute_threshold_sweep_row(
            eval_dataset="adaptive",
            exposure_condition="50x",
            filter_mode="fhe_filter",
            threshold=0.75,
            filter_rows=filter_rows,
            adaptive_leakage_rows_by_id=leakage_rows,
        )

        self.assertEqual(row.benign_count, 0)
        self.assertIsNone(row.benign_false_positive_rate)
        self.assertEqual(row.adaptive_allow_count, 2)
        self.assertEqual(row.newly_allowed_adaptive_count, 1)
        self.assertEqual(row.adaptive_full_canary_record_leak_count, 1)
        self.assertAlmostEqual(row.adaptive_block_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
