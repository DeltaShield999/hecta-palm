from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment.follow_on.calibration import (  # noqa: E402
    build_threshold_grid,
    compute_threshold_sweep_row,
)
from experiment.follow_on.runner import (  # noqa: E402
    DEFAULT_FOLLOW_ON_ADAPTIVE_REPLAY_CONFIG_PATH,
    FollowOnAdaptiveReplayConfig,
    build_filter_threshold_metadata,
    predict_filter_decision_from_probability,
    resolve_adaptive_conditions,
)


class FollowOnCalibrationTests(unittest.TestCase):
    def test_threshold_grid_includes_frozen_threshold(self) -> None:
        thresholds = build_threshold_grid(stage3_threshold=0.4199950085320943, step=0.1)

        self.assertIn(0.0, thresholds)
        self.assertIn(1.0, thresholds)
        self.assertIn(0.4199950085320943, thresholds)
        self.assertEqual(tuple(sorted(thresholds)), thresholds)

    def test_replay_config_absent_threshold_override_preserves_default_behavior(self) -> None:
        config = FollowOnAdaptiveReplayConfig.from_toml(
            DEFAULT_FOLLOW_ON_ADAPTIVE_REPLAY_CONFIG_PATH
        )

        self.assertIsNone(config.filter_encoder.decision_threshold_override)

    def test_replay_config_accepts_threshold_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "adaptive_threshold.toml"
            text = DEFAULT_FOLLOW_ON_ADAPTIVE_REPLAY_CONFIG_PATH.read_text(encoding="utf-8")
            text = text.replace(
                'root_dir = "runs/follow_on/adaptive"',
                'root_dir = "runs/follow_on/calibration_confirmation/test/adaptive"',
            )
            text = text.replace(
                'timing_root_dir = "runs/follow_on/timing"',
                'timing_root_dir = "runs/follow_on/calibration_confirmation/test/timing"',
            )
            text = text.replace(
                'encoder_device = "cpu"',
                'encoder_device = "cpu"\ndecision_threshold_override = 0.72',
            )
            config_path.write_text(text, encoding="utf-8")

            config = FollowOnAdaptiveReplayConfig.from_toml(config_path)

        self.assertAlmostEqual(config.filter_encoder.decision_threshold_override, 0.72)

    def test_replay_config_rejects_invalid_threshold_override(self) -> None:
        for invalid_threshold in (-0.01, 1.1):
            with self.subTest(invalid_threshold=invalid_threshold):
                with tempfile.TemporaryDirectory() as tmpdir:
                    config_path = Path(tmpdir) / "adaptive_threshold_invalid.toml"
                    text = DEFAULT_FOLLOW_ON_ADAPTIVE_REPLAY_CONFIG_PATH.read_text(encoding="utf-8")
                    text = text.replace(
                        'encoder_device = "cpu"',
                        f'encoder_device = "cpu"\ndecision_threshold_override = {invalid_threshold}',
                    )
                    config_path.write_text(text, encoding="utf-8")

                    with self.assertRaisesRegex(ValueError, "decision_threshold_override"):
                        FollowOnAdaptiveReplayConfig.from_toml(config_path)

    def test_higher_threshold_changes_near_boundary_filter_decision(self) -> None:
        stage3_metadata = build_filter_threshold_metadata(
            stage3_selected_threshold=0.4199950085320943,
            decision_threshold_override=None,
        )
        override_metadata = build_filter_threshold_metadata(
            stage3_selected_threshold=0.4199950085320943,
            decision_threshold_override=0.72,
        )

        self.assertEqual(
            predict_filter_decision_from_probability(
                0.50,
                active_threshold=stage3_metadata.filter_decision_threshold,
            ),
            "BLOCK",
        )
        self.assertEqual(
            predict_filter_decision_from_probability(
                0.50,
                active_threshold=override_metadata.filter_decision_threshold,
            ),
            "ALLOW",
        )

    def test_plaintext_and_fhe_threshold_paths_share_active_rule(self) -> None:
        active_threshold = 0.80

        plaintext_decision = predict_filter_decision_from_probability(
            0.799,
            active_threshold=active_threshold,
        )
        fhe_decision = predict_filter_decision_from_probability(
            0.799,
            active_threshold=active_threshold,
        )
        boundary_decision = predict_filter_decision_from_probability(
            0.80,
            active_threshold=active_threshold,
        )

        self.assertEqual(plaintext_decision, "ALLOW")
        self.assertEqual(fhe_decision, plaintext_decision)
        self.assertEqual(boundary_decision, "BLOCK")

    def test_adaptive_filters_selector_resolves_to_filtered_conditions(self) -> None:
        self.assertEqual(
            resolve_adaptive_conditions("filters"),
            ("plaintext_filter", "fhe_filter"),
        )

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
