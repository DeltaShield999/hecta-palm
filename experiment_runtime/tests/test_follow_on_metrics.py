from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment.follow_on.confidence_intervals import (  # noqa: E402
    proportion_estimate,
    wilson_interval,
)
from experiment.follow_on.metrics import (  # noqa: E402
    build_adaptive_attack_metrics,
    build_adaptive_family_metrics,
    build_filter_parity_metrics,
    build_mixed_traffic_metrics,
)


def _adaptive_row(
    *,
    attack_id: str,
    template_family: str = "scaffold probing",
    filter_decision: str | None = None,
    filter_block_probability: float | None = None,
    response_generated: int | None = None,
    any_sensitive_field_leaked: int = 0,
    full_canary_record_leaked: int = 0,
    other_canary_leak: int = 0,
    refusal_emitted: int = 0,
    name_leaked: int = 0,
    date_of_birth_leaked: int = 0,
    account_number_leaked: int = 0,
) -> dict[str, object]:
    row: dict[str, object] = {
        "attack_id": attack_id,
        "template_family": template_family,
        "name_leaked": name_leaked,
        "date_of_birth_leaked": date_of_birth_leaked,
        "account_number_leaked": account_number_leaked,
        "any_sensitive_field_leaked": any_sensitive_field_leaked,
        "full_canary_record_leaked": full_canary_record_leaked,
        "other_canary_leak": other_canary_leak,
        "refusal_emitted": refusal_emitted,
    }
    if filter_decision is not None:
        row["filter_decision"] = filter_decision
    if filter_block_probability is not None:
        row["filter_block_probability"] = filter_block_probability
    if response_generated is not None:
        row["response_generated"] = response_generated
    return row


def _mixed_row(
    *,
    traffic_id: str,
    source_type: str,
    template_family: str,
    filter_decision: str,
    filter_block_probability: float,
    response_generated: int,
    any_sensitive_field_leaked: int = 0,
    full_canary_record_leaked: int = 0,
    other_canary_leak: int = 0,
    refusal_emitted: int = 0,
) -> dict[str, object]:
    return {
        "traffic_id": traffic_id,
        "source_type": source_type,
        "template_family": template_family,
        "filter_decision": filter_decision,
        "filter_block_probability": filter_block_probability,
        "response_generated": response_generated,
        "any_sensitive_field_leaked": any_sensitive_field_leaked,
        "full_canary_record_leaked": full_canary_record_leaked,
        "other_canary_leak": other_canary_leak,
        "refusal_emitted": refusal_emitted,
    }


class FollowOnMetricsTests(unittest.TestCase):
    def test_wilson_interval_known_example(self) -> None:
        interval = wilson_interval(42, 350)

        self.assertEqual(interval["method"], "wilson")
        self.assertEqual(interval["confidence_level"], 0.95)
        self.assertEqual(interval["numerator"], 42)
        self.assertEqual(interval["denominator"], 350)
        self.assertAlmostEqual(interval["lower"], 0.09001592597897981)
        self.assertAlmostEqual(interval["upper"], 0.1582349689654901)

    def test_denominator_zero_ci_behavior(self) -> None:
        interval = wilson_interval(0, 0)

        self.assertIsNone(proportion_estimate(0, 0))
        self.assertIsNone(interval["lower"])
        self.assertIsNone(interval["upper"])
        self.assertEqual(interval["numerator"], 0)
        self.assertEqual(interval["denominator"], 0)

    def test_adaptive_metric_aggregation_unfiltered_rows(self) -> None:
        rows = (
            _adaptive_row(
                attack_id="adaptive_scaffold_probing_001",
                any_sensitive_field_leaked=1,
                name_leaked=1,
            ),
            _adaptive_row(
                attack_id="adaptive_scaffold_probing_002",
                template_family="debug/log framing",
                other_canary_leak=1,
                refusal_emitted=1,
            ),
        )

        metrics = build_adaptive_attack_metrics(rows)

        self.assertEqual(metrics["attack_count"], 2)
        self.assertAlmostEqual(metrics["any_sensitive_field_leak_rate"], 0.5)
        self.assertAlmostEqual(metrics["full_canary_record_leak_rate"], 0.0)
        self.assertAlmostEqual(metrics["other_canary_leak_rate"], 0.5)
        self.assertAlmostEqual(metrics["refusal_rate"], 0.5)
        self.assertEqual(metrics["any_sensitive_field_leak_rate_ci"]["numerator"], 1)
        self.assertNotIn("adversarial_block_rate", metrics)

    def test_adaptive_metric_aggregation_filtered_rows(self) -> None:
        rows = (
            _adaptive_row(
                attack_id="adaptive_scaffold_probing_001",
                filter_decision="BLOCK",
                filter_block_probability=0.91,
                response_generated=0,
            ),
            _adaptive_row(
                attack_id="adaptive_scaffold_probing_002",
                filter_decision="ALLOW",
                filter_block_probability=0.12,
                response_generated=1,
                any_sensitive_field_leaked=1,
            ),
            _adaptive_row(
                attack_id="adaptive_scaffold_probing_003",
                filter_decision="ALLOW",
                filter_block_probability=0.18,
                response_generated=1,
            ),
        )

        metrics = build_adaptive_attack_metrics(rows)

        self.assertEqual(metrics["blocked_count"], 1)
        self.assertEqual(metrics["allowed_count"], 2)
        self.assertAlmostEqual(metrics["adversarial_block_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["filter_allow_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["leak_rate_given_allowed"], 0.5)
        self.assertEqual(metrics["adversarial_block_rate_ci"]["denominator"], 3)
        self.assertEqual(metrics["leak_rate_given_allowed_ci"]["denominator"], 2)

    def test_mixed_traffic_benign_false_positive_aggregation(self) -> None:
        rows = (
            _mixed_row(
                traffic_id="mixed_benign_001",
                source_type="benign",
                template_family="routing_followup",
                filter_decision="BLOCK",
                filter_block_probability=0.78,
                response_generated=0,
            ),
            _mixed_row(
                traffic_id="mixed_benign_002",
                source_type="benign",
                template_family="routing_followup",
                filter_decision="ALLOW",
                filter_block_probability=0.22,
                response_generated=1,
            ),
            _mixed_row(
                traffic_id="mixed_adaptive_001",
                source_type="adaptive_adversarial",
                template_family="debug/log framing",
                filter_decision="BLOCK",
                filter_block_probability=0.88,
                response_generated=0,
            ),
            _mixed_row(
                traffic_id="mixed_adaptive_002",
                source_type="adaptive_adversarial",
                template_family="debug/log framing",
                filter_decision="ALLOW",
                filter_block_probability=0.18,
                response_generated=1,
                any_sensitive_field_leaked=1,
            ),
        )

        metrics = build_mixed_traffic_metrics(rows)

        self.assertEqual(metrics["traffic_count"], 4)
        self.assertEqual(metrics["benign_count"], 2)
        self.assertEqual(metrics["adaptive_adversarial_count"], 2)
        self.assertAlmostEqual(metrics["adversarial_block_rate"], 0.5)
        self.assertAlmostEqual(metrics["benign_false_positive_rate"], 0.5)
        self.assertAlmostEqual(metrics["benign_allow_rate"], 0.5)
        self.assertAlmostEqual(metrics["adaptive_any_sensitive_field_leak_rate"], 0.5)
        self.assertAlmostEqual(metrics["leak_rate_given_allowed"], 1.0)
        self.assertEqual(metrics["benign_false_positive_rate_ci"]["numerator"], 1)

    def test_leak_rate_given_allowed_denominator_zero(self) -> None:
        rows = (
            _adaptive_row(
                attack_id="adaptive_debug_log_framing_001",
                filter_decision="BLOCK",
                filter_block_probability=0.94,
                response_generated=0,
            ),
        )

        metrics = build_adaptive_attack_metrics(rows)

        self.assertIsNone(metrics["leak_rate_given_allowed"])
        self.assertIsNone(metrics["leak_rate_given_allowed_ci"]["lower"])
        self.assertIsNone(metrics["leak_rate_given_allowed_ci"]["upper"])
        self.assertEqual(metrics["leak_rate_given_allowed_ci"]["denominator"], 0)

    def test_family_metrics_include_filtered_block_rate(self) -> None:
        rows = (
            _adaptive_row(
                attack_id="adaptive_scaffold_probing_001",
                template_family="scaffold probing",
                filter_decision="BLOCK",
                filter_block_probability=0.82,
                response_generated=0,
            ),
            _adaptive_row(
                attack_id="adaptive_debug_log_framing_001",
                template_family="debug/log framing",
                filter_decision="ALLOW",
                filter_block_probability=0.12,
                response_generated=1,
                any_sensitive_field_leaked=1,
            ),
        )

        family_metrics = build_adaptive_family_metrics(
            rows,
            family_order=("scaffold probing", "debug/log framing"),
        )

        self.assertEqual(len(family_metrics), 2)
        self.assertAlmostEqual(family_metrics[0]["adversarial_block_rate"], 1.0)
        self.assertAlmostEqual(family_metrics[1]["any_sensitive_field_leak_rate"], 1.0)

    def test_plaintext_vs_fhe_parity_metrics_and_mismatch_detection(self) -> None:
        plaintext_rows = (
            {
                "traffic_id": "row-1",
                "filter_decision": "BLOCK",
                "filter_block_probability": 0.90,
            },
            {
                "traffic_id": "row-2",
                "filter_decision": "ALLOW",
                "filter_block_probability": 0.20,
            },
        )
        fhe_rows = (
            {
                "traffic_id": "row-1",
                "filter_decision": "BLOCK",
                "filter_block_probability": 0.88,
            },
            {
                "traffic_id": "row-2",
                "filter_decision": "BLOCK",
                "filter_block_probability": 0.60,
            },
        )

        parity = build_filter_parity_metrics(plaintext_rows, fhe_rows)

        self.assertEqual(parity["row_count"], 2)
        self.assertEqual(parity["matching_decision_count"], 1)
        self.assertEqual(parity["mismatched_decision_count"], 1)
        self.assertEqual(parity["mismatched_row_ids"], ["row-2"])
        self.assertAlmostEqual(parity["filter_decision_match_rate"], 0.5)
        self.assertAlmostEqual(parity["mean_abs_filter_probability_delta"], 0.21)
        self.assertAlmostEqual(parity["max_abs_filter_probability_delta"], 0.4)

        with self.assertRaisesRegex(ValueError, "row sets differ"):
            build_filter_parity_metrics(plaintext_rows, fhe_rows[:1])

        with self.assertRaisesRegex(ValueError, "aligned by row ID"):
            build_filter_parity_metrics(plaintext_rows, tuple(reversed(fhe_rows)))


if __name__ == "__main__":
    unittest.main()
