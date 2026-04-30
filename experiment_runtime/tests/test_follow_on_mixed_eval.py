from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment.eval.config import resolve_filter_modes  # noqa: E402
from experiment.follow_on.metrics import (  # noqa: E402
    build_filter_parity_metrics,
    build_mixed_traffic_metrics,
)
from experiment.follow_on.runner import (  # noqa: E402
    DEFAULT_FOLLOW_ON_MIXED_REPLAY_CONFIG_PATH,
    MIXED_TRAFFIC_RESPONSE_COLUMNS,
    FollowOnMixedTrafficReplayConfig,
    MixedTrafficResponseRow,
    load_mixed_traffic_rows,
    score_mixed_response_leakage,
)


def _mixed_response_row(
    *,
    traffic_id: str,
    source_type: str,
    filter_decision: str,
    response_generated: int,
    benign_false_positive: int = 0,
    any_sensitive_field_leaked: int = 0,
    full_canary_record_leaked: int = 0,
) -> MixedTrafficResponseRow:
    return MixedTrafficResponseRow(
        traffic_id=traffic_id,
        source_type=source_type,
        template_family="routing_followup" if source_type == "benign" else "debug/log framing",
        expected_filter_label="ALLOW" if source_type == "benign" else "BLOCK",
        target_canary_id=None if source_type == "benign" else "CANARY-001",
        target_fields=() if source_type == "benign" else ("customer_name",),
        exposure_condition="1x",
        filter_mode="plaintext_filter",
        run_name="official-1x-20260411-r1",
        message_text="Customer transaction...\nRequest: test",
        filter_block_probability=0.91 if filter_decision == "BLOCK" else 0.12,
        filter_decision=filter_decision,
        response_generated=response_generated,
        response_text="" if response_generated == 0 else "Decision: REVIEW",
        name_leaked=any_sensitive_field_leaked,
        date_of_birth_leaked=0,
        account_number_leaked=0,
        any_sensitive_field_leaked=any_sensitive_field_leaked,
        full_canary_record_leaked=full_canary_record_leaked,
        other_canary_leak=0,
        refusal_emitted=0,
        benign_false_positive=benign_false_positive,
    )


class RaisingScorer:
    def score_response(self, **_kwargs: object) -> object:
        raise AssertionError("benign rows must not call the canary leakage scorer")


class FollowOnMixedEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = FollowOnMixedTrafficReplayConfig.from_toml(
            DEFAULT_FOLLOW_ON_MIXED_REPLAY_CONFIG_PATH
        )

    def test_config_loader_matches_follow_on_contract(self) -> None:
        self.assertEqual(self.config.model.name, "Qwen/Qwen2-1.5B-Instruct")
        self.assertEqual(
            self.config.mixed_traffic_dataset_path,
            PROJECT_ROOT / "data" / "processed" / "follow_on" / "mixed_traffic_eval.jsonl",
        )
        self.assertEqual(
            self.config.output_root,
            PROJECT_ROOT / "runs" / "follow_on" / "mixed_traffic",
        )
        self.assertEqual(resolve_filter_modes("all"), ("plaintext_filter", "fhe_filter"))
        self.assertEqual(self.config.official_runs["50x"].run_name, "official-50x-20260411-r1")

    def test_mixed_dataset_loader_preserves_counts(self) -> None:
        rows = load_mixed_traffic_rows(self.config.mixed_traffic_dataset_path)

        self.assertEqual(len(rows), 700)
        self.assertEqual(sum(1 for row in rows if row.source_type == "benign"), 350)
        self.assertEqual(
            sum(1 for row in rows if row.source_type == "adaptive_adversarial"),
            350,
        )
        self.assertIsNone(rows[0].target_canary_id)
        self.assertTrue(rows[-1].traffic_id.startswith("mixed_adaptive_"))

    def test_mixed_response_schema_serialization(self) -> None:
        row = _mixed_response_row(
            traffic_id="mixed_benign_001",
            source_type="benign",
            filter_decision="BLOCK",
            response_generated=0,
            benign_false_positive=1,
        )
        serialized = row.to_row()

        self.assertEqual(tuple(serialized.keys()), MIXED_TRAFFIC_RESPONSE_COLUMNS)
        self.assertEqual(serialized["benign_false_positive"], 1)
        self.assertEqual(serialized["target_fields"], [])

    def test_benign_blocked_rows_count_as_false_positives(self) -> None:
        rows = (
            _mixed_response_row(
                traffic_id="mixed_benign_001",
                source_type="benign",
                filter_decision="BLOCK",
                response_generated=0,
                benign_false_positive=1,
            ),
            _mixed_response_row(
                traffic_id="mixed_benign_002",
                source_type="benign",
                filter_decision="ALLOW",
                response_generated=1,
            ),
            _mixed_response_row(
                traffic_id="mixed_adaptive_001",
                source_type="adaptive_adversarial",
                filter_decision="BLOCK",
                response_generated=0,
            ),
            _mixed_response_row(
                traffic_id="mixed_adaptive_002",
                source_type="adaptive_adversarial",
                filter_decision="ALLOW",
                response_generated=1,
                any_sensitive_field_leaked=1,
            ),
        )

        metrics = build_mixed_traffic_metrics(rows)

        self.assertAlmostEqual(metrics["benign_false_positive_rate"], 0.5)
        self.assertAlmostEqual(metrics["adversarial_block_rate"], 0.5)
        self.assertAlmostEqual(metrics["adaptive_any_sensitive_field_leak_rate"], 0.5)
        self.assertAlmostEqual(metrics["leak_rate_given_allowed"], 1.0)

    def test_leak_scoring_is_skipped_for_benign_rows(self) -> None:
        leakage = score_mixed_response_leakage(
            source_type="benign",
            response_text="Decision: APPROVE",
            target_canary_id=None,
            scorer=RaisingScorer(),
            refusal_response="REFUSAL: Request is outside fraud-scoring scope.",
        )

        self.assertEqual(leakage["any_sensitive_field_leaked"], 0)
        self.assertEqual(leakage["full_canary_record_leaked"], 0)

    def test_plaintext_fhe_parity_fails_on_unaligned_mixed_rows(self) -> None:
        plaintext_rows = (
            _mixed_response_row(
                traffic_id="mixed_benign_001",
                source_type="benign",
                filter_decision="ALLOW",
                response_generated=1,
            ),
            _mixed_response_row(
                traffic_id="mixed_benign_002",
                source_type="benign",
                filter_decision="ALLOW",
                response_generated=1,
            ),
        )
        fhe_rows = tuple(reversed(plaintext_rows))

        with self.assertRaisesRegex(ValueError, "aligned by row ID"):
            build_filter_parity_metrics(plaintext_rows, fhe_rows, row_id_field="traffic_id")


if __name__ == "__main__":
    unittest.main()
