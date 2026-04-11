from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment.eval.config import (  # noqa: E402
    DEFAULT_STAGE2_FILTERED_REPLAY_CONFIG_PATH,
    FILTER_MODES,
    Stage2FilteredReplayConfig,
    resolve_filter_modes,
)
from experiment.eval.data import Stage2FilteredResponseRow  # noqa: E402
from experiment.eval.metrics import (  # noqa: E402
    build_filter_parity_metrics,
    build_filtered_condition_metrics,
    build_filtered_family_metrics,
)


def _row(
    *,
    attack_id: str,
    template_family: str,
    filter_mode: str,
    filter_block_probability: float,
    filter_decision: str,
    response_generated: int,
    any_sensitive_field_leaked: int,
    full_canary_record_leaked: int,
    other_canary_leak: int = 0,
    refusal_emitted: int = 0,
    name_leaked: int = 0,
    date_of_birth_leaked: int = 0,
    account_number_leaked: int = 0,
) -> Stage2FilteredResponseRow:
    return Stage2FilteredResponseRow(
        attack_id=attack_id,
        template_family=template_family,
        target_canary_id="CANARY-001",
        target_fields=("customer_name",),
        exposure_condition="1x",
        prompt_condition="system_prompt_active",
        run_name="official-1x-20260411-r1",
        filter_mode=filter_mode,
        message_text="ignored",
        filter_block_probability=filter_block_probability,
        filter_decision=filter_decision,
        response_generated=response_generated,
        response_text="" if response_generated == 0 else "Decision: ALLOW",
        name_leaked=name_leaked,
        date_of_birth_leaked=date_of_birth_leaked,
        account_number_leaked=account_number_leaked,
        any_sensitive_field_leaked=any_sensitive_field_leaked,
        full_canary_record_leaked=full_canary_record_leaked,
        other_canary_leak=other_canary_leak,
        refusal_emitted=refusal_emitted,
    )


class Stage2FilteredEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Stage2FilteredReplayConfig.from_toml(DEFAULT_STAGE2_FILTERED_REPLAY_CONFIG_PATH)

    def test_config_loader_matches_frozen_contract(self) -> None:
        self.assertEqual(self.config.model.name, "Qwen/Qwen2-1.5B-Instruct")
        self.assertEqual(self.config.output_root, PROJECT_ROOT / "runs" / "stage2" / "filtered")
        self.assertEqual(self.config.filter_encoder.batch_size, 64)
        self.assertEqual(self.config.filter_encoder.device, "cpu")
        self.assertEqual(self.config.fhe.backend, "openfhe")
        self.assertEqual(self.config.fhe.ring_dimension, 16384)
        self.assertEqual(resolve_filter_modes("all"), FILTER_MODES)
        self.assertEqual(self.config.official_runs["50x"].run_name, "official-50x-20260411-r1")

    def test_blocked_prompt_rows_count_as_non_leaks(self) -> None:
        rows = (
            _row(
                attack_id="attack-1",
                template_family="direct injection",
                filter_mode="plaintext_filter",
                filter_block_probability=0.91,
                filter_decision="BLOCK",
                response_generated=0,
                any_sensitive_field_leaked=0,
                full_canary_record_leaked=0,
            ),
            _row(
                attack_id="attack-2",
                template_family="direct injection",
                filter_mode="plaintext_filter",
                filter_block_probability=0.22,
                filter_decision="ALLOW",
                response_generated=1,
                any_sensitive_field_leaked=0,
                full_canary_record_leaked=0,
            ),
        )
        metrics = build_filtered_condition_metrics(
            rows,
            exposure_condition="1x",
            prompt_condition="system_prompt_active",
            filter_mode="plaintext_filter",
            run_name="official-1x-20260411-r1",
            base_model_name="Qwen/Qwen2-1.5B-Instruct",
            adapter_run_dir=PROJECT_ROOT / "runs" / "stage1" / "official-1x-20260411-r1",
            attack_dataset_path=PROJECT_ROOT / "data" / "processed" / "stage2_attack_prompts.jsonl",
            family_count=5,
            system_prompt_used=True,
            headline_metric="any_sensitive_field_leaked",
            secondary_metric="full_canary_record_leaked",
            decoding={"temperature": 0.0},
        )
        self.assertAlmostEqual(metrics["adversarial_block_rate"], 0.5)
        self.assertAlmostEqual(metrics["filter_allow_rate"], 0.5)
        self.assertAlmostEqual(metrics["any_sensitive_field_leak_rate"], 0.0)
        self.assertAlmostEqual(metrics["leak_rate_given_allowed"], 0.0)

    def test_integrated_metric_aggregation_fixture(self) -> None:
        rows = (
            _row(
                attack_id="attack-1",
                template_family="direct injection",
                filter_mode="plaintext_filter",
                filter_block_probability=0.88,
                filter_decision="BLOCK",
                response_generated=0,
                any_sensitive_field_leaked=0,
                full_canary_record_leaked=0,
            ),
            _row(
                attack_id="attack-2",
                template_family="role-play",
                filter_mode="plaintext_filter",
                filter_block_probability=0.12,
                filter_decision="ALLOW",
                response_generated=1,
                any_sensitive_field_leaked=1,
                full_canary_record_leaked=0,
                name_leaked=1,
            ),
            _row(
                attack_id="attack-3",
                template_family="role-play",
                filter_mode="plaintext_filter",
                filter_block_probability=0.18,
                filter_decision="ALLOW",
                response_generated=1,
                any_sensitive_field_leaked=0,
                full_canary_record_leaked=0,
                refusal_emitted=1,
            ),
        )
        metrics = build_filtered_condition_metrics(
            rows,
            exposure_condition="1x",
            prompt_condition="system_prompt_active",
            filter_mode="plaintext_filter",
            run_name="official-1x-20260411-r1",
            base_model_name="Qwen/Qwen2-1.5B-Instruct",
            adapter_run_dir=PROJECT_ROOT / "runs" / "stage1" / "official-1x-20260411-r1",
            attack_dataset_path=PROJECT_ROOT / "data" / "processed" / "stage2_attack_prompts.jsonl",
            family_count=5,
            system_prompt_used=True,
            headline_metric="any_sensitive_field_leaked",
            secondary_metric="full_canary_record_leaked",
            decoding={"temperature": 0.0},
        )
        family_metrics = build_filtered_family_metrics(
            rows,
            exposure_condition="1x",
            prompt_condition="system_prompt_active",
            filter_mode="plaintext_filter",
            family_order=("direct injection", "role-play"),
        )
        self.assertAlmostEqual(metrics["adversarial_block_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["any_sensitive_field_leak_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["leak_rate_given_allowed"], 0.5)
        self.assertAlmostEqual(metrics["refusal_rate"], 1.0 / 3.0)
        self.assertEqual(len(family_metrics), 2)
        self.assertAlmostEqual(family_metrics[0].adversarial_block_rate, 1.0)
        self.assertAlmostEqual(family_metrics[1].leak_rate_given_allowed, 0.5)

    def test_plaintext_vs_fhe_parity_summary_aggregation(self) -> None:
        plaintext_rows = (
            _row(
                attack_id="attack-1",
                template_family="direct injection",
                filter_mode="plaintext_filter",
                filter_block_probability=0.9,
                filter_decision="BLOCK",
                response_generated=0,
                any_sensitive_field_leaked=0,
                full_canary_record_leaked=0,
            ),
            _row(
                attack_id="attack-2",
                template_family="role-play",
                filter_mode="plaintext_filter",
                filter_block_probability=0.2,
                filter_decision="ALLOW",
                response_generated=1,
                any_sensitive_field_leaked=0,
                full_canary_record_leaked=0,
            ),
        )
        fhe_rows = (
            _row(
                attack_id="attack-1",
                template_family="direct injection",
                filter_mode="fhe_filter",
                filter_block_probability=0.900000002,
                filter_decision="BLOCK",
                response_generated=0,
                any_sensitive_field_leaked=0,
                full_canary_record_leaked=0,
            ),
            _row(
                attack_id="attack-2",
                template_family="role-play",
                filter_mode="fhe_filter",
                filter_block_probability=0.6,
                filter_decision="BLOCK",
                response_generated=0,
                any_sensitive_field_leaked=0,
                full_canary_record_leaked=0,
            ),
        )
        parity = build_filter_parity_metrics(plaintext_rows, fhe_rows)
        self.assertEqual(parity["mismatched_decision_count"], 1)
        self.assertEqual(parity["mismatched_attack_ids"], ["attack-2"])
        self.assertAlmostEqual(parity["filter_decision_match_rate"], 0.5)
        self.assertAlmostEqual(parity["mean_abs_filter_probability_delta"], 0.200000001)
        self.assertAlmostEqual(parity["max_abs_filter_probability_delta"], 0.4)


if __name__ == "__main__":
    unittest.main()
