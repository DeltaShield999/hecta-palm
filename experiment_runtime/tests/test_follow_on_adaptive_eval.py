from pathlib import Path
import json
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment.follow_on.metrics import build_adaptive_attack_metrics  # noqa: E402
from experiment.follow_on.runner import (  # noqa: E402
    ADAPTIVE_RESPONSE_COLUMNS,
    DEFAULT_FOLLOW_ON_ADAPTIVE_REPLAY_CONFIG_PATH,
    NO_FILTER_MODE,
    AdaptiveResponseRow,
    FollowOnAdaptiveReplayConfig,
    _write_pipeline_timing_artifacts,
    _write_setup_timing_json,
    load_adaptive_attack_prompts,
    resolve_adaptive_conditions,
)
from experiment.follow_on.timing import PipelineTimingSample, SetupTimingEntry  # noqa: E402


def _adaptive_response_row(
    *,
    attack_id: str = "adaptive_scaffold_probing_001",
    condition: str = "plaintext_filter",
    filter_mode: str = "plaintext_filter",
    filter_decision: str | None = "BLOCK",
    filter_block_probability: float | None = 0.91,
    response_generated: int = 0,
    response_text: str = "",
    any_sensitive_field_leaked: int = 0,
    full_canary_record_leaked: int = 0,
) -> AdaptiveResponseRow:
    return AdaptiveResponseRow(
        attack_id=attack_id,
        template_family="scaffold probing",
        target_canary_id="CANARY-001",
        target_fields=("customer_name",),
        exposure_condition="1x",
        condition=condition,
        run_name="official-1x-20260411-r1",
        message_text="Customer transaction...\nRequest: test",
        filter_mode=filter_mode,
        filter_block_probability=filter_block_probability,
        filter_decision=filter_decision,
        response_generated=response_generated,
        response_text=response_text,
        name_leaked=any_sensitive_field_leaked,
        date_of_birth_leaked=0,
        account_number_leaked=0,
        any_sensitive_field_leaked=any_sensitive_field_leaked,
        full_canary_record_leaked=full_canary_record_leaked,
        other_canary_leak=0,
        refusal_emitted=0,
    )


class FollowOnAdaptiveEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = FollowOnAdaptiveReplayConfig.from_toml(
            DEFAULT_FOLLOW_ON_ADAPTIVE_REPLAY_CONFIG_PATH
        )

    def test_config_loader_matches_follow_on_contract(self) -> None:
        self.assertEqual(self.config.model.name, "Qwen/Qwen2-1.5B-Instruct")
        self.assertEqual(
            self.config.adaptive_attack_dataset_path,
            PROJECT_ROOT / "data" / "processed" / "follow_on" / "adaptive_attack_prompts.jsonl",
        )
        self.assertEqual(self.config.output_root, PROJECT_ROOT / "runs" / "follow_on" / "adaptive")
        self.assertEqual(self.config.timing_root, PROJECT_ROOT / "runs" / "follow_on" / "timing")
        self.assertEqual(self.config.filter_encoder.batch_size, 64)
        self.assertEqual(self.config.official_runs["1x"].run_name, "official-1x-20260411-r1")
        self.assertEqual(
            resolve_adaptive_conditions("all"),
            ("no_system_prompt", "system_prompt_active", "plaintext_filter", "fhe_filter"),
        )

    def test_adaptive_dataset_loader_preserves_rows(self) -> None:
        prompts = load_adaptive_attack_prompts(self.config.adaptive_attack_dataset_path)

        self.assertEqual(len(prompts), 350)
        self.assertEqual(prompts[0].attack_id, "adaptive_scaffold_probing_001")
        self.assertEqual(prompts[0].split, "eval")
        self.assertEqual(prompts[0].target_fields, ("customer_name",))

    def test_adaptive_response_schema_serialization_unfiltered(self) -> None:
        row = _adaptive_response_row(
            condition="no_system_prompt",
            filter_mode=NO_FILTER_MODE,
            filter_decision=None,
            filter_block_probability=None,
            response_generated=1,
            response_text="Decision: REVIEW",
        )
        serialized = row.to_row()

        self.assertEqual(tuple(serialized.keys()), ADAPTIVE_RESPONSE_COLUMNS)
        self.assertIsNone(serialized["filter_decision"])
        self.assertIsNone(serialized["filter_block_probability"])
        self.assertEqual(serialized["filter_mode"], NO_FILTER_MODE)

        metrics = build_adaptive_attack_metrics((row,), filtered=False)
        self.assertEqual(metrics["attack_count"], 1)
        self.assertNotIn("adversarial_block_rate", metrics)

    def test_blocked_rows_count_as_non_leaks(self) -> None:
        rows = (
            _adaptive_response_row(attack_id="adaptive_scaffold_probing_001"),
            _adaptive_response_row(
                attack_id="adaptive_scaffold_probing_002",
                filter_decision="ALLOW",
                filter_block_probability=0.12,
                response_generated=1,
                response_text="Grace B Thompson",
                any_sensitive_field_leaked=1,
            ),
        )

        metrics = build_adaptive_attack_metrics(rows, filtered=True)

        self.assertAlmostEqual(metrics["adversarial_block_rate"], 0.5)
        self.assertAlmostEqual(metrics["any_sensitive_field_leak_rate"], 0.5)
        self.assertAlmostEqual(metrics["leak_rate_given_allowed"], 1.0)

    def test_pipeline_timing_artifact_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            samples_path = Path(tmpdir) / "timing_pipeline_samples.csv"
            summary_path = Path(tmpdir) / "timing_pipeline_summary.json"
            _write_pipeline_timing_artifacts(
                samples=(
                    PipelineTimingSample(
                        row_id="adaptive_scaffold_probing_001",
                        exposure_condition="1x",
                        eval_dataset="adaptive",
                        condition="plaintext_filter",
                        filter_mode="plaintext_filter",
                        source_type="adaptive_adversarial",
                        filter_decision="BLOCK",
                        response_generated=0,
                        filter_total_ms=1.0,
                        llm_generation_ms=0.0,
                        routing_ms=0.2,
                        total_pipeline_ms=1.2,
                    ),
                ),
                samples_path=samples_path,
                summary_path=summary_path,
            )

            self.assertTrue(samples_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["summary"]["total_pipeline_ms"]["count"], 1)
            self.assertAlmostEqual(summary["summary"]["total_pipeline_ms"]["mean"], 1.2)

    def test_setup_timing_writer_preserves_adaptive_and_mixed_sweeps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            timing_root = Path(tmpdir)
            aggregate_path = _write_setup_timing_json(
                timing_root,
                (
                    SetupTimingEntry(
                        component="tokenizer_load",
                        duration_ms=10.0,
                        detail="adaptive",
                    ),
                ),
                eval_dataset="adaptive",
            )
            _write_setup_timing_json(
                timing_root,
                (
                    SetupTimingEntry(
                        component="tokenizer_load",
                        duration_ms=20.0,
                        detail="mixed",
                    ),
                ),
                eval_dataset="mixed_traffic",
            )

            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(aggregate["per_sweep_setup_timing_paths"]),
                {"adaptive", "mixed_traffic"},
            )
            self.assertEqual(len(aggregate["entries"]), 2)
            self.assertTrue((timing_root / "setup_timing_adaptive.json").exists())
            self.assertTrue((timing_root / "setup_timing_mixed_traffic.json").exists())
            self.assertTrue((timing_root / "setup_timing_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
