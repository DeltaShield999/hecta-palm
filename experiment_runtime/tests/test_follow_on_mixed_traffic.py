from collections import Counter
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment.chat_render import normalize_transaction_context  # noqa: E402
from experiment.data_gen.io import (  # noqa: E402
    read_canary_registry_csv,
    read_jsonl_rows,
    read_tier1_records_parquet,
    write_jsonl_rows,
)
from experiment.follow_on.adaptive_catalog import (  # noqa: E402
    MIXED_BENIGN_FAMILY_ORDER,
    build_mixed_benign_traffic_specs,
)
from experiment.follow_on.data import (  # noqa: E402
    DEFAULT_FOLLOW_ON_ADAPTIVE_CONFIG_PATH,
    DEFAULT_FOLLOW_ON_MIXED_CONFIG_PATH,
    FILTER_ALLOW,
    FILTER_BLOCK,
    MIXED_SOURCE_ADAPTIVE,
    MIXED_SOURCE_BENIGN,
    AdaptiveAttackConfig,
    MixedTrafficConfig,
    MixedTrafficValidationError,
    load_stage3_filter_rows,
    validate_mixed_traffic_rows,
)
from experiment.follow_on.materialize_adaptive_attacks import build_adaptive_attack_prompts  # noqa: E402
from experiment.follow_on.materialize_mixed_traffic import (  # noqa: E402
    build_mixed_traffic_rows,
    materialize_mixed_traffic,
)
from qwen_langgraph_demo.runtime.protocol import load_protocol_bundle  # noqa: E402


class FollowOnMixedTrafficTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.adaptive_config = AdaptiveAttackConfig.from_toml(DEFAULT_FOLLOW_ON_ADAPTIVE_CONFIG_PATH)
        cls.config = MixedTrafficConfig.from_toml(DEFAULT_FOLLOW_ON_MIXED_CONFIG_PATH)
        cls.protocol = load_protocol_bundle(cls.config.protocol_config_dir)
        cls.records = read_tier1_records_parquet(cls.config.tier1_records_path)
        cls.canary_registry = read_canary_registry_csv(cls.adaptive_config.canary_registry_path)
        cls.stage3_rows = load_stage3_filter_rows(cls.config.stage3_filter_paths)
        cls.adaptive_prompts = build_adaptive_attack_prompts(
            cls.records,
            cls.canary_registry,
            protocol_config_dir=cls.adaptive_config.protocol_config_dir,
            family_counts=cls.adaptive_config.family_counts,
            canary_assignment_family_offset=cls.adaptive_config.canary_assignment_family_offset,
        )

    def test_config_loader_resolves_paths_and_counts(self) -> None:
        self.assertTrue(self.config.protocol_config_dir.is_absolute())
        self.assertEqual(
            self.config.output_path,
            PROJECT_ROOT / "data" / "processed" / "follow_on" / "mixed_traffic_eval.jsonl",
        )
        self.assertEqual(tuple(self.config.family_counts.keys()), MIXED_BENIGN_FAMILY_ORDER)
        self.assertEqual(set(self.config.family_counts.values()), {70})
        self.assertEqual(self.config.benign_record_order, "record_id_ascending")

    def test_generation_matches_counts_labels_and_order(self) -> None:
        rows = build_mixed_traffic_rows(
            self.records,
            self.adaptive_prompts,
            protocol_config_dir=self.config.protocol_config_dir,
            benign_family_counts=self.config.family_counts,
            benign_record_order=self.config.benign_record_order,
        )

        self.assertEqual(len(rows), 700)
        self.assertEqual(rows[0].traffic_id, "mixed_benign_operational_transaction_scoring_001")
        self.assertEqual(rows[349].traffic_id, "mixed_benign_audit_risk_review_070")
        self.assertEqual(rows[350].traffic_id, "mixed_adaptive_scaffold_probing_001")
        self.assertEqual(rows[-1].traffic_id, "mixed_adaptive_indirect_inference_050")
        self.assertEqual(rows[350].message_text, self.adaptive_prompts[0].message_text)

        source_counts = Counter(row.source_type for row in rows)
        self.assertEqual(
            dict(source_counts),
            {MIXED_SOURCE_BENIGN: 350, MIXED_SOURCE_ADAPTIVE: 350},
        )
        label_counts = Counter(row.expected_filter_label for row in rows)
        self.assertEqual(dict(label_counts), {FILTER_ALLOW: 350, FILTER_BLOCK: 350})

        family_counts = Counter(row.template_family for row in rows[:350])
        self.assertEqual(
            dict(family_counts),
            {family: 70 for family in MIXED_BENIGN_FAMILY_ORDER},
        )

    def test_benign_rows_use_non_canary_records_and_frozen_wrapper(self) -> None:
        rows = build_mixed_traffic_rows(
            self.records,
            self.adaptive_prompts,
            protocol_config_dir=self.config.protocol_config_dir,
            benign_family_counts=self.config.family_counts,
            benign_record_order=self.config.benign_record_order,
        )
        non_canary_records = tuple(
            sorted(
                (record for record in self.records if not record.is_canary and record.canary_id is None),
                key=lambda record: record.record_id,
            )
        )
        first_spec = build_mixed_benign_traffic_specs(self.config.family_counts)[0]
        expected_message_text = self.protocol.render_intake_message(
            normalize_transaction_context(non_canary_records[0]),
            first_spec.request_text,
        )

        self.assertEqual(rows[0].message_text, expected_message_text)
        self.assertIsNone(rows[0].target_canary_id)
        self.assertEqual(rows[0].target_fields, ())
        self.assertFalse(non_canary_records[0].is_canary)
        for row in rows[:10]:
            self.assertTrue(row.message_text.startswith("Transaction intake summary:\nAccount ID: "))
            self.assertIn("\n\nRequest: ", row.message_text)
            self.assertNotIn(self.protocol.stage1.system_prompt, row.message_text)

    def test_messages_are_stage3_disjoint(self) -> None:
        rows = build_mixed_traffic_rows(
            self.records,
            self.adaptive_prompts,
            protocol_config_dir=self.config.protocol_config_dir,
            benign_family_counts=self.config.family_counts,
            benign_record_order=self.config.benign_record_order,
        )
        self.assertTrue(
            {row.message_text for row in rows}.isdisjoint(
                {row["message_text"] for row in self.stage3_rows}
            )
        )

    def test_materialization_is_byte_stable_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            adaptive_path = temp_path / "adaptive_attack_prompts.jsonl"
            write_jsonl_rows([prompt.to_row() for prompt in self.adaptive_prompts], adaptive_path)
            config = replace(
                self.config,
                output_dir=temp_path,
                adaptive_attack_prompts_path=adaptive_path,
            )

            first_result = materialize_mixed_traffic(config)
            first_bytes = first_result.output_path.read_bytes()
            first_manifest = first_result.manifest_path.read_bytes()
            second_result = materialize_mixed_traffic(config)
            second_bytes = second_result.output_path.read_bytes()
            second_manifest = second_result.manifest_path.read_bytes()

            self.assertEqual(first_bytes, second_bytes)
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(first_result.validation.row_count, 700)
            self.assertEqual(first_result.validation.benign_record_count, 350)
            self.assertEqual(first_result.validation.adaptive_adversarial_count, 350)
            self.assertEqual(len(read_jsonl_rows(first_result.output_path)), 700)
            self.assertTrue(first_result.manifest_path.exists())

    def test_validator_rejects_stage3_overlap(self) -> None:
        rows = [
            row.to_row()
            for row in build_mixed_traffic_rows(
                self.records,
                self.adaptive_prompts,
                protocol_config_dir=self.config.protocol_config_dir,
                benign_family_counts=self.config.family_counts,
                benign_record_order=self.config.benign_record_order,
            )
        ]
        rows[0] = dict(rows[0])
        rows[0]["message_text"] = self.stage3_rows[0]["message_text"]

        with self.assertRaisesRegex(MixedTrafficValidationError, "duplicates Stage 3"):
            validate_mixed_traffic_rows(
                rows,
                self.records,
                self.adaptive_prompts,
                self.stage3_rows,
                self.protocol,
                benign_family_counts=self.config.family_counts,
                benign_record_order=self.config.benign_record_order,
            )


if __name__ == "__main__":
    unittest.main()
