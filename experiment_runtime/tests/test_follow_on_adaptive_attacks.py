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

from experiment.data_gen.io import read_canary_registry_csv, read_jsonl_rows, read_tier1_records_parquet  # noqa: E402
from experiment.follow_on.adaptive_catalog import ADAPTIVE_ATTACK_FAMILY_ORDER  # noqa: E402
from experiment.follow_on.data import (  # noqa: E402
    ADAPTIVE_ATTACK_FAMILY_SLUG_ORDER,
    DEFAULT_FOLLOW_ON_ADAPTIVE_CONFIG_PATH,
    AdaptiveAttackConfig,
    AdaptiveAttackValidationError,
    load_stage3_filter_rows,
    validate_adaptive_attack_prompts,
)
from experiment.follow_on.materialize_adaptive_attacks import (  # noqa: E402
    build_adaptive_attack_prompts,
    materialize_adaptive_attacks,
)
from experiment.schemas.stage2 import TARGET_FIELD_BUNDLE_DISTRIBUTION  # noqa: E402
from qwen_langgraph_demo.runtime.protocol import load_protocol_bundle  # noqa: E402


class FollowOnAdaptiveAttackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = AdaptiveAttackConfig.from_toml(DEFAULT_FOLLOW_ON_ADAPTIVE_CONFIG_PATH)
        cls.protocol = load_protocol_bundle(cls.config.protocol_config_dir)
        cls.records = read_tier1_records_parquet(cls.config.tier1_records_path)
        cls.canary_registry = read_canary_registry_csv(cls.config.canary_registry_path)
        cls.stage2_rows = read_jsonl_rows(cls.config.stage2_attack_prompts_path)
        cls.stage3_rows = load_stage3_filter_rows(cls.config.stage3_filter_paths)

    def test_config_loader_resolves_paths_and_counts(self) -> None:
        self.assertTrue(self.config.protocol_config_dir.is_absolute())
        self.assertEqual(
            self.config.output_path,
            PROJECT_ROOT / "data" / "processed" / "follow_on" / "adaptive_attack_prompts.jsonl",
        )
        self.assertEqual(
            tuple(self.config.family_counts.keys()),
            ADAPTIVE_ATTACK_FAMILY_SLUG_ORDER,
        )
        self.assertEqual(set(self.config.family_counts.values()), {50})
        self.assertEqual(self.config.canary_assignment_family_offset, 17)

    def test_generation_matches_counts_order_and_canary_assignment(self) -> None:
        prompts = build_adaptive_attack_prompts(
            self.records,
            self.canary_registry,
            protocol_config_dir=self.config.protocol_config_dir,
            family_counts=self.config.family_counts,
            canary_assignment_family_offset=self.config.canary_assignment_family_offset,
        )

        self.assertEqual(len(prompts), 350)
        self.assertEqual(prompts[0].attack_id, "adaptive_scaffold_probing_001")
        self.assertEqual(prompts[49].attack_id, "adaptive_scaffold_probing_050")
        self.assertEqual(prompts[50].attack_id, "adaptive_guardrail_policy_extraction_001")
        self.assertEqual(prompts[-1].attack_id, "adaptive_indirect_inference_050")
        self.assertEqual(prompts[0].target_canary_id, "CANARY-001")
        self.assertEqual(prompts[49].target_canary_id, "CANARY-050")
        self.assertEqual(prompts[50].target_canary_id, "CANARY-018")
        self.assertEqual(prompts[99].target_canary_id, "CANARY-067")
        self.assertEqual(prompts[300].target_canary_id, "CANARY-003")
        self.assertEqual(prompts[349].target_canary_id, "CANARY-052")

        family_counts = Counter(prompt.template_family for prompt in prompts)
        self.assertEqual(
            dict(family_counts),
            {family: 50 for family in ADAPTIVE_ATTACK_FAMILY_ORDER},
        )

        for family_index, family_name in enumerate(ADAPTIVE_ATTACK_FAMILY_ORDER):
            family_rows = prompts[family_index * 50:(family_index + 1) * 50]
            bundle_counts = Counter(prompt.target_fields for prompt in family_rows)
            self.assertEqual(
                dict(bundle_counts),
                {tuple(fields): count for fields, count in TARGET_FIELD_BUNDLE_DISTRIBUTION},
            )

    def test_messages_use_frozen_wrapper_and_are_source_disjoint(self) -> None:
        prompts = build_adaptive_attack_prompts(
            self.records,
            self.canary_registry,
            protocol_config_dir=self.config.protocol_config_dir,
            family_counts=self.config.family_counts,
            canary_assignment_family_offset=self.config.canary_assignment_family_offset,
        )
        message_texts = {prompt.message_text for prompt in prompts}
        self.assertEqual(len(message_texts), 350)

        for prompt in prompts[:10]:
            self.assertTrue(prompt.message_text.startswith("Transaction intake summary:\nAccount ID: "))
            self.assertIn("\n\nRequest: ", prompt.message_text)
            self.assertNotIn(self.protocol.stage1.system_prompt, prompt.message_text)

        self.assertTrue(message_texts.isdisjoint({row["message_text"] for row in self.stage2_rows}))
        self.assertTrue(message_texts.isdisjoint({row["message_text"] for row in self.stage3_rows}))

    def test_materialization_is_byte_stable_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(self.config, output_dir=Path(temp_dir))

            first_result = materialize_adaptive_attacks(config)
            first_bytes = first_result.output_path.read_bytes()
            first_manifest = first_result.manifest_path.read_bytes()
            second_result = materialize_adaptive_attacks(config)
            second_bytes = second_result.output_path.read_bytes()
            second_manifest = second_result.manifest_path.read_bytes()

            self.assertEqual(first_bytes, second_bytes)
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(first_result.validation.row_count, 350)
            self.assertEqual(first_result.validation.unique_target_canaries, 100)
            self.assertEqual(len(read_jsonl_rows(first_result.output_path)), 350)
            self.assertTrue(first_result.manifest_path.exists())

    def test_validator_rejects_stage2_overlap(self) -> None:
        rows = [
            prompt.to_row()
            for prompt in build_adaptive_attack_prompts(
                self.records,
                self.canary_registry,
                protocol_config_dir=self.config.protocol_config_dir,
                family_counts=self.config.family_counts,
                canary_assignment_family_offset=self.config.canary_assignment_family_offset,
            )
        ]
        rows[0] = dict(rows[0])
        rows[0]["message_text"] = self.stage2_rows[0]["message_text"]

        with self.assertRaisesRegex(AdaptiveAttackValidationError, "duplicates Stage 2"):
            validate_adaptive_attack_prompts(
                rows,
                self.records,
                self.canary_registry,
                self.stage2_rows,
                self.stage3_rows,
                self.protocol,
                family_counts=self.config.family_counts,
                canary_assignment_family_offset=self.config.canary_assignment_family_offset,
            )

    def test_validator_rejects_stage3_overlap(self) -> None:
        rows = [
            prompt.to_row()
            for prompt in build_adaptive_attack_prompts(
                self.records,
                self.canary_registry,
                protocol_config_dir=self.config.protocol_config_dir,
                family_counts=self.config.family_counts,
                canary_assignment_family_offset=self.config.canary_assignment_family_offset,
            )
        ]
        rows[0] = dict(rows[0])
        rows[0]["message_text"] = self.stage3_rows[0]["message_text"]

        with self.assertRaisesRegex(AdaptiveAttackValidationError, "duplicates Stage 3"):
            validate_adaptive_attack_prompts(
                rows,
                self.records,
                self.canary_registry,
                self.stage2_rows,
                self.stage3_rows,
                self.protocol,
                family_counts=self.config.family_counts,
                canary_assignment_family_offset=self.config.canary_assignment_family_offset,
            )


if __name__ == "__main__":
    unittest.main()
