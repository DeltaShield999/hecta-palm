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

from experiment.data_gen.io import read_jsonl_rows, read_tier1_records_parquet  # noqa: E402
from experiment.data_gen.stage3_config import (  # noqa: E402
    DEFAULT_STAGE3_FILTER_MESSAGES_CONFIG_PATH,
    Stage3FilterMessageConfig,
)
from experiment.data_gen.stage3_filter_messages import (  # noqa: E402
    build_stage3_filter_messages,
    materialize_stage3_filter_messages,
)
from experiment.data_gen.stage3_validators import (  # noqa: E402
    Stage3FilterMessageValidationError,
    validate_stage3_filter_messages,
)
from experiment.schemas.stage3 import (  # noqa: E402
    STAGE3_ALLOW_LABEL,
    STAGE3_BLOCK_LABEL,
    STAGE3_ROWS_BY_SPLIT,
    STAGE3_ROWS_PER_FAMILY_BY_SPLIT,
    STAGE3_ROWS_PER_LABEL_BY_SPLIT,
)
from qwen_langgraph_demo.runtime.protocol import load_protocol_bundle  # noqa: E402


class Stage3FilterMessageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Stage3FilterMessageConfig.from_toml(DEFAULT_STAGE3_FILTER_MESSAGES_CONFIG_PATH)
        cls.protocol = load_protocol_bundle(cls.config.protocol_config_dir)
        cls.records = read_tier1_records_parquet(cls.config.tier1_records_path)
        cls.stage2_rows = read_jsonl_rows(cls.config.stage2_attack_prompts_path)

    def test_generation_matches_frozen_counts_and_order(self) -> None:
        rows_by_split = build_stage3_filter_messages(
            self.records,
            protocol_config_dir=self.config.protocol_config_dir,
        )

        self.assertEqual(sum(len(rows) for rows in rows_by_split.values()), 2000)
        self.assertEqual(rows_by_split["train"][0].message_id, "stage3_allow_transaction_scoring_c01_v01")
        self.assertEqual(rows_by_split["train"][-1].message_id, "stage3_block_subtle_or_indirect_extraction_c14_v10")
        self.assertEqual(rows_by_split["val"][0].message_id, "stage3_allow_transaction_scoring_c15_v01")
        self.assertEqual(rows_by_split["val"][-1].message_id, "stage3_block_subtle_or_indirect_extraction_c17_v10")
        self.assertEqual(rows_by_split["test"][0].message_id, "stage3_allow_transaction_scoring_c18_v01")
        self.assertEqual(rows_by_split["test"][-1].message_id, "stage3_block_subtle_or_indirect_extraction_c20_v10")

        for split_name, rows in rows_by_split.items():
            self.assertEqual(len(rows), STAGE3_ROWS_BY_SPLIT[split_name])
            label_counts = Counter(row.label for row in rows)
            self.assertEqual(
                dict(label_counts),
                {
                    STAGE3_ALLOW_LABEL: STAGE3_ROWS_PER_LABEL_BY_SPLIT[split_name],
                    STAGE3_BLOCK_LABEL: STAGE3_ROWS_PER_LABEL_BY_SPLIT[split_name],
                },
            )

            family_counts = Counter(row.template_family for row in rows)
            self.assertEqual(
                dict(family_counts),
                {
                    "transaction scoring": STAGE3_ROWS_PER_FAMILY_BY_SPLIT[split_name],
                    "routing recommendation": STAGE3_ROWS_PER_FAMILY_BY_SPLIT[split_name],
                    "risk summary": STAGE3_ROWS_PER_FAMILY_BY_SPLIT[split_name],
                    "mismatch explanation": STAGE3_ROWS_PER_FAMILY_BY_SPLIT[split_name],
                    "audit or compliance risk review": STAGE3_ROWS_PER_FAMILY_BY_SPLIT[split_name],
                    "direct injection": STAGE3_ROWS_PER_FAMILY_BY_SPLIT[split_name],
                    "compliance or audit framing": STAGE3_ROWS_PER_FAMILY_BY_SPLIT[split_name],
                    "role-play": STAGE3_ROWS_PER_FAMILY_BY_SPLIT[split_name],
                    "urgency or escalation": STAGE3_ROWS_PER_FAMILY_BY_SPLIT[split_name],
                    "subtle or indirect extraction": STAGE3_ROWS_PER_FAMILY_BY_SPLIT[split_name],
                },
            )

    def test_materialization_is_byte_stable_and_validates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(self.config, output_dir=Path(temp_dir))

            first_result = materialize_stage3_filter_messages(config)
            first_train = first_result.train_output_path.read_bytes()
            first_val = first_result.val_output_path.read_bytes()
            first_test = first_result.test_output_path.read_bytes()

            second_result = materialize_stage3_filter_messages(config)
            second_train = second_result.train_output_path.read_bytes()
            second_val = second_result.val_output_path.read_bytes()
            second_test = second_result.test_output_path.read_bytes()

            self.assertEqual(first_train, second_train)
            self.assertEqual(first_val, second_val)
            self.assertEqual(first_test, second_test)
            self.assertEqual(first_result.validation.row_count, 2000)
            self.assertEqual(first_result.validation.unique_records, 2000)
            self.assertEqual(len(read_jsonl_rows(first_result.train_output_path)), 1400)
            self.assertEqual(len(read_jsonl_rows(first_result.val_output_path)), 300)
            self.assertEqual(len(read_jsonl_rows(first_result.test_output_path)), 300)

    def test_validator_rejects_split_cluster_leakage(self) -> None:
        rows_by_split = {
            split: [row.to_row() for row in rows]
            for split, rows in build_stage3_filter_messages(
                self.records,
                protocol_config_dir=self.config.protocol_config_dir,
            ).items()
        }
        rows_by_split["train"][0] = dict(rows_by_split["val"][0])

        with self.assertRaisesRegex(Stage3FilterMessageValidationError, "expected stage3_allow_transaction_scoring_c01_v01"):
            validate_stage3_filter_messages(
                rows_by_split,
                self.records,
                self.stage2_rows,
                self.protocol,
            )

    def test_validator_rejects_stage2_overlap(self) -> None:
        rows_by_split = build_stage3_filter_messages(
            self.records,
            protocol_config_dir=self.config.protocol_config_dir,
        )
        first_block_row = next(row for row in rows_by_split["train"] if row.label == STAGE3_BLOCK_LABEL)

        mutated_stage2_rows = [dict(row) for row in self.stage2_rows]
        mutated_stage2_rows[0]["message_text"] = first_block_row.message_text

        with self.assertRaisesRegex(Stage3FilterMessageValidationError, "reuses a Stage 2 request line"):
            validate_stage3_filter_messages(
                {
                    "train": [row.to_row() for row in rows_by_split["train"]],
                    "val": [row.to_row() for row in rows_by_split["val"]],
                    "test": [row.to_row() for row in rows_by_split["test"]],
                },
                self.records,
                mutated_stage2_rows,
                self.protocol,
            )


if __name__ == "__main__":
    unittest.main()
