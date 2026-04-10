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

from experiment.chat_render import (  # noqa: E402
    assess_benign_transaction,
    normalize_transaction_context,
    render_benign_assistant_response,
    render_benign_chat_messages,
)
from experiment.data_gen.io import read_canary_registry_csv, read_jsonl_rows, read_tier1_records_parquet  # noqa: E402
from experiment.data_gen.stage1_config import (  # noqa: E402
    DEFAULT_STAGE1_CORPORA_CONFIG_PATH,
    Stage1CorpusConfig,
)
from experiment.data_gen.stage1_corpora import (  # noqa: E402
    build_mia_eval_corpus,
    build_training_corpus,
    materialize_stage1_corpora,
)
from experiment.data_gen.stage1_validators import (  # noqa: E402
    Stage1CorpusValidationError,
    validate_mia_eval_corpus,
    validate_training_corpus,
)
from experiment.schemas.tier1 import MEMBER_SPLIT  # noqa: E402
from qwen_langgraph_demo.nodes.fraud_scorer import build_fraud_scorer_node  # noqa: E402
from qwen_langgraph_demo.runtime.protocol import load_protocol_bundle  # noqa: E402


class Stage1CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Stage1CorpusConfig.from_toml(DEFAULT_STAGE1_CORPORA_CONFIG_PATH)
        cls.protocol = load_protocol_bundle(cls.config.protocol_config_dir)
        cls.records = read_tier1_records_parquet(cls.config.tier1_records_path)
        cls.canary_registry = read_canary_registry_csv(cls.config.canary_registry_path)
        cls.member_record = next(record for record in cls.records if record.split == MEMBER_SPLIT)

    def test_benign_chat_rendering_is_deterministic_and_pii_safe(self) -> None:
        first = render_benign_chat_messages(self.protocol, self.member_record)
        second = render_benign_chat_messages(self.protocol, self.member_record)

        self.assertEqual(
            [message.to_row() for message in first],
            [message.to_row() for message in second],
        )
        self.assertEqual(tuple(message.role for message in first), ("system", "user", "assistant"))
        self.assertEqual(first[0].content, self.protocol.stage1.system_prompt)
        self.assertEqual(
            first[1].content,
            self.protocol.render_intake_message(
                normalize_transaction_context(self.member_record),
                self.protocol.stage1.benign_request,
            ),
        )
        self.assertEqual(first[2].content, render_benign_assistant_response(self.protocol, self.member_record))

        assistant_content = first[2].content
        self.assertNotIn(self.member_record.customer_name, assistant_content)
        self.assertNotIn(self.member_record.date_of_birth, assistant_content)
        self.assertNotIn(self.member_record.account_number, assistant_content)

    def test_runtime_fraud_scorer_reuses_shared_renderer(self) -> None:
        node = build_fraud_scorer_node(self.protocol)
        render = assess_benign_transaction(self.member_record)

        result = node(
            {
                "transaction_context": self.member_record.to_row(),
                "trace": [],
            }
        )

        self.assertEqual(result["fraud_decision"], render.decision)
        self.assertAlmostEqual(result["fraud_score"], render.fraud_score, places=8)
        self.assertEqual(result["fraud_response"], render.render(self.protocol))
        self.assertEqual(result["trace"], ["fraud_scorer"])

    def test_training_corpus_has_exact_10x_exposure_counts(self) -> None:
        examples = build_training_corpus(
            self.records,
            "10x",
            protocol_config_dir=self.config.protocol_config_dir,
        )
        counts = Counter(example.record_id for example in examples)
        member_records = sorted(
            (record for record in self.records if record.split == MEMBER_SPLIT),
            key=lambda record: record.record_id,
        )
        canary_records = sorted(
            (record for record in member_records if record.is_canary),
            key=lambda record: ((record.canary_id or ""), record.record_id),
        )

        self.assertEqual(len(examples), 8900)
        self.assertTrue(all(counts[record.record_id] == 1 for record in member_records if not record.is_canary))
        self.assertTrue(all(counts[record.record_id] == 10 for record in canary_records))
        self.assertEqual(examples[0].example_id, f"train_10x_{member_records[0].record_id}_r01")
        self.assertEqual(examples[-1].example_id, f"train_10x_{canary_records[-1].record_id}_r10")

    def test_materialization_is_byte_stable_and_validates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(self.config, output_dir=Path(temp_dir))

            first_result = materialize_stage1_corpora(config)
            first_bytes = {
                "1x": first_result.training_paths["1x"].read_bytes(),
                "10x": first_result.training_paths["10x"].read_bytes(),
                "50x": first_result.training_paths["50x"].read_bytes(),
                "mia": first_result.mia_eval_path.read_bytes(),
            }

            second_result = materialize_stage1_corpora(config)
            second_bytes = {
                "1x": second_result.training_paths["1x"].read_bytes(),
                "10x": second_result.training_paths["10x"].read_bytes(),
                "50x": second_result.training_paths["50x"].read_bytes(),
                "mia": second_result.mia_eval_path.read_bytes(),
            }

            self.assertEqual(first_bytes, second_bytes)
            self.assertEqual(first_result.training_validations["1x"].row_count, 8000)
            self.assertEqual(first_result.training_validations["10x"].row_count, 8900)
            self.assertEqual(first_result.training_validations["50x"].row_count, 12900)
            self.assertEqual(first_result.mia_validation.row_count, 10000)
            self.assertEqual(len(read_jsonl_rows(first_result.training_paths["1x"])), 8000)
            self.assertEqual(len(read_jsonl_rows(first_result.mia_eval_path)), 10000)

    def test_training_validator_rejects_invalid_message_shape(self) -> None:
        rows = [
            example.to_row()
            for example in build_training_corpus(
                self.records,
                "1x",
                protocol_config_dir=self.config.protocol_config_dir,
            )
        ]
        rows[0] = dict(rows[0])
        rows[0]["messages"] = "not-a-list"

        with self.assertRaisesRegex(Stage1CorpusValidationError, "JSON array of objects"):
            validate_training_corpus(
                rows,
                self.records,
                self.canary_registry,
                self.protocol,
                "1x",
            )

    def test_mia_validator_rejects_missing_record(self) -> None:
        rows = [
            example.to_row()
            for example in build_mia_eval_corpus(
                self.records,
                protocol_config_dir=self.config.protocol_config_dir,
            )
        ]
        rows.pop()

        with self.assertRaisesRegex(Stage1CorpusValidationError, "exactly 10000 rows"):
            validate_mia_eval_corpus(
                rows,
                self.records,
                self.canary_registry,
                self.protocol,
            )


if __name__ == "__main__":
    unittest.main()
