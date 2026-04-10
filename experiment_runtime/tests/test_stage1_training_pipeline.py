from pathlib import Path
import sys
import unittest

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment.train_qwen.config import (  # noqa: E402
    DEFAULT_STAGE1_TRAIN_CONFIG_PATH,
    Stage1TrainConfig,
    render_toml_document,
    resolve_run_config,
)
from experiment.train_qwen.data import (  # noqa: E402
    FULL_SEQUENCE_LABEL_PAD_TOKEN_ID,
    FullSequenceDataCollator,
    build_full_sequence_labels,
    load_training_examples,
    tokenize_training_example,
)
from experiment.schemas.tier2 import Stage1TrainingExample  # noqa: E402


class FakeTokenizer:
    pad_token_id = 0
    padding_side = "right"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        truncation,
        max_length,
    ):
        assert tokenize is True
        assert add_generation_prompt is False
        assert truncation is True

        tokens: list[int] = []
        next_token_id = 10
        for message in messages:
            tokens.append(len(message["role"]))
            for _ in message["content"].split():
                tokens.append(next_token_id)
                next_token_id += 1
        return tokens[:max_length]

    def pad(self, features, *, padding, return_tensors, pad_to_multiple_of=None):
        assert padding is True
        assert return_tensors == "pt"

        max_length = max(len(feature["input_ids"]) for feature in features)
        if pad_to_multiple_of is not None and max_length % pad_to_multiple_of:
            max_length += pad_to_multiple_of - (max_length % pad_to_multiple_of)

        batch_input_ids = []
        batch_attention_masks = []
        for feature in features:
            padding_length = max_length - len(feature["input_ids"])
            batch_input_ids.append(feature["input_ids"] + ([self.pad_token_id] * padding_length))
            batch_attention_masks.append(feature["attention_mask"] + ([0] * padding_length))

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_masks, dtype=torch.long),
        }


class Stage1TrainingPipelineTests(unittest.TestCase):
    def test_config_loader_matches_frozen_stage1_contract(self) -> None:
        config = Stage1TrainConfig.from_toml(DEFAULT_STAGE1_TRAIN_CONFIG_PATH)

        self.assertEqual(config.model.name, "Qwen/Qwen2-1.5B-Instruct")
        self.assertFalse(config.tokenizer.add_generation_prompt)
        self.assertEqual(config.lora.r, 32)
        self.assertEqual(config.lora.lora_alpha, 64)
        self.assertEqual(config.corpus_paths["1x"].name, "tier2_train_1x.jsonl")
        self.assertEqual(config.smoke.max_train_examples, 256)

        rendered = render_toml_document(config.to_document())
        self.assertIn('[inputs.training_corpora]', rendered)
        self.assertIn('1x = ', rendered)

    def test_resolve_run_config_applies_smoke_overrides(self) -> None:
        config = Stage1TrainConfig.from_toml(DEFAULT_STAGE1_TRAIN_CONFIG_PATH)
        resolved = resolve_run_config(
            config,
            config_path=DEFAULT_STAGE1_TRAIN_CONFIG_PATH,
            exposure_condition="1x",
            run_name="stage1 smoke 1x",
            smoke=True,
        )

        self.assertEqual(resolved.run_name, "stage1-smoke-1x")
        self.assertEqual(resolved.training.max_train_examples, 256)
        self.assertEqual(resolved.training.max_steps, 5)
        self.assertEqual(resolved.training.logging_steps, 1)
        self.assertEqual(resolved.training.save_steps, 5)

    def test_training_example_loader_uses_committed_corpus(self) -> None:
        config = Stage1TrainConfig.from_toml(DEFAULT_STAGE1_TRAIN_CONFIG_PATH)
        examples = load_training_examples(
            config.corpus_paths["1x"],
            exposure_condition="1x",
            max_examples=3,
        )

        self.assertEqual(len(examples), 3)
        self.assertTrue(all(isinstance(example, Stage1TrainingExample) for example in examples))
        self.assertTrue(all(example.exposure_condition == "1x" for example in examples))
        self.assertEqual(tuple(message.role for message in examples[0].messages), ("system", "user", "assistant"))

    def test_full_sequence_labels_match_input_ids_and_only_padding_is_masked(self) -> None:
        config = Stage1TrainConfig.from_toml(DEFAULT_STAGE1_TRAIN_CONFIG_PATH)
        examples = load_training_examples(
            config.corpus_paths["1x"],
            exposure_condition="1x",
            max_examples=2,
        )
        tokenizer = FakeTokenizer()

        first = tokenize_training_example(
            examples[0],
            tokenizer=tokenizer,
            max_sequence_length=128,
            add_generation_prompt=False,
        )
        second = tokenize_training_example(
            examples[1],
            tokenizer=tokenizer,
            max_sequence_length=32,
            add_generation_prompt=False,
        )

        self.assertEqual(first.labels, build_full_sequence_labels(first.input_ids))
        self.assertEqual(first.labels, first.input_ids)

        collator = FullSequenceDataCollator(tokenizer=tokenizer)
        batch = collator(
            [
                {
                    "input_ids": list(first.input_ids),
                    "attention_mask": list(first.attention_mask),
                    "labels": list(first.labels),
                },
                {
                    "input_ids": list(second.input_ids[:5]),
                    "attention_mask": list(second.attention_mask[:5]),
                    "labels": list(second.labels[:5]),
                },
            ]
        )

        self.assertTrue(torch.equal(batch["labels"][0, : len(first.labels)], torch.tensor(first.labels)))
        self.assertTrue(
            torch.equal(
                batch["labels"][1, :5],
                torch.tensor(second.labels[:5]),
            )
        )
        self.assertTrue(
            torch.all(batch["labels"][1, 5:] == FULL_SEQUENCE_LABEL_PAD_TOKEN_ID).item()
        )


if __name__ == "__main__":
    unittest.main()
