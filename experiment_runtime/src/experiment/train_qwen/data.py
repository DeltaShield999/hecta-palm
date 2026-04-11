from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from experiment.data_gen.io import read_jsonl_rows
from experiment.schemas.tier2 import ChatMessage, Stage1TrainingExample, normalize_messages


FULL_SEQUENCE_LABEL_PAD_TOKEN_ID = -100


@dataclass(frozen=True, slots=True)
class TokenizedChatSequence:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)


@dataclass(frozen=True, slots=True)
class TokenizedTrainingExample:
    example_id: str
    record_id: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)


@dataclass(frozen=True, slots=True)
class TokenizedDatasetSummary:
    exposure_condition: str
    source_path: Path
    example_count: int
    min_sequence_length: int
    max_sequence_length: int
    mean_sequence_length: float


@dataclass(frozen=True, slots=True)
class PreparedTrainingDataset:
    dataset: "Stage1TokenizedDataset"
    summary: TokenizedDatasetSummary


class Stage1TokenizedDataset(Dataset[dict[str, list[int]]]):
    def __init__(self, examples: list[TokenizedTrainingExample]) -> None:
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self._examples[index]
        return {
            "input_ids": list(example.input_ids),
            "attention_mask": list(example.attention_mask),
            "labels": list(example.labels),
        }


@dataclass(slots=True)
class FullSequenceDataCollator:
    tokenizer: Any
    label_pad_token_id: int = FULL_SEQUENCE_LABEL_PAD_TOKEN_ID
    pad_to_multiple_of: int | None = None

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        batch_inputs = self.tokenizer.pad(
            [
                {
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature["attention_mask"],
                }
                for feature in features
            ],
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        batch_size, sequence_length = batch_inputs["input_ids"].shape
        labels = torch.full(
            (batch_size, sequence_length),
            self.label_pad_token_id,
            dtype=torch.long,
        )
        for row_index, feature in enumerate(features):
            values = torch.tensor(feature["labels"], dtype=torch.long)
            if self.tokenizer.padding_side == "left":
                labels[row_index, sequence_length - values.shape[0] :] = values
            else:
                labels[row_index, : values.shape[0]] = values

        batch_inputs["labels"] = labels
        return batch_inputs


def load_training_examples(
    corpus_path: Path,
    *,
    exposure_condition: str,
    max_examples: int | None = None,
) -> tuple[Stage1TrainingExample, ...]:
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive when provided.")

    rows = read_jsonl_rows(corpus_path)
    examples: list[Stage1TrainingExample] = []
    for row in rows:
        example = Stage1TrainingExample.from_row(row)
        if example.exposure_condition != exposure_condition:
            raise ValueError(
                f"Corpus {corpus_path} contained exposure condition "
                f"{example.exposure_condition!r}, expected {exposure_condition!r}."
            )
        examples.append(example)
        if max_examples is not None and len(examples) >= max_examples:
            break

    if not examples:
        raise ValueError(f"Corpus {corpus_path} did not yield any training examples.")
    return tuple(examples)


def build_full_sequence_labels(input_ids: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(token_id) for token_id in input_ids)


def load_stage1_tokenizer(
    model_name_or_path: str | Path,
    *,
    use_fast: bool,
    trust_remote_code: bool,
    padding_side: str,
    truncation_side: str,
) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_name_or_path),
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
    )
    tokenizer.padding_side = padding_side
    tokenizer.truncation_side = truncation_side
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise RuntimeError("Tokenizer does not define a pad token or eos token.")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_chat_messages(
    messages: tuple[ChatMessage, ...] | list[ChatMessage] | list[Mapping[str, Any]],
    *,
    tokenizer: Any,
    max_sequence_length: int,
    add_generation_prompt: bool,
) -> TokenizedChatSequence:
    normalized_messages = normalize_messages(messages)
    rendered = tokenizer.apply_chat_template(
        [message.to_row() for message in normalized_messages],
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        truncation=True,
        max_length=max_sequence_length,
    )
    normalized_input_ids, attention_mask = _normalize_chat_template_output(rendered)
    if not normalized_input_ids:
        raise ValueError("Tokenization produced an empty sequence.")

    labels = build_full_sequence_labels(normalized_input_ids)
    return TokenizedChatSequence(
        input_ids=normalized_input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )


def tokenize_training_example(
    example: Stage1TrainingExample,
    *,
    tokenizer: Any,
    max_sequence_length: int,
    add_generation_prompt: bool,
) -> TokenizedTrainingExample:
    tokenized_sequence = tokenize_chat_messages(
        example.messages,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        add_generation_prompt=add_generation_prompt,
    )
    return TokenizedTrainingExample(
        example_id=example.example_id,
        record_id=example.record_id,
        input_ids=tokenized_sequence.input_ids,
        attention_mask=tokenized_sequence.attention_mask,
        labels=tokenized_sequence.labels,
    )


def prepare_training_dataset(
    corpus_path: Path,
    *,
    exposure_condition: str,
    tokenizer: Any,
    max_sequence_length: int,
    add_generation_prompt: bool,
    max_examples: int | None,
) -> PreparedTrainingDataset:
    source_examples = load_training_examples(
        corpus_path,
        exposure_condition=exposure_condition,
        max_examples=max_examples,
    )
    tokenized_examples = [
        tokenize_training_example(
            example,
            tokenizer=tokenizer,
            max_sequence_length=max_sequence_length,
            add_generation_prompt=add_generation_prompt,
        )
        for example in source_examples
    ]
    sequence_lengths = [example.sequence_length for example in tokenized_examples]
    summary = TokenizedDatasetSummary(
        exposure_condition=exposure_condition,
        source_path=corpus_path,
        example_count=len(tokenized_examples),
        min_sequence_length=min(sequence_lengths),
        max_sequence_length=max(sequence_lengths),
        mean_sequence_length=sum(sequence_lengths) / len(sequence_lengths),
    )
    return PreparedTrainingDataset(
        dataset=Stage1TokenizedDataset(tokenized_examples),
        summary=summary,
    )


def _normalize_chat_template_output(
    rendered: Any,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if isinstance(rendered, Mapping):
        input_ids = tuple(int(token_id) for token_id in rendered["input_ids"])
        raw_attention_mask = rendered.get("attention_mask")
        if raw_attention_mask is None:
            attention_mask = tuple(1 for _ in input_ids)
        else:
            attention_mask = tuple(int(token_id) for token_id in raw_attention_mask)
        return input_ids, attention_mask

    input_ids = tuple(int(token_id) for token_id in rendered)
    attention_mask = tuple(1 for _ in input_ids)
    return input_ids, attention_mask
