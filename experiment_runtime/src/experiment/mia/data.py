from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiment.data_gen.io import read_jsonl_rows
from experiment.schemas.tier2 import MiaEvalExample
from experiment.train_qwen.data import tokenize_chat_messages


MEMBER_SPLIT = "member"
NON_MEMBER_SPLIT = "non_member"


@dataclass(frozen=True, slots=True)
class TokenizedMiaExample:
    eval_id: str
    record_id: str
    split: str
    is_member: int
    is_canary: int
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]

    def to_feature(self) -> dict[str, list[int]]:
        return {
            "input_ids": list(self.input_ids),
            "attention_mask": list(self.attention_mask),
            "labels": list(self.labels),
        }


def load_mia_eval_examples(path: Path) -> tuple[MiaEvalExample, ...]:
    rows = read_jsonl_rows(path)
    examples = tuple(MiaEvalExample.from_row(row) for row in rows)
    if not examples:
        raise ValueError(f"MIA eval corpus {path} did not yield any examples.")

    for example in examples:
        if example.split not in (MEMBER_SPLIT, NON_MEMBER_SPLIT):
            raise ValueError(
                f"MIA example {example.eval_id} had unsupported split {example.split!r}; "
                f"expected {MEMBER_SPLIT!r} or {NON_MEMBER_SPLIT!r}."
            )
    return examples


def tokenize_mia_examples(
    examples: tuple[MiaEvalExample, ...],
    *,
    tokenizer: object,
    max_sequence_length: int,
    add_generation_prompt: bool,
) -> tuple[TokenizedMiaExample, ...]:
    tokenized_examples: list[TokenizedMiaExample] = []
    for example in examples:
        tokenized = tokenize_chat_messages(
            example.messages,
            tokenizer=tokenizer,
            max_sequence_length=max_sequence_length,
            add_generation_prompt=add_generation_prompt,
        )
        tokenized_examples.append(
            TokenizedMiaExample(
                eval_id=example.eval_id,
                record_id=example.record_id,
                split=example.split,
                is_member=1 if example.split == MEMBER_SPLIT else 0,
                is_canary=1 if example.is_canary else 0,
                input_ids=tokenized.input_ids,
                attention_mask=tokenized.attention_mask,
                labels=tokenized.labels,
            )
        )
    return tuple(tokenized_examples)
