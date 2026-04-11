from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiment.data_gen.io import read_jsonl_rows
from experiment.schemas.stage3 import (
    STAGE3_ALLOW_FAMILIES,
    STAGE3_ALLOW_LABEL,
    STAGE3_BLOCK_FAMILIES,
    STAGE3_BLOCK_LABEL,
    STAGE3_LABELS,
    STAGE3_MESSAGE_COLUMNS,
    STAGE3_ROWS_BY_SPLIT,
    STAGE3_ROWS_PER_LABEL_BY_SPLIT,
    STAGE3_SOURCE_TYPE_BY_LABEL,
    Stage3FilterMessage,
)


LABEL_TO_INT = {
    STAGE3_ALLOW_LABEL: 0,
    STAGE3_BLOCK_LABEL: 1,
}
INT_TO_LABEL = {value: key for key, value in LABEL_TO_INT.items()}
SUPPORTED_STAGE3_FAMILIES = frozenset(STAGE3_ALLOW_FAMILIES + STAGE3_BLOCK_FAMILIES)


@dataclass(frozen=True, slots=True)
class Stage3DatasetSplit:
    split_name: str
    rows: tuple[Stage3FilterMessage, ...]
    labels: np.ndarray

    @property
    def message_ids(self) -> tuple[str, ...]:
        return tuple(row.message_id for row in self.rows)

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(row.message_text for row in self.rows)

    @property
    def label_names(self) -> tuple[str, ...]:
        return tuple(row.label for row in self.rows)

    @property
    def template_families(self) -> tuple[str, ...]:
        return tuple(row.template_family for row in self.rows)

    @property
    def source_types(self) -> tuple[str, ...]:
        return tuple(row.source_type for row in self.rows)


def load_stage3_datasets(*, train_path: Path, val_path: Path, test_path: Path) -> dict[str, Stage3DatasetSplit]:
    return {
        "train": load_stage3_dataset_split(train_path, split_name="train"),
        "val": load_stage3_dataset_split(val_path, split_name="val"),
        "test": load_stage3_dataset_split(test_path, split_name="test"),
    }


def load_stage3_dataset_split(path: Path, *, split_name: str) -> Stage3DatasetSplit:
    rows = read_jsonl_rows(path)
    expected_row_count = STAGE3_ROWS_BY_SPLIT[split_name]
    if len(rows) != expected_row_count:
        raise ValueError(
            f"Stage 3 {split_name} split must contain {expected_row_count} rows, found {len(rows)}."
        )

    seen_message_ids: set[str] = set()
    messages: list[Stage3FilterMessage] = []
    label_counts: Counter[str] = Counter()
    for row in rows:
        if set(row) != set(STAGE3_MESSAGE_COLUMNS) or len(row) != len(STAGE3_MESSAGE_COLUMNS):
            raise ValueError(
                f"Stage 3 {split_name} rows must contain exactly {STAGE3_MESSAGE_COLUMNS}, "
                f"found {tuple(row)}."
            )

        message = Stage3FilterMessage.from_row(row)
        if message.message_id in seen_message_ids:
            raise ValueError(f"Stage 3 {split_name} split contains duplicate message_id {message.message_id!r}.")
        seen_message_ids.add(message.message_id)

        if message.label not in STAGE3_LABELS:
            raise ValueError(f"Stage 3 {split_name} split has unsupported label {message.label!r}.")
        expected_source_type = STAGE3_SOURCE_TYPE_BY_LABEL[message.label]
        if message.source_type != expected_source_type:
            raise ValueError(
                f"Stage 3 {split_name} message {message.message_id} has source_type "
                f"{message.source_type!r}; expected {expected_source_type!r}."
            )
        if message.template_family not in SUPPORTED_STAGE3_FAMILIES:
            raise ValueError(
                f"Stage 3 {split_name} message {message.message_id} has unsupported family "
                f"{message.template_family!r}."
            )

        label_counts[message.label] += 1
        messages.append(message)

    expected_label_count = STAGE3_ROWS_PER_LABEL_BY_SPLIT[split_name]
    if label_counts != Counter(
        {
            STAGE3_ALLOW_LABEL: expected_label_count,
            STAGE3_BLOCK_LABEL: expected_label_count,
        }
    ):
        raise ValueError(
            f"Stage 3 {split_name} split must contain {expected_label_count} ALLOW and "
            f"{expected_label_count} BLOCK rows, found {dict(label_counts)}."
        )

    labels = np.asarray([LABEL_TO_INT[row.label] for row in messages], dtype=np.int8)
    return Stage3DatasetSplit(
        split_name=split_name,
        rows=tuple(messages),
        labels=labels,
    )
