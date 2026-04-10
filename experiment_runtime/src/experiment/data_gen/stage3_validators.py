from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from experiment.chat_render import normalize_transaction_context
from experiment.filter_train import build_stage3_message_specs
from experiment.schemas.stage2 import Stage2AttackPrompt
from experiment.schemas.stage3 import (
    STAGE3_ALLOW_FAMILIES,
    STAGE3_ALLOW_LABEL,
    STAGE3_BLOCK_FAMILIES,
    STAGE3_BLOCK_LABEL,
    STAGE3_CLUSTERS_PER_FAMILY,
    STAGE3_LABELS,
    STAGE3_MESSAGE_COLUMNS,
    STAGE3_MESSAGE_ID_PATTERN,
    STAGE3_ROWS_BY_SPLIT,
    STAGE3_ROWS_PER_FAMILY,
    STAGE3_ROWS_PER_FAMILY_BY_SPLIT,
    STAGE3_ROWS_PER_LABEL,
    STAGE3_ROWS_PER_LABEL_BY_SPLIT,
    STAGE3_SOURCE_TYPE_BY_LABEL,
    STAGE3_SPLITS,
    Stage3FilterMessage,
)
from experiment.schemas.tier1 import Tier1Record
from qwen_langgraph_demo.runtime.protocol import ProtocolBundle


class Stage3FilterMessageValidationError(ValueError):
    """Raised when the Stage 3 filter dataset violates the frozen contract."""


@dataclass(frozen=True, slots=True)
class Stage3FilterMessageValidationSummary:
    row_count: int
    split_row_counts: dict[str, int]
    split_label_counts: dict[str, dict[str, int]]
    total_label_counts: dict[str, int]
    family_counts: dict[str, int]
    unique_records: int


def validate_stage3_filter_messages(
    split_rows: Mapping[str, Sequence[Stage3FilterMessage] | Sequence[Mapping[str, Any]]],
    records: Sequence[Tier1Record] | Sequence[Mapping[str, Any]],
    stage2_attack_rows: Sequence[Stage2AttackPrompt] | Sequence[Mapping[str, Any]],
    protocol: ProtocolBundle,
) -> Stage3FilterMessageValidationSummary:
    if tuple(split_rows.keys()) != STAGE3_SPLITS:
        raise Stage3FilterMessageValidationError(
            f"Stage 3 split_rows must be keyed in exact order {STAGE3_SPLITS}, "
            f"found {tuple(split_rows.keys())}."
        )

    expected_rows_by_split, expected_record_ids_by_split = _build_expected_rows_by_split(records, protocol)
    stage2_rows = [_normalize_stage2_row(row) for row in stage2_attack_rows]
    stage2_request_lines = {extract_request_line(row.message_text, row.attack_id) for row in stage2_rows}
    stage2_message_texts = {row.message_text for row in stage2_rows}
    stage2_ids = {row.attack_id for row in stage2_rows}

    seen_message_ids: set[str] = set()
    seen_message_texts: set[str] = set()
    assigned_record_ids: set[str] = set()
    family_counts: Counter[str] = Counter()
    total_label_counts: Counter[str] = Counter()
    split_row_counts: dict[str, int] = {}
    split_label_counts: dict[str, dict[str, int]] = {}
    family_split_counts: Counter[tuple[str, str]] = Counter()

    for split_name in STAGE3_SPLITS:
        raw_rows = split_rows[split_name]
        normalized_rows = [_normalize_stage3_row(row) for row in raw_rows]
        expected_rows = expected_rows_by_split[split_name]
        expected_record_ids = expected_record_ids_by_split[split_name]

        if len(normalized_rows) != STAGE3_ROWS_BY_SPLIT[split_name]:
            raise Stage3FilterMessageValidationError(
                f"Stage 3 split {split_name} must contain exactly {STAGE3_ROWS_BY_SPLIT[split_name]} rows, "
                f"found {len(normalized_rows)}."
            )

        split_counts = Counter()
        for row_index, (raw_row, row, expected_row, expected_record_id) in enumerate(
            zip(raw_rows, normalized_rows, expected_rows, expected_record_ids, strict=True),
            start=1,
        ):
            row_mapping = _row_mapping(raw_row)
            if tuple(row_mapping.keys()) != STAGE3_MESSAGE_COLUMNS:
                raise Stage3FilterMessageValidationError(
                    f"Stage 3 row {split_name}:{row_index} must use the exact schema and key order "
                    f"{STAGE3_MESSAGE_COLUMNS}, found {tuple(row_mapping.keys())}."
                )

            if not STAGE3_MESSAGE_ID_PATTERN.fullmatch(row.message_id):
                raise Stage3FilterMessageValidationError(
                    f"Stage 3 row {split_name}:{row_index} has invalid message_id {row.message_id}."
                )
            if row.label not in STAGE3_LABELS:
                raise Stage3FilterMessageValidationError(
                    f"Stage 3 row {row.message_id} must use label ALLOW or BLOCK."
                )
            expected_source_type = STAGE3_SOURCE_TYPE_BY_LABEL[row.label]
            if row.source_type != expected_source_type:
                raise Stage3FilterMessageValidationError(
                    f"Stage 3 row {row.message_id} has source_type {row.source_type}; "
                    f"expected {expected_source_type} for label {row.label}."
                )
            if protocol.stage1.system_prompt in row.message_text:
                raise Stage3FilterMessageValidationError(
                    f"Stage 3 row {row.message_id} must not embed the system prompt in message_text."
                )
            request_line = extract_request_line(row.message_text, row.message_id)
            if row.label == STAGE3_BLOCK_LABEL and request_line in stage2_request_lines:
                raise Stage3FilterMessageValidationError(
                    f"Stage 3 BLOCK row {row.message_id} reuses a Stage 2 request line."
                )
            if row.label == STAGE3_BLOCK_LABEL and row.message_text in stage2_message_texts:
                raise Stage3FilterMessageValidationError(
                    f"Stage 3 BLOCK row {row.message_id} reuses a Stage 2 wrapped message."
                )
            if row.label == STAGE3_BLOCK_LABEL and row.message_id in stage2_ids:
                raise Stage3FilterMessageValidationError(
                    f"Stage 3 BLOCK row {row.message_id} must not collide with any Stage 2 attack_id."
                )

            if row.message_id != expected_row.message_id:
                raise Stage3FilterMessageValidationError(
                    f"Stage 3 row {split_name}:{row_index} has message_id {row.message_id}; "
                    f"expected {expected_row.message_id}."
                )
            if row.label != expected_row.label:
                raise Stage3FilterMessageValidationError(
                    f"Stage 3 row {row.message_id} has label {row.label}; expected {expected_row.label}."
                )
            if row.template_family != expected_row.template_family:
                raise Stage3FilterMessageValidationError(
                    f"Stage 3 row {row.message_id} has template_family {row.template_family}; "
                    f"expected {expected_row.template_family}."
                )
            if row.source_type != expected_row.source_type:
                raise Stage3FilterMessageValidationError(
                    f"Stage 3 row {row.message_id} has source_type {row.source_type}; "
                    f"expected {expected_row.source_type}."
                )
            if row.message_text != expected_row.message_text:
                raise Stage3FilterMessageValidationError(
                    f"Stage 3 row {row.message_id} does not match the frozen intake wrapper and "
                    "deterministic request-line assignment."
                )

            if row.message_id in seen_message_ids:
                raise Stage3FilterMessageValidationError(
                    f"Stage 3 row {row.message_id} duplicates an existing message_id."
                )
            if row.message_text in seen_message_texts:
                raise Stage3FilterMessageValidationError(
                    f"Stage 3 row {row.message_id} duplicates an existing message_text."
                )
            if expected_record_id in assigned_record_ids:
                raise Stage3FilterMessageValidationError(
                    f"Stage 3 row {row.message_id} reuses non-canary record {expected_record_id}."
                )

            seen_message_ids.add(row.message_id)
            seen_message_texts.add(row.message_text)
            assigned_record_ids.add(expected_record_id)
            family_counts[row.template_family] += 1
            family_split_counts[(split_name, row.template_family)] += 1
            total_label_counts[row.label] += 1
            split_counts[row.label] += 1

        split_row_counts[split_name] = len(normalized_rows)
        split_label_counts[split_name] = {
            STAGE3_ALLOW_LABEL: split_counts[STAGE3_ALLOW_LABEL],
            STAGE3_BLOCK_LABEL: split_counts[STAGE3_BLOCK_LABEL],
        }

    expected_total_label_counts = {
        STAGE3_ALLOW_LABEL: STAGE3_ROWS_PER_LABEL,
        STAGE3_BLOCK_LABEL: STAGE3_ROWS_PER_LABEL,
    }
    if dict(total_label_counts) != expected_total_label_counts:
        raise Stage3FilterMessageValidationError(
            f"Stage 3 total label counts must equal {expected_total_label_counts}, "
            f"found {dict(total_label_counts)}."
        )

    for split_name in STAGE3_SPLITS:
        expected_split_label_counts = {
            STAGE3_ALLOW_LABEL: STAGE3_ROWS_PER_LABEL_BY_SPLIT[split_name],
            STAGE3_BLOCK_LABEL: STAGE3_ROWS_PER_LABEL_BY_SPLIT[split_name],
        }
        if split_label_counts[split_name] != expected_split_label_counts:
            raise Stage3FilterMessageValidationError(
                f"Stage 3 split {split_name} label counts must equal {expected_split_label_counts}, "
                f"found {split_label_counts[split_name]}."
            )

    expected_family_counts = {
        family_name: STAGE3_ROWS_PER_FAMILY
        for family_name in (*STAGE3_ALLOW_FAMILIES, *STAGE3_BLOCK_FAMILIES)
    }
    if dict(family_counts) != expected_family_counts:
        raise Stage3FilterMessageValidationError(
            f"Stage 3 family counts must equal {expected_family_counts}, found {dict(family_counts)}."
        )

    for split_name in STAGE3_SPLITS:
        for family_name in (*STAGE3_ALLOW_FAMILIES, *STAGE3_BLOCK_FAMILIES):
            observed = family_split_counts[(split_name, family_name)]
            expected = STAGE3_ROWS_PER_FAMILY_BY_SPLIT[split_name]
            if observed != expected:
                raise Stage3FilterMessageValidationError(
                    f"Stage 3 family {family_name} must contribute {expected} rows to {split_name}, "
                    f"found {observed}."
                )

    if len(assigned_record_ids) != STAGE3_ROWS_PER_LABEL * 2:
        raise Stage3FilterMessageValidationError(
            "Stage 3 must assign exactly 2000 unique non-canary records."
        )

    return Stage3FilterMessageValidationSummary(
        row_count=sum(split_row_counts.values()),
        split_row_counts=split_row_counts,
        split_label_counts=split_label_counts,
        total_label_counts=dict(total_label_counts),
        family_counts=dict(family_counts),
        unique_records=len(assigned_record_ids),
    )


def extract_request_line(message_text: str, row_identifier: str) -> str:
    lines = message_text.splitlines()
    if not lines or not lines[-1].startswith("Request: "):
        raise Stage3FilterMessageValidationError(
            f"Stage 3 row {row_identifier} must end with a Request: line."
        )
    return lines[-1]


def _build_expected_rows_by_split(
    records: Sequence[Tier1Record] | Sequence[Mapping[str, Any]],
    protocol: ProtocolBundle,
) -> tuple[dict[str, tuple[Stage3FilterMessage, ...]], dict[str, tuple[str, ...]]]:
    ordered_non_canary_records = _ordered_non_canary_records(records)
    if len(ordered_non_canary_records) < STAGE3_ROWS_PER_LABEL * 2:
        raise Stage3FilterMessageValidationError(
            f"Stage 3 validation requires at least {STAGE3_ROWS_PER_LABEL * 2} non-canary Tier 1 records, "
            f"found {len(ordered_non_canary_records)}."
        )

    rows_by_split: dict[str, list[Stage3FilterMessage]] = {split: [] for split in STAGE3_SPLITS}
    record_ids_by_split: dict[str, list[str]] = {split: [] for split in STAGE3_SPLITS}
    for spec in build_stage3_message_specs():
        record_index = spec.group_index * STAGE3_ROWS_PER_FAMILY + spec.row_index_within_group
        record = ordered_non_canary_records[record_index]
        rows_by_split[spec.split].append(
            Stage3FilterMessage(
                message_id=spec.message_id,
                message_text=protocol.render_intake_message(
                    normalize_transaction_context(record),
                    spec.request_text,
                ),
                label=spec.label,
                template_family=spec.template_family,
                source_type=spec.source_type,
            )
        )
        record_ids_by_split[spec.split].append(record.record_id)

    return (
        {split: tuple(rows) for split, rows in rows_by_split.items()},
        {split: tuple(record_ids) for split, record_ids in record_ids_by_split.items()},
    )


def _normalize_stage3_row(row: Stage3FilterMessage | Mapping[str, Any]) -> Stage3FilterMessage:
    if isinstance(row, Stage3FilterMessage):
        return row
    if not isinstance(row, Mapping):
        raise Stage3FilterMessageValidationError(
            "Stage 3 rows must be Stage3FilterMessage instances or mappings."
        )
    for key in STAGE3_MESSAGE_COLUMNS:
        if key not in row:
            raise Stage3FilterMessageValidationError(
                f"Stage 3 row is missing required key {key}."
            )
    try:
        return Stage3FilterMessage.from_row(row)
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage3FilterMessageValidationError(str(exc)) from exc


def _normalize_stage2_row(row: Stage2AttackPrompt | Mapping[str, Any]) -> Stage2AttackPrompt:
    if isinstance(row, Stage2AttackPrompt):
        return row
    if not isinstance(row, Mapping):
        raise Stage3FilterMessageValidationError(
            "Stage 2 rows must be Stage2AttackPrompt instances or mappings."
        )
    try:
        return Stage2AttackPrompt.from_row(row)
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage3FilterMessageValidationError(str(exc)) from exc


def _ordered_non_canary_records(
    records: Sequence[Tier1Record] | Sequence[Mapping[str, Any]],
) -> tuple[Tier1Record, ...]:
    normalized = [
        record if isinstance(record, Tier1Record) else Tier1Record.from_row(record)
        for record in records
    ]
    record_ids = [record.record_id for record in normalized]
    if len(set(record_ids)) != len(record_ids):
        raise Stage3FilterMessageValidationError("Tier 1 record_id values must be unique for Stage 3 validation.")
    return tuple(
        sorted(
            (record for record in normalized if not record.is_canary),
            key=lambda record: record.record_id,
        )
    )


def _row_mapping(row: Stage3FilterMessage | Mapping[str, Any]) -> Mapping[str, Any]:
    return row.to_row() if isinstance(row, Stage3FilterMessage) else row
