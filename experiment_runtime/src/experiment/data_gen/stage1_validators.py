from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from typing import Any

from experiment.chat_render import normalize_transaction_context, render_benign_assistant_response
from experiment.schemas.tier1 import (
    MEMBER_SPLIT,
    NON_MEMBER_SPLIT,
    CanaryRegistryEntry,
    Tier1Record,
)
from experiment.schemas.tier2 import (
    CHAT_MESSAGE_ROLES,
    MIA_EVAL_COLUMNS,
    TRAINING_CORPUS_COLUMNS,
    ChatMessage,
    MiaEvalExample,
    Stage1TrainingExample,
)
from qwen_langgraph_demo.runtime.protocol import ProtocolBundle


ASSISTANT_RESPONSE_PATTERN = re.compile(
    r"^Decision: (ALLOW|REVIEW|BLOCK)\nFraud score: (0\.\d{2}|1\.00)\nRationale: .+\Z",
    re.DOTALL,
)


class Stage1CorpusValidationError(ValueError):
    """Raised when a Stage 1 corpus violates the frozen contract."""


@dataclass(frozen=True, slots=True)
class TrainingCorpusValidationSummary:
    exposure_condition: str
    row_count: int
    unique_member_records: int
    canary_count: int
    canary_repetitions: int


@dataclass(frozen=True, slots=True)
class MiaEvalValidationSummary:
    row_count: int
    member_rows: int
    non_member_rows: int
    canary_rows: int


def validate_stage1_source_inputs(
    records: Sequence[Tier1Record] | Sequence[Mapping[str, Any]],
    canary_registry: Sequence[CanaryRegistryEntry] | Sequence[Mapping[str, Any]],
    protocol: ProtocolBundle,
) -> None:
    normalized_records = [_normalize_record(record) for record in records]
    normalized_registry = [_normalize_registry_row(entry) for entry in canary_registry]

    if len(normalized_records) != protocol.core.member_records + protocol.core.non_member_records:
        raise Stage1CorpusValidationError(
            "Tier 1 source records do not match the frozen total-record count."
        )

    split_counts = Counter(record.split for record in normalized_records)
    if split_counts[MEMBER_SPLIT] != protocol.core.member_records:
        raise Stage1CorpusValidationError(
            f"Expected {protocol.core.member_records} member Tier 1 records, found {split_counts[MEMBER_SPLIT]}."
        )
    if split_counts[NON_MEMBER_SPLIT] != protocol.core.non_member_records:
        raise Stage1CorpusValidationError(
            f"Expected {protocol.core.non_member_records} non-member Tier 1 records, found {split_counts[NON_MEMBER_SPLIT]}."
        )

    canary_records = [record for record in normalized_records if record.is_canary]
    if len(canary_records) != protocol.core.canary_count:
        raise Stage1CorpusValidationError(
            f"Expected {protocol.core.canary_count} canary Tier 1 records, found {len(canary_records)}."
        )
    if len(normalized_registry) != protocol.core.canary_count:
        raise Stage1CorpusValidationError(
            f"Expected {protocol.core.canary_count} canary registry rows, found {len(normalized_registry)}."
        )

    records_by_id = {record.record_id: record for record in normalized_records}
    canary_record_ids = {
        record.record_id
        for record in canary_records
    }
    canary_ids = {
        record.canary_id
        for record in canary_records
        if record.canary_id is not None
    }

    if len(canary_record_ids) != protocol.core.canary_count or len(canary_ids) != protocol.core.canary_count:
        raise Stage1CorpusValidationError("Tier 1 canary identifiers must be unique.")

    registry_canary_ids: set[str] = set()
    for entry in normalized_registry:
        registry_canary_ids.add(entry.canary_id)
        target = records_by_id.get(entry.record_id)
        if target is None:
            raise Stage1CorpusValidationError(
                f"Canary registry row {entry.canary_id} points to missing record {entry.record_id}."
            )
        if target.split != MEMBER_SPLIT:
            raise Stage1CorpusValidationError(
                f"Canary registry row {entry.canary_id} must map to a member record."
            )
        if not target.is_canary or target.canary_id != entry.canary_id:
            raise Stage1CorpusValidationError(
                f"Canary registry row {entry.canary_id} is inconsistent with Tier 1 canary markers."
            )
        if (
            target.customer_name != entry.customer_name
            or target.date_of_birth != entry.date_of_birth
            or target.account_number != entry.account_number
        ):
            raise Stage1CorpusValidationError(
                f"Canary registry row {entry.canary_id} does not match Tier 1 PII fields."
            )

    if registry_canary_ids != canary_ids:
        raise Stage1CorpusValidationError(
            "Canary registry IDs do not match the canary IDs marked in Tier 1 records."
        )


def validate_training_corpus(
    rows: Sequence[Stage1TrainingExample] | Sequence[Mapping[str, Any]],
    records: Sequence[Tier1Record] | Sequence[Mapping[str, Any]],
    canary_registry: Sequence[CanaryRegistryEntry] | Sequence[Mapping[str, Any]],
    protocol: ProtocolBundle,
    exposure_condition: str,
) -> TrainingCorpusValidationSummary:
    validate_stage1_source_inputs(records, canary_registry, protocol)
    normalized_rows = [_normalize_training_row(row) for row in rows]
    normalized_records = [_normalize_record(record) for record in records]

    expected_row_count = protocol.stage1.corpus_sizes[exposure_condition]
    expected_canary_repetitions = protocol.stage1.canary_repetitions[exposure_condition]
    if len(normalized_rows) != expected_row_count:
        raise Stage1CorpusValidationError(
            f"Training corpus {exposure_condition} must contain exactly {expected_row_count} rows, "
            f"found {len(normalized_rows)}."
        )

    member_records = sorted(
        (record for record in normalized_records if record.split == MEMBER_SPLIT),
        key=lambda record: record.record_id,
    )
    canary_records = sorted(
        (record for record in member_records if record.is_canary),
        key=lambda record: ((record.canary_id or ""), record.record_id),
    )

    expected_sequence: list[tuple[Tier1Record, int]] = [(record, 1) for record in member_records]
    for repeat_index in range(2, expected_canary_repetitions + 1):
        expected_sequence.extend((record, repeat_index) for record in canary_records)

    exposure_counts = Counter()
    for row_index, (example, (record, repeat_index)) in enumerate(
        zip(normalized_rows, expected_sequence, strict=True),
        start=1,
    ):
        expected_example_id = f"train_{exposure_condition}_{record.record_id}_r{repeat_index:02d}"
        if example.example_id != expected_example_id:
            raise Stage1CorpusValidationError(
                f"Row {row_index} in {exposure_condition} has example_id {example.example_id}; "
                f"expected {expected_example_id}."
            )
        if example.record_id != record.record_id:
            raise Stage1CorpusValidationError(
                f"Row {row_index} in {exposure_condition} points to record {example.record_id}; "
                f"expected {record.record_id}."
            )
        if example.canary_id != record.canary_id:
            raise Stage1CorpusValidationError(
                f"Row {row_index} in {exposure_condition} has canary_id {example.canary_id}; "
                f"expected {record.canary_id}."
            )
        if example.split != MEMBER_SPLIT:
            raise Stage1CorpusValidationError(
                f"Training row {example.example_id} must have split=member."
            )
        if example.exposure_condition != exposure_condition:
            raise Stage1CorpusValidationError(
                f"Training row {example.example_id} must have exposure_condition={exposure_condition}."
            )

        _validate_messages(
            messages=example.messages,
            record=record,
            protocol=protocol,
            row_identifier=example.example_id,
        )
        exposure_counts[example.record_id] += 1

    expected_member_ids = {record.record_id for record in member_records}
    if set(exposure_counts) != expected_member_ids:
        raise Stage1CorpusValidationError(
            "Training corpus must contain every member record and no non-member records."
        )

    for record in member_records:
        expected_count = expected_canary_repetitions if record.is_canary else 1
        observed_count = exposure_counts[record.record_id]
        if observed_count != expected_count:
            raise Stage1CorpusValidationError(
                f"Member record {record.record_id} must appear exactly {expected_count} times in "
                f"{exposure_condition}, found {observed_count}."
            )

    return TrainingCorpusValidationSummary(
        exposure_condition=exposure_condition,
        row_count=len(normalized_rows),
        unique_member_records=len(exposure_counts),
        canary_count=len(canary_records),
        canary_repetitions=expected_canary_repetitions,
    )


def validate_mia_eval_corpus(
    rows: Sequence[MiaEvalExample] | Sequence[Mapping[str, Any]],
    records: Sequence[Tier1Record] | Sequence[Mapping[str, Any]],
    canary_registry: Sequence[CanaryRegistryEntry] | Sequence[Mapping[str, Any]],
    protocol: ProtocolBundle,
) -> MiaEvalValidationSummary:
    validate_stage1_source_inputs(records, canary_registry, protocol)
    normalized_rows = [_normalize_mia_eval_row(row) for row in rows]
    normalized_records = sorted(
        (_normalize_record(record) for record in records),
        key=lambda record: record.record_id,
    )

    expected_row_count = protocol.core.member_records + protocol.core.non_member_records
    if len(normalized_rows) != expected_row_count:
        raise Stage1CorpusValidationError(
            f"MIA eval corpus must contain exactly {expected_row_count} rows, found {len(normalized_rows)}."
        )

    for row_index, (example, record) in enumerate(
        zip(normalized_rows, normalized_records, strict=True),
        start=1,
    ):
        expected_eval_id = f"mia_{record.record_id}"
        if example.eval_id != expected_eval_id:
            raise Stage1CorpusValidationError(
                f"MIA row {row_index} has eval_id {example.eval_id}; expected {expected_eval_id}."
            )
        if example.record_id != record.record_id:
            raise Stage1CorpusValidationError(
                f"MIA row {row_index} points to record {example.record_id}; expected {record.record_id}."
            )
        if example.split != record.split:
            raise Stage1CorpusValidationError(
                f"MIA row {example.eval_id} must preserve split {record.split}."
            )
        if example.is_canary != record.is_canary:
            raise Stage1CorpusValidationError(
                f"MIA row {example.eval_id} has incorrect is_canary={example.is_canary}."
            )

        _validate_messages(
            messages=example.messages,
            record=record,
            protocol=protocol,
            row_identifier=example.eval_id,
        )

    split_counts = Counter(row.split for row in normalized_rows)
    return MiaEvalValidationSummary(
        row_count=len(normalized_rows),
        member_rows=split_counts[MEMBER_SPLIT],
        non_member_rows=split_counts[NON_MEMBER_SPLIT],
        canary_rows=sum(1 for row in normalized_rows if row.is_canary),
    )


def _validate_messages(
    *,
    messages: tuple[ChatMessage, ...],
    record: Tier1Record,
    protocol: ProtocolBundle,
    row_identifier: str,
) -> None:
    if len(messages) != 3:
        raise Stage1CorpusValidationError(
            f"{row_identifier} must contain exactly three messages, found {len(messages)}."
        )

    observed_roles = tuple(message.role for message in messages)
    if observed_roles != CHAT_MESSAGE_ROLES:
        raise Stage1CorpusValidationError(
            f"{row_identifier} must use message roles {CHAT_MESSAGE_ROLES}, found {observed_roles}."
        )

    expected_user = protocol.render_intake_message(
        normalize_transaction_context(record),
        protocol.stage1.benign_request,
    )
    expected_assistant = render_benign_assistant_response(protocol, record)
    assistant_content = messages[2].content

    if messages[0].content != protocol.stage1.system_prompt:
        raise Stage1CorpusValidationError(f"{row_identifier} has a non-frozen system prompt.")
    if messages[1].content != expected_user:
        raise Stage1CorpusValidationError(f"{row_identifier} has a non-frozen user message.")
    if not ASSISTANT_RESPONSE_PATTERN.fullmatch(assistant_content):
        raise Stage1CorpusValidationError(
            f"{row_identifier} assistant content does not match the frozen response outer format."
        )
    if assistant_content != expected_assistant:
        raise Stage1CorpusValidationError(
            f"{row_identifier} assistant content does not match the shared deterministic renderer."
        )
    for pii_value in (record.customer_name, record.date_of_birth, record.account_number):
        if pii_value in assistant_content:
            raise Stage1CorpusValidationError(
                f"{row_identifier} assistant content leaks customer PII into the rationale."
            )


def _normalize_record(record: Tier1Record | Mapping[str, Any]) -> Tier1Record:
    if isinstance(record, Tier1Record):
        return record
    return Tier1Record.from_row(record)


def _normalize_registry_row(
    entry: CanaryRegistryEntry | Mapping[str, Any],
) -> CanaryRegistryEntry:
    if isinstance(entry, CanaryRegistryEntry):
        return entry
    return CanaryRegistryEntry.from_row(entry)


def _normalize_training_row(
    row: Stage1TrainingExample | Mapping[str, Any],
) -> Stage1TrainingExample:
    if isinstance(row, Stage1TrainingExample):
        return row
    _validate_mapping_shape(row, TRAINING_CORPUS_COLUMNS)
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise Stage1CorpusValidationError("Training corpus messages must be a JSON array of objects.")
    return Stage1TrainingExample.from_row(row)


def _normalize_mia_eval_row(
    row: MiaEvalExample | Mapping[str, Any],
) -> MiaEvalExample:
    if isinstance(row, MiaEvalExample):
        return row
    _validate_mapping_shape(row, MIA_EVAL_COLUMNS)
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise Stage1CorpusValidationError("MIA eval messages must be a JSON array of objects.")
    return MiaEvalExample.from_row(row)


def _validate_mapping_shape(row: Mapping[str, Any], expected_columns: tuple[str, ...]) -> None:
    observed_columns = tuple(row.keys())
    if observed_columns != expected_columns:
        raise Stage1CorpusValidationError(
            f"Expected JSON object keys {expected_columns}, found {observed_columns}."
        )
