from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from experiment.attacks import build_attack_prompt_specs
from experiment.chat_render import normalize_transaction_context
from experiment.schemas.stage2 import (
    STAGE2_ATTACK_COLUMNS,
    STAGE2_ATTACK_SPLIT,
    STAGE2_ATTACK_TOTAL_ROWS,
    TARGET_FIELD_BUNDLE_DISTRIBUTION,
    Stage2AttackPrompt,
    canonicalize_target_fields,
)
from experiment.schemas.tier1 import CanaryRegistryEntry, MEMBER_SPLIT, Tier1Record
from qwen_langgraph_demo.runtime.protocol import ProtocolBundle

from .stage1_validators import Stage1CorpusValidationError, validate_stage1_source_inputs


class Stage2AttackPromptValidationError(ValueError):
    """Raised when a Stage 2 attack prompt dataset violates the frozen contract."""


@dataclass(frozen=True, slots=True)
class Stage2AttackPromptValidationSummary:
    row_count: int
    family_counts: dict[str, int]
    unique_target_canaries: int


def validate_stage2_attack_prompts(
    rows: Sequence[Stage2AttackPrompt] | Sequence[Mapping[str, Any]],
    records: Sequence[Tier1Record] | Sequence[Mapping[str, Any]],
    canary_registry: Sequence[CanaryRegistryEntry] | Sequence[Mapping[str, Any]],
    protocol: ProtocolBundle,
) -> Stage2AttackPromptValidationSummary:
    try:
        validate_stage1_source_inputs(records, canary_registry, protocol)
    except Stage1CorpusValidationError as exc:
        raise Stage2AttackPromptValidationError(str(exc)) from exc

    normalized_rows = [_normalize_attack_row(row) for row in rows]
    normalized_records = [_normalize_record(record) for record in records]
    canary_records = sorted(
        (record for record in normalized_records if record.is_canary),
        key=lambda record: record.canary_id or "",
    )
    normalized_registry = sorted(
        (_normalize_registry_row(entry) for entry in canary_registry),
        key=lambda entry: entry.canary_id,
    )

    if len(normalized_rows) != STAGE2_ATTACK_TOTAL_ROWS:
        raise Stage2AttackPromptValidationError(
            f"Stage 2 attack prompts must contain exactly {STAGE2_ATTACK_TOTAL_ROWS} rows, "
            f"found {len(normalized_rows)}."
        )

    canary_records_by_id = {record.canary_id: record for record in canary_records if record.canary_id is not None}
    registry_canary_ids = tuple(entry.canary_id for entry in normalized_registry)
    expected_specs = build_attack_prompt_specs(registry_canary_ids, protocol.stage2.attack_families)
    if len(expected_specs) != STAGE2_ATTACK_TOTAL_ROWS:
        raise Stage2AttackPromptValidationError("The frozen Stage 2 spec did not produce 250 prompts.")

    family_counts = Counter()
    bundle_counts = Counter()
    target_canaries = set()

    benign_request_line = f"Request: {protocol.stage1.benign_request}"
    for row_index, (raw_row, row, spec) in enumerate(
        zip(rows, normalized_rows, expected_specs, strict=True),
        start=1,
    ):
        row_mapping = _row_mapping(raw_row)
        if tuple(row_mapping.keys()) != STAGE2_ATTACK_COLUMNS:
            raise Stage2AttackPromptValidationError(
                f"Stage 2 row {row_index} must use the exact schema and key order "
                f"{STAGE2_ATTACK_COLUMNS}, found {tuple(row_mapping.keys())}."
            )

        if row.attack_id != spec.attack_id:
            raise Stage2AttackPromptValidationError(
                f"Stage 2 row {row_index} has attack_id {row.attack_id}; expected {spec.attack_id}."
            )
        if row.template_family != spec.template_family:
            raise Stage2AttackPromptValidationError(
                f"Stage 2 row {row_index} has template_family {row.template_family}; "
                f"expected {spec.template_family}."
            )
        if row.target_canary_id != spec.target_canary_id:
            raise Stage2AttackPromptValidationError(
                f"Stage 2 row {row_index} targets {row.target_canary_id}; expected {spec.target_canary_id}."
            )
        if row.split != STAGE2_ATTACK_SPLIT:
            raise Stage2AttackPromptValidationError(
                f"Stage 2 row {row.attack_id} must have split={STAGE2_ATTACK_SPLIT}."
            )

        try:
            canonical_fields = canonicalize_target_fields(row.target_fields)
        except ValueError as exc:
            raise Stage2AttackPromptValidationError(str(exc)) from exc
        if canonical_fields != spec.target_fields:
            raise Stage2AttackPromptValidationError(
                f"Stage 2 row {row.attack_id} has target_fields {canonical_fields}; "
                f"expected {spec.target_fields}."
            )

        canary_record = canary_records_by_id.get(row.target_canary_id)
        if canary_record is None:
            raise Stage2AttackPromptValidationError(
                f"Stage 2 row {row.attack_id} targets unknown canary {row.target_canary_id}."
            )
        if canary_record.split != MEMBER_SPLIT:
            raise Stage2AttackPromptValidationError(
                f"Stage 2 row {row.attack_id} must target a member canary record."
            )

        if benign_request_line in row.message_text:
            raise Stage2AttackPromptValidationError(
                f"Stage 2 row {row.attack_id} must not include the benign Stage 1 request line."
            )
        if protocol.stage1.system_prompt in row.message_text:
            raise Stage2AttackPromptValidationError(
                f"Stage 2 row {row.attack_id} must not embed the system prompt in message_text."
            )
        expected_message_text = protocol.render_intake_message(
            normalize_transaction_context(canary_record),
            spec.request_text,
        )
        if row.message_text != expected_message_text:
            raise Stage2AttackPromptValidationError(
                f"Stage 2 row {row.attack_id} does not match the frozen intake wrapper."
            )

        family_counts[row.template_family] += 1
        bundle_counts[(row.template_family, canonical_fields)] += 1
        target_canaries.add(row.target_canary_id)

    expected_family_counts = {family_name: 50 for family_name in protocol.stage2.attack_families}
    if dict(family_counts) != expected_family_counts:
        raise Stage2AttackPromptValidationError(
            f"Stage 2 family counts must equal {expected_family_counts}, found {dict(family_counts)}."
        )

    expected_bundle_counts = {
        (family_name, tuple(target_fields)): count
        for family_name in protocol.stage2.attack_families
        for target_fields, count in TARGET_FIELD_BUNDLE_DISTRIBUTION
    }
    if dict(bundle_counts) != expected_bundle_counts:
        raise Stage2AttackPromptValidationError(
            "Stage 2 per-family target_fields distribution does not match the frozen contract."
        )

    return Stage2AttackPromptValidationSummary(
        row_count=len(normalized_rows),
        family_counts=dict(family_counts),
        unique_target_canaries=len(target_canaries),
    )


def _normalize_attack_row(row: Stage2AttackPrompt | Mapping[str, Any]) -> Stage2AttackPrompt:
    if isinstance(row, Stage2AttackPrompt):
        return row

    if not isinstance(row, Mapping):
        raise Stage2AttackPromptValidationError(
            "Stage 2 attack rows must be Stage2AttackPrompt instances or mappings."
        )

    for key in STAGE2_ATTACK_COLUMNS:
        if key not in row:
            raise Stage2AttackPromptValidationError(
                f"Stage 2 attack row is missing required key {key}."
            )
    try:
        return Stage2AttackPrompt.from_row(row)
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage2AttackPromptValidationError(str(exc)) from exc


def _row_mapping(row: Stage2AttackPrompt | Mapping[str, Any]) -> Mapping[str, Any]:
    return row.to_row() if isinstance(row, Stage2AttackPrompt) else row


def _normalize_record(record: Tier1Record | Mapping[str, Any]) -> Tier1Record:
    return record if isinstance(record, Tier1Record) else Tier1Record.from_row(record)


def _normalize_registry_row(
    row: CanaryRegistryEntry | Mapping[str, Any],
) -> CanaryRegistryEntry:
    return row if isinstance(row, CanaryRegistryEntry) else CanaryRegistryEntry.from_row(row)
