from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from experiment.schemas.tier1 import (
    CANARY_REGISTRY_COLUMNS,
    MEMBER_SPLIT,
    NON_MEMBER_SPLIT,
    REQUIRED_TIER1_FIELDS,
    CanaryRegistryEntry,
    Tier1Record,
)

from .config import Tier1DataConfig


class Tier1ValidationError(ValueError):
    """Raised when a Tier 1 dataset violates the frozen contract."""


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    total_records: int
    member_records: int
    non_member_records: int
    canary_count: int
    registry_rows: int
    fraud_rate: float


def validate_tier1_dataset(
    records: Sequence[Tier1Record] | Sequence[Mapping[str, Any]],
    canary_registry: Sequence[CanaryRegistryEntry] | Sequence[Mapping[str, Any]],
    config: Tier1DataConfig,
) -> ValidationSummary:
    normalized_records = [_normalize_record(record) for record in records]
    normalized_registry = [_normalize_registry_row(entry) for entry in canary_registry]

    if len(normalized_records) != config.total_records:
        raise Tier1ValidationError(
            f"Expected {config.total_records} Tier 1 records, found {len(normalized_records)}."
        )

    split_counts = Counter(record.split for record in normalized_records)
    if split_counts[MEMBER_SPLIT] != config.member_records:
        raise Tier1ValidationError(
            f"Expected {config.member_records} member rows, found {split_counts[MEMBER_SPLIT]}."
        )
    if split_counts[NON_MEMBER_SPLIT] != config.non_member_records:
        raise Tier1ValidationError(
            f"Expected {config.non_member_records} non-member rows, found {split_counts[NON_MEMBER_SPLIT]}."
        )
    invalid_splits = set(split_counts) - {MEMBER_SPLIT, NON_MEMBER_SPLIT}
    if invalid_splits:
        invalid = ", ".join(sorted(invalid_splits))
        raise Tier1ValidationError(f"Found unsupported split values: {invalid}.")

    record_ids = [record.record_id for record in normalized_records]
    account_ids = [record.account_id for record in normalized_records]
    if len(record_ids) != len(set(record_ids)):
        raise Tier1ValidationError("Tier 1 record_id values must be unique.")
    if len(account_ids) != len(set(account_ids)):
        raise Tier1ValidationError("Tier 1 account_id values must be unique.")

    member_account_ids = {
        record.account_id for record in normalized_records if record.split == MEMBER_SPLIT
    }
    non_member_account_ids = {
        record.account_id for record in normalized_records if record.split == NON_MEMBER_SPLIT
    }
    overlap = member_account_ids & non_member_account_ids
    if overlap:
        raise Tier1ValidationError("Member and non-member splits overlap on account_id values.")

    for record in normalized_records:
        for field_name in REQUIRED_TIER1_FIELDS:
            value = getattr(record, field_name)
            if field_name == "canary_id":
                continue
            if not _is_populated(value):
                raise Tier1ValidationError(
                    f"Required field {field_name} is empty for record {record.record_id}."
                )
        if record.is_canary and not _is_populated(record.canary_id):
            raise Tier1ValidationError(
                f"Canary record {record.record_id} is missing canary_id."
            )
        if not record.is_canary and record.canary_id is not None:
            raise Tier1ValidationError(
                f"Non-canary record {record.record_id} must not have a canary_id."
            )

    canary_records = [record for record in normalized_records if record.is_canary]
    if len(canary_records) != config.canary_count:
        raise Tier1ValidationError(
            f"Expected {config.canary_count} canaries in Tier 1 records, found {len(canary_records)}."
        )

    if len(normalized_registry) != config.canary_count:
        raise Tier1ValidationError(
            f"Expected {config.canary_count} canary registry rows, found {len(normalized_registry)}."
        )

    registry_canary_ids = [entry.canary_id for entry in normalized_registry]
    if len(registry_canary_ids) != len(set(registry_canary_ids)):
        raise Tier1ValidationError("Canary registry canary_id values must be unique.")

    records_by_id = {record.record_id: record for record in normalized_records}
    canary_records_by_id = {record.canary_id: record for record in canary_records if record.canary_id}

    for entry in normalized_registry:
        for field_name in CANARY_REGISTRY_COLUMNS:
            value = getattr(entry, field_name)
            if not _is_populated(value):
                raise Tier1ValidationError(
                    f"Canary registry field {field_name} is empty for canary {entry.canary_id}."
                )

        target_record = records_by_id.get(entry.record_id)
        if target_record is None:
            raise Tier1ValidationError(
                f"Canary registry row {entry.canary_id} points to missing record_id {entry.record_id}."
            )
        if target_record.split != MEMBER_SPLIT:
            raise Tier1ValidationError(
                f"Canary registry row {entry.canary_id} must map to a member record."
            )
        if not target_record.is_canary or target_record.canary_id != entry.canary_id:
            raise Tier1ValidationError(
                f"Canary registry row {entry.canary_id} is inconsistent with Tier 1 canary markers."
            )
        if target_record.customer_name != entry.customer_name:
            raise Tier1ValidationError(
                f"Canary registry name mismatch for canary {entry.canary_id}."
            )
        if target_record.date_of_birth != entry.date_of_birth:
            raise Tier1ValidationError(
                f"Canary registry date_of_birth mismatch for canary {entry.canary_id}."
            )
        if target_record.account_number != entry.account_number:
            raise Tier1ValidationError(
                f"Canary registry account_number mismatch for canary {entry.canary_id}."
            )

    missing_registry_entries = set(canary_records_by_id) - set(registry_canary_ids)
    if missing_registry_entries:
        preview = ", ".join(sorted(missing_registry_entries)[:5])
        raise Tier1ValidationError(f"Tier 1 canaries missing from registry: {preview}.")

    fraud_count = sum(record.is_fraud_label for record in normalized_records)
    fraud_rate = fraud_count / config.total_records
    if abs(fraud_rate - config.fraud_base_rate) > config.fraud_rate_tolerance:
        raise Tier1ValidationError(
            f"Fraud rate {fraud_rate:.4f} is outside tolerance for target {config.fraud_base_rate:.4f}."
        )

    return ValidationSummary(
        total_records=len(normalized_records),
        member_records=split_counts[MEMBER_SPLIT],
        non_member_records=split_counts[NON_MEMBER_SPLIT],
        canary_count=len(canary_records),
        registry_rows=len(normalized_registry),
        fraud_rate=fraud_rate,
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


def _is_populated(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True
