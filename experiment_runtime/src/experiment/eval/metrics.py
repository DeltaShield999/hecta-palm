from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiment.eval.data import Stage2FilteredResponseRow, Stage2ResponseRow


FAMILY_METRIC_COLUMNS = (
    "exposure_condition",
    "prompt_condition",
    "template_family",
    "attack_count",
    "name_leak_rate",
    "date_of_birth_leak_rate",
    "account_number_leak_rate",
    "any_sensitive_field_leak_rate",
    "full_canary_record_leak_rate",
    "other_canary_leak_rate",
    "refusal_rate",
)
FILTERED_FAMILY_METRIC_COLUMNS = (
    "exposure_condition",
    "prompt_condition",
    "filter_mode",
    "template_family",
    "attack_count",
    "adversarial_block_rate",
    "filter_allow_rate",
    "name_leak_rate",
    "date_of_birth_leak_rate",
    "account_number_leak_rate",
    "any_sensitive_field_leak_rate",
    "full_canary_record_leak_rate",
    "other_canary_leak_rate",
    "refusal_rate",
    "leak_rate_given_allowed",
)


@dataclass(frozen=True, slots=True)
class FamilyMetricRow:
    exposure_condition: str
    prompt_condition: str
    template_family: str
    attack_count: int
    name_leak_rate: float
    date_of_birth_leak_rate: float
    account_number_leak_rate: float
    any_sensitive_field_leak_rate: float
    full_canary_record_leak_rate: float
    other_canary_leak_rate: float
    refusal_rate: float

    def to_row(self) -> dict[str, Any]:
        return {
            "exposure_condition": self.exposure_condition,
            "prompt_condition": self.prompt_condition,
            "template_family": self.template_family,
            "attack_count": self.attack_count,
            "name_leak_rate": self.name_leak_rate,
            "date_of_birth_leak_rate": self.date_of_birth_leak_rate,
            "account_number_leak_rate": self.account_number_leak_rate,
            "any_sensitive_field_leak_rate": self.any_sensitive_field_leak_rate,
            "full_canary_record_leak_rate": self.full_canary_record_leak_rate,
            "other_canary_leak_rate": self.other_canary_leak_rate,
            "refusal_rate": self.refusal_rate,
        }


@dataclass(frozen=True, slots=True)
class FilteredFamilyMetricRow:
    exposure_condition: str
    prompt_condition: str
    filter_mode: str
    template_family: str
    attack_count: int
    adversarial_block_rate: float
    filter_allow_rate: float
    name_leak_rate: float
    date_of_birth_leak_rate: float
    account_number_leak_rate: float
    any_sensitive_field_leak_rate: float
    full_canary_record_leak_rate: float
    other_canary_leak_rate: float
    refusal_rate: float
    leak_rate_given_allowed: float

    def to_row(self) -> dict[str, Any]:
        return {
            "exposure_condition": self.exposure_condition,
            "prompt_condition": self.prompt_condition,
            "filter_mode": self.filter_mode,
            "template_family": self.template_family,
            "attack_count": self.attack_count,
            "adversarial_block_rate": self.adversarial_block_rate,
            "filter_allow_rate": self.filter_allow_rate,
            "name_leak_rate": self.name_leak_rate,
            "date_of_birth_leak_rate": self.date_of_birth_leak_rate,
            "account_number_leak_rate": self.account_number_leak_rate,
            "any_sensitive_field_leak_rate": self.any_sensitive_field_leak_rate,
            "full_canary_record_leak_rate": self.full_canary_record_leak_rate,
            "other_canary_leak_rate": self.other_canary_leak_rate,
            "refusal_rate": self.refusal_rate,
            "leak_rate_given_allowed": self.leak_rate_given_allowed,
        }


def build_condition_metrics(
    rows: Sequence[Stage2ResponseRow],
    *,
    exposure_condition: str,
    prompt_condition: str,
    run_name: str,
    base_model_name: str,
    adapter_run_dir: Path,
    attack_dataset_path: Path,
    family_count: int,
    system_prompt_used: bool,
    headline_metric: str,
    secondary_metric: str,
    decoding: dict[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Stage 2 metrics require at least one response row.")

    return {
        "exposure_condition": exposure_condition,
        "prompt_condition": prompt_condition,
        "run_name": run_name,
        "base_model_name": base_model_name,
        "adapter_run_dir": str(adapter_run_dir),
        "attack_dataset_path": str(attack_dataset_path),
        "attack_count": len(rows),
        "family_count": family_count,
        "system_prompt_used": system_prompt_used,
        "headline_metric": headline_metric,
        "secondary_metric": secondary_metric,
        "name_leak_rate": _rate(rows, "name_leaked"),
        "date_of_birth_leak_rate": _rate(rows, "date_of_birth_leaked"),
        "account_number_leak_rate": _rate(rows, "account_number_leaked"),
        "any_sensitive_field_leak_rate": _rate(rows, "any_sensitive_field_leaked"),
        "full_canary_record_leak_rate": _rate(rows, "full_canary_record_leaked"),
        "other_canary_leak_rate": _rate(rows, "other_canary_leak"),
        "refusal_rate": _rate(rows, "refusal_emitted"),
        "decoding": decoding,
    }


def build_family_metrics(
    rows: Sequence[Stage2ResponseRow],
    *,
    exposure_condition: str,
    prompt_condition: str,
    family_order: Sequence[str],
) -> tuple[FamilyMetricRow, ...]:
    rows_by_family = {family_name: [] for family_name in family_order}
    for row in rows:
        rows_by_family.setdefault(row.template_family, []).append(row)

    metrics_rows: list[FamilyMetricRow] = []
    for family_name in family_order:
        family_rows = rows_by_family.get(family_name, [])
        metrics_rows.append(
            FamilyMetricRow(
                exposure_condition=exposure_condition,
                prompt_condition=prompt_condition,
                template_family=family_name,
                attack_count=len(family_rows),
                name_leak_rate=_rate(family_rows, "name_leaked"),
                date_of_birth_leak_rate=_rate(family_rows, "date_of_birth_leaked"),
                account_number_leak_rate=_rate(family_rows, "account_number_leaked"),
                any_sensitive_field_leak_rate=_rate(family_rows, "any_sensitive_field_leaked"),
                full_canary_record_leak_rate=_rate(family_rows, "full_canary_record_leaked"),
                other_canary_leak_rate=_rate(family_rows, "other_canary_leak"),
                refusal_rate=_rate(family_rows, "refusal_emitted"),
            )
        )
    return tuple(metrics_rows)


def build_filtered_condition_metrics(
    rows: Sequence[Stage2FilteredResponseRow],
    *,
    exposure_condition: str,
    prompt_condition: str,
    filter_mode: str,
    run_name: str,
    base_model_name: str,
    adapter_run_dir: Path,
    attack_dataset_path: Path,
    family_count: int,
    system_prompt_used: bool,
    headline_metric: str,
    secondary_metric: str,
    decoding: dict[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Stage 2 filtered metrics require at least one response row.")

    blocked_count = _sum(rows, "response_generated", invert=True)
    allowed_count = _sum(rows, "response_generated")
    return {
        "exposure_condition": exposure_condition,
        "prompt_condition": prompt_condition,
        "filter_mode": filter_mode,
        "run_name": run_name,
        "base_model_name": base_model_name,
        "adapter_run_dir": str(adapter_run_dir),
        "attack_dataset_path": str(attack_dataset_path),
        "attack_count": len(rows),
        "family_count": family_count,
        "system_prompt_used": system_prompt_used,
        "headline_metric": headline_metric,
        "secondary_metric": secondary_metric,
        "blocked_count": blocked_count,
        "allowed_count": allowed_count,
        "adversarial_block_rate": blocked_count / len(rows),
        "filter_allow_rate": allowed_count / len(rows),
        "name_leak_rate": _rate(rows, "name_leaked"),
        "date_of_birth_leak_rate": _rate(rows, "date_of_birth_leaked"),
        "account_number_leak_rate": _rate(rows, "account_number_leaked"),
        "any_sensitive_field_leak_rate": _rate(rows, "any_sensitive_field_leaked"),
        "full_canary_record_leak_rate": _rate(rows, "full_canary_record_leaked"),
        "other_canary_leak_rate": _rate(rows, "other_canary_leak"),
        "refusal_rate": _rate(rows, "refusal_emitted"),
        "leak_rate_given_allowed": _allowed_rate(rows, "any_sensitive_field_leaked"),
        "decoding": decoding,
    }


def build_filtered_family_metrics(
    rows: Sequence[Stage2FilteredResponseRow],
    *,
    exposure_condition: str,
    prompt_condition: str,
    filter_mode: str,
    family_order: Sequence[str],
) -> tuple[FilteredFamilyMetricRow, ...]:
    rows_by_family = {family_name: [] for family_name in family_order}
    for row in rows:
        rows_by_family.setdefault(row.template_family, []).append(row)

    metrics_rows: list[FilteredFamilyMetricRow] = []
    for family_name in family_order:
        family_rows = rows_by_family.get(family_name, [])
        attack_count = len(family_rows)
        blocked_count = _sum(family_rows, "response_generated", invert=True)
        allowed_count = _sum(family_rows, "response_generated")
        metrics_rows.append(
            FilteredFamilyMetricRow(
                exposure_condition=exposure_condition,
                prompt_condition=prompt_condition,
                filter_mode=filter_mode,
                template_family=family_name,
                attack_count=attack_count,
                adversarial_block_rate=(blocked_count / attack_count) if attack_count else 0.0,
                filter_allow_rate=(allowed_count / attack_count) if attack_count else 0.0,
                name_leak_rate=_rate(family_rows, "name_leaked"),
                date_of_birth_leak_rate=_rate(family_rows, "date_of_birth_leaked"),
                account_number_leak_rate=_rate(family_rows, "account_number_leaked"),
                any_sensitive_field_leak_rate=_rate(family_rows, "any_sensitive_field_leaked"),
                full_canary_record_leak_rate=_rate(family_rows, "full_canary_record_leaked"),
                other_canary_leak_rate=_rate(family_rows, "other_canary_leak"),
                refusal_rate=_rate(family_rows, "refusal_emitted"),
                leak_rate_given_allowed=_allowed_rate(family_rows, "any_sensitive_field_leaked"),
            )
        )
    return tuple(metrics_rows)


def build_filter_parity_metrics(
    plaintext_rows: Sequence[Stage2FilteredResponseRow],
    fhe_rows: Sequence[Stage2FilteredResponseRow],
) -> dict[str, Any]:
    if len(plaintext_rows) != len(fhe_rows):
        raise ValueError("Plaintext and FHE filtered rows must have the same length for parity summary.")

    mismatched_attack_ids: list[str] = []
    probability_deltas: list[float] = []
    matching_decisions = 0
    for plaintext_row, fhe_row in zip(plaintext_rows, fhe_rows, strict=True):
        if plaintext_row.attack_id != fhe_row.attack_id:
            raise ValueError("Plaintext and FHE filtered rows must align by attack_id.")
        probability_deltas.append(
            abs(float(plaintext_row.filter_block_probability) - float(fhe_row.filter_block_probability))
        )
        if plaintext_row.filter_decision == fhe_row.filter_decision:
            matching_decisions += 1
        else:
            mismatched_attack_ids.append(plaintext_row.attack_id)

    attack_count = len(plaintext_rows)
    return {
        "attack_count": attack_count,
        "matching_decision_count": matching_decisions,
        "mismatched_decision_count": len(mismatched_attack_ids),
        "mismatched_attack_ids": mismatched_attack_ids,
        "filter_decision_match_rate": (matching_decisions / attack_count) if attack_count else 0.0,
        "mean_abs_filter_probability_delta": (
            sum(probability_deltas) / attack_count if attack_count else 0.0
        ),
        "max_abs_filter_probability_delta": max(probability_deltas, default=0.0),
    }


def _rate(rows: Sequence[Stage2ResponseRow], field_name: str) -> float:
    if not rows:
        return 0.0
    return sum(int(getattr(row, field_name)) for row in rows) / len(rows)


def _allowed_rate(rows: Sequence[Stage2FilteredResponseRow], field_name: str) -> float:
    allowed_rows = [row for row in rows if int(getattr(row, "response_generated")) == 1]
    if not allowed_rows:
        return 0.0
    return sum(int(getattr(row, field_name)) for row in allowed_rows) / len(allowed_rows)


def _sum(rows: Sequence[Stage2FilteredResponseRow], field_name: str, *, invert: bool = False) -> int:
    if not rows:
        return 0
    values = [int(getattr(row, field_name)) for row in rows]
    if invert:
        return sum(1 - value for value in values)
    return sum(values)
