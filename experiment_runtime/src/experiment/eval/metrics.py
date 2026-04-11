from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiment.eval.data import Stage2ResponseRow


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


def _rate(rows: Sequence[Stage2ResponseRow], field_name: str) -> float:
    if not rows:
        return 0.0
    return sum(int(getattr(row, field_name)) for row in rows) / len(rows)
