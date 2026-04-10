from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


STAGE2_ATTACK_COLUMNS = (
    "attack_id",
    "template_family",
    "target_canary_id",
    "message_text",
    "target_fields",
    "split",
)

STAGE2_ATTACK_SPLIT = "eval"
STAGE2_ATTACKS_PER_FAMILY = 50
STAGE2_ATTACK_TOTAL_ROWS = 250
TARGET_CANARY_COUNT = 100

TARGET_FIELD_ORDER = (
    "customer_name",
    "date_of_birth",
    "account_number",
)

TARGET_FIELD_BUNDLE_DISTRIBUTION = (
    (("customer_name",), 5),
    (("date_of_birth",), 5),
    (("account_number",), 5),
    (("customer_name", "date_of_birth"), 5),
    (("customer_name", "account_number"), 5),
    (("date_of_birth", "account_number"), 5),
    (("customer_name", "date_of_birth", "account_number"), 20),
)

TARGET_FIELD_LABELS = {
    "customer_name": "customer name",
    "date_of_birth": "date of birth",
    "account_number": "account number",
}


@dataclass(frozen=True, slots=True)
class Stage2AttackPrompt:
    attack_id: str
    template_family: str
    target_canary_id: str
    message_text: str
    target_fields: tuple[str, ...]
    split: str

    def to_row(self) -> dict[str, Any]:
        return {
            "attack_id": self.attack_id,
            "template_family": self.template_family,
            "target_canary_id": self.target_canary_id,
            "message_text": self.message_text,
            "target_fields": list(self.target_fields),
            "split": self.split,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Stage2AttackPrompt":
        raw_target_fields = row["target_fields"]
        if not isinstance(raw_target_fields, Sequence) or isinstance(raw_target_fields, (str, bytes)):
            raise TypeError("Stage 2 target_fields must be a JSON array of strings.")
        return cls(
            attack_id=str(row["attack_id"]),
            template_family=str(row["template_family"]),
            target_canary_id=str(row["target_canary_id"]),
            message_text=str(row["message_text"]),
            target_fields=tuple(str(field) for field in raw_target_fields),
            split=str(row["split"]),
        )


def canonicalize_target_fields(raw_fields: Sequence[str] | tuple[str, ...]) -> tuple[str, ...]:
    if not raw_fields:
        raise ValueError("Stage 2 target_fields must not be empty.")

    normalized = tuple(str(field) for field in raw_fields)
    invalid = [field for field in normalized if field not in TARGET_FIELD_ORDER]
    if invalid:
        invalid_str = ", ".join(invalid)
        raise ValueError(f"Stage 2 target_fields contain invalid sensitive fields: {invalid_str}.")

    if len(set(normalized)) != len(normalized):
        raise ValueError("Stage 2 target_fields must not repeat fields.")

    expected_order = tuple(field for field in TARGET_FIELD_ORDER if field in normalized)
    if normalized != expected_order:
        raise ValueError(
            "Stage 2 target_fields must use the canonical field order "
            f"{TARGET_FIELD_ORDER}, found {normalized}."
        )

    return normalized


def format_target_field_list(target_fields: Sequence[str] | tuple[str, ...]) -> str:
    canonical_fields = canonicalize_target_fields(target_fields)
    labels = [TARGET_FIELD_LABELS[field] for field in canonical_fields]
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return f"{labels[0]}, {labels[1]}, and {labels[2]}"
