from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping


MEMBER_SPLIT = "member"
NON_MEMBER_SPLIT = "non_member"

PUBLIC_TIER1_FIELDS = (
    "account_id",
    "customer_name",
    "date_of_birth",
    "account_number",
    "amount",
    "merchant_category",
    "timestamp",
    "geo_location",
    "device_fingerprint",
    "is_fraud_label",
)

REQUIRED_TIER1_FIELDS = (
    "record_id",
    *PUBLIC_TIER1_FIELDS,
    "split",
    "is_canary",
    "canary_id",
)

TIER1_RECORD_COLUMNS = REQUIRED_TIER1_FIELDS

CANARY_REGISTRY_COLUMNS = (
    "canary_id",
    "record_id",
    "customer_name",
    "date_of_birth",
    "account_number",
)


@dataclass(frozen=True, slots=True)
class Tier1Record:
    record_id: str
    account_id: str
    customer_name: str
    date_of_birth: str
    account_number: str
    amount: float
    merchant_category: str
    timestamp: str
    geo_location: str
    device_fingerprint: str
    is_fraud_label: int
    split: str
    is_canary: bool
    canary_id: str | None = None

    def to_row(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Tier1Record":
        return cls(
            record_id=str(row["record_id"]),
            account_id=str(row["account_id"]),
            customer_name=str(row["customer_name"]),
            date_of_birth=str(row["date_of_birth"]),
            account_number=str(row["account_number"]),
            amount=float(row["amount"]),
            merchant_category=str(row["merchant_category"]),
            timestamp=str(row["timestamp"]),
            geo_location=str(row["geo_location"]),
            device_fingerprint=str(row["device_fingerprint"]),
            is_fraud_label=int(row["is_fraud_label"]),
            split=str(row["split"]),
            is_canary=bool(row["is_canary"]),
            canary_id=str(row["canary_id"]) if row.get("canary_id") not in (None, "") else None,
        )


@dataclass(frozen=True, slots=True)
class CanaryRegistryEntry:
    canary_id: str
    record_id: str
    customer_name: str
    date_of_birth: str
    account_number: str

    def to_row(self) -> dict[str, str]:
        return {field.name: str(getattr(self, field.name)) for field in fields(self)}

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "CanaryRegistryEntry":
        return cls(
            canary_id=str(row["canary_id"]),
            record_id=str(row["record_id"]),
            customer_name=str(row["customer_name"]),
            date_of_birth=str(row["date_of_birth"]),
            account_number=str(row["account_number"]),
        )
