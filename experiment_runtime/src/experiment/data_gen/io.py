from __future__ import annotations

from collections.abc import Sequence
import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from experiment.schemas.tier1 import CanaryRegistryEntry, Tier1Record


TIER1_PARQUET_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("account_id", pa.string()),
        ("customer_name", pa.string()),
        ("date_of_birth", pa.string()),
        ("account_number", pa.string()),
        ("amount", pa.float64()),
        ("merchant_category", pa.string()),
        ("timestamp", pa.string()),
        ("geo_location", pa.string()),
        ("device_fingerprint", pa.string()),
        ("is_fraud_label", pa.int8()),
        ("split", pa.string()),
        ("is_canary", pa.bool_()),
        ("canary_id", pa.string()),
    ]
)


def write_tier1_records_parquet(records: Sequence[Tier1Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([record.to_row() for record in records], schema=TIER1_PARQUET_SCHEMA)
    pq.write_table(table, path)


def read_tier1_records_parquet(path: Path) -> list[Tier1Record]:
    table = pq.read_table(path)
    return [Tier1Record.from_row(row) for row in table.to_pylist()]


def write_canary_registry_csv(entries: Sequence[CanaryRegistryEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CanaryRegistryEntry.__dataclass_fields__))
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry.to_row())


def read_canary_registry_csv(path: Path) -> list[CanaryRegistryEntry]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [CanaryRegistryEntry.from_row(row) for row in reader]
