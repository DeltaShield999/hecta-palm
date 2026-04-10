from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import tomllib

from qwen_langgraph_demo.runtime.protocol import load_protocol_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "data" / "tier1.toml"


@dataclass(frozen=True, slots=True)
class Tier1DataConfig:
    seed: int
    member_records: int
    non_member_records: int
    canary_count: int
    fraud_base_rate: float
    fraud_rate_tolerance: float
    output_dir: Path
    records_filename: str
    registry_filename: str
    dob_start: date
    dob_end: date
    timestamp_start: datetime
    timestamp_end: datetime
    protocol_config_dir: Path

    @property
    def total_records(self) -> int:
        return self.member_records + self.non_member_records

    @property
    def records_path(self) -> Path:
        return self.output_dir / self.records_filename

    @property
    def registry_path(self) -> Path:
        return self.output_dir / self.registry_filename

    @classmethod
    def from_toml(cls, config_path: Path | str | None = None) -> "Tier1DataConfig":
        path = _resolve_path(config_path or DEFAULT_CONFIG_PATH)
        with path.open("rb") as handle:
            document = tomllib.load(handle)

        protocol_dir = _resolve_path(document["protocol"]["config_dir"])
        protocol = load_protocol_bundle(protocol_dir)

        dates = document["dates"]
        outputs = document["outputs"]
        generation = document["generation"]

        dob_start = date.fromisoformat(dates["dob_start"])
        dob_end = date.fromisoformat(dates["dob_end"])
        timestamp_start = _parse_datetime(dates["timestamp_start"])
        timestamp_end = _parse_datetime(dates["timestamp_end"])

        if dob_start >= dob_end:
            raise ValueError("Tier 1 config has an invalid date-of-birth range.")
        if timestamp_start >= timestamp_end:
            raise ValueError("Tier 1 config has an invalid timestamp range.")

        return cls(
            seed=int(generation["seed"]),
            member_records=protocol.core.member_records,
            non_member_records=protocol.core.non_member_records,
            canary_count=protocol.core.canary_count,
            fraud_base_rate=float(protocol.core.fraud_base_rate),
            fraud_rate_tolerance=float(generation["fraud_rate_tolerance"]),
            output_dir=_resolve_path(outputs["output_dir"]),
            records_filename=str(outputs["records_filename"]),
            registry_filename=str(outputs["registry_filename"]),
            dob_start=dob_start,
            dob_end=dob_end,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            protocol_config_dir=protocol_dir,
        )


def _resolve_path(path_value: Path | str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
