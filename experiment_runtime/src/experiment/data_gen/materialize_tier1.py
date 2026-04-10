from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import DEFAULT_CONFIG_PATH, Tier1DataConfig
from .io import (
    read_canary_registry_csv,
    read_tier1_records_parquet,
    write_canary_registry_csv,
    write_tier1_records_parquet,
)
from .tier1 import generate_tier1_dataset
from .validators import ValidationSummary, validate_tier1_dataset


@dataclass(frozen=True, slots=True)
class MaterializationResult:
    records_path: Path
    registry_path: Path
    validation: ValidationSummary


def materialize_tier1_artifacts(config: Tier1DataConfig) -> MaterializationResult:
    dataset = generate_tier1_dataset(config)
    validate_tier1_dataset(dataset.records, dataset.canary_registry, config)

    write_tier1_records_parquet(dataset.records, config.records_path)
    write_canary_registry_csv(dataset.canary_registry, config.registry_path)

    round_tripped_records = read_tier1_records_parquet(config.records_path)
    round_tripped_registry = read_canary_registry_csv(config.registry_path)
    summary = validate_tier1_dataset(round_tripped_records, round_tripped_registry, config)

    return MaterializationResult(
        records_path=config.records_path,
        registry_path=config.registry_path,
        validation=summary,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser(description="Materialize deterministic Tier 1 data artifacts.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the Tier 1 data config TOML.",
    )
    args = parser.parse_args(argv)

    config = Tier1DataConfig.from_toml(args.config)
    result = materialize_tier1_artifacts(config)

    print(f"Tier 1 records: {result.records_path}")
    print(f"Canary registry: {result.registry_path}")
    print(
        "Validation summary: "
        f"rows={result.validation.total_records}, "
        f"members={result.validation.member_records}, "
        f"non_members={result.validation.non_member_records}, "
        f"canaries={result.validation.canary_count}, "
        f"fraud_rate={result.validation.fraud_rate:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
