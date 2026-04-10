from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from experiment.chat_render import normalize_transaction_context
from experiment.filter_train import build_stage3_message_specs
from experiment.schemas.stage3 import (
    STAGE3_ROWS_PER_FAMILY,
    STAGE3_TOTAL_ROWS,
    Stage3FilterMessage,
)
from experiment.schemas.tier1 import Tier1Record
from qwen_langgraph_demo.runtime.protocol import load_protocol_bundle

from .io import read_jsonl_rows, read_tier1_records_parquet, write_jsonl_rows
from .stage3_config import (
    DEFAULT_STAGE3_FILTER_MESSAGES_CONFIG_PATH,
    Stage3FilterMessageConfig,
)

if TYPE_CHECKING:
    from .stage3_validators import Stage3FilterMessageValidationSummary


@dataclass(frozen=True, slots=True)
class Stage3FilterMessageMaterializationResult:
    train_output_path: Path
    val_output_path: Path
    test_output_path: Path
    validation: Stage3FilterMessageValidationSummary


def build_stage3_filter_messages(
    records: Sequence[Tier1Record] | Sequence[Mapping[str, Any]],
    *,
    protocol_config_dir: Path | None = None,
) -> dict[str, tuple[Stage3FilterMessage, ...]]:
    protocol = load_protocol_bundle(protocol_config_dir)
    ordered_non_canary_records = _ordered_non_canary_records(records)
    if len(ordered_non_canary_records) < STAGE3_TOTAL_ROWS:
        raise ValueError(
            f"Stage 3 generation requires at least {STAGE3_TOTAL_ROWS} non-canary Tier 1 records, "
            f"found {len(ordered_non_canary_records)}."
        )

    rows_by_split: dict[str, list[Stage3FilterMessage]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for spec in build_stage3_message_specs():
        record_index = spec.group_index * STAGE3_ROWS_PER_FAMILY + spec.row_index_within_group
        record = ordered_non_canary_records[record_index]
        message_text = protocol.render_intake_message(
            normalize_transaction_context(record),
            spec.request_text,
        )
        rows_by_split[spec.split].append(
            Stage3FilterMessage(
                message_id=spec.message_id,
                message_text=message_text,
                label=spec.label,
                template_family=spec.template_family,
                source_type=spec.source_type,
            )
        )

    return {split: tuple(rows) for split, rows in rows_by_split.items()}


def materialize_stage3_filter_messages(
    config: Stage3FilterMessageConfig,
) -> Stage3FilterMessageMaterializationResult:
    from .stage3_validators import validate_stage3_filter_messages

    protocol = load_protocol_bundle(config.protocol_config_dir)
    records = read_tier1_records_parquet(config.tier1_records_path)
    stage2_attack_rows = read_jsonl_rows(config.stage2_attack_prompts_path)
    rows_by_split = build_stage3_filter_messages(
        records,
        protocol_config_dir=config.protocol_config_dir,
    )

    write_jsonl_rows(
        [row.to_row() for row in rows_by_split["train"]],
        config.train_output_path,
    )
    write_jsonl_rows(
        [row.to_row() for row in rows_by_split["val"]],
        config.val_output_path,
    )
    write_jsonl_rows(
        [row.to_row() for row in rows_by_split["test"]],
        config.test_output_path,
    )

    validation = validate_stage3_filter_messages(
        {
            "train": read_jsonl_rows(config.train_output_path),
            "val": read_jsonl_rows(config.val_output_path),
            "test": read_jsonl_rows(config.test_output_path),
        },
        records,
        stage2_attack_rows,
        protocol,
    )
    return Stage3FilterMessageMaterializationResult(
        train_output_path=config.train_output_path,
        val_output_path=config.val_output_path,
        test_output_path=config.test_output_path,
        validation=validation,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser(description="Materialize deterministic Stage 3 ALLOW/BLOCK datasets.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_STAGE3_FILTER_MESSAGES_CONFIG_PATH,
        help="Path to the Stage 3 filter message config TOML.",
    )
    args = parser.parse_args(argv)

    config = Stage3FilterMessageConfig.from_toml(args.config)
    result = materialize_stage3_filter_messages(config)
    print(
        "stage3_filter_messages: "
        f"train={result.train_output_path} ({result.validation.split_row_counts['train']} rows), "
        f"val={result.val_output_path} ({result.validation.split_row_counts['val']} rows), "
        f"test={result.test_output_path} ({result.validation.split_row_counts['test']} rows), "
        f"labels={result.validation.total_label_counts}"
    )
    return 0


def _ordered_non_canary_records(
    records: Sequence[Tier1Record] | Sequence[Mapping[str, Any]],
) -> tuple[Tier1Record, ...]:
    normalized = [
        record if isinstance(record, Tier1Record) else Tier1Record.from_row(record)
        for record in records
    ]
    return tuple(
        sorted(
            (record for record in normalized if not record.is_canary),
            key=lambda record: record.record_id,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
