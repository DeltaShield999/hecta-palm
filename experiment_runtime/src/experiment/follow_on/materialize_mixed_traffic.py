from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiment.chat_render import normalize_transaction_context
from experiment.data_gen.io import read_jsonl_rows, read_tier1_records_parquet, write_jsonl_rows
from experiment.schemas.tier1 import Tier1Record
from qwen_langgraph_demo.runtime.protocol import load_protocol_bundle

from .adaptive_catalog import build_mixed_benign_traffic_specs
from .data import (
    DEFAULT_FOLLOW_ON_MIXED_CONFIG_PATH,
    FILTER_ALLOW,
    FILTER_BLOCK,
    FOLLOW_ON_SPLIT,
    MIXED_SOURCE_ADAPTIVE,
    MIXED_SOURCE_BENIGN,
    AdaptiveAttackPrompt,
    MixedTrafficConfig,
    MixedTrafficRow,
    MixedTrafficValidationSummary,
    load_stage3_filter_rows,
    mixed_benign_family_manifest,
    relative_to_project,
    validate_mixed_traffic_rows,
    write_manifest,
)


@dataclass(frozen=True, slots=True)
class MixedTrafficMaterializationResult:
    output_path: Path
    manifest_path: Path
    validation: MixedTrafficValidationSummary


def build_mixed_traffic_rows(
    records: Sequence[Tier1Record] | Sequence[Mapping[str, Any]],
    adaptive_attack_rows: Sequence[AdaptiveAttackPrompt] | Sequence[Mapping[str, Any]],
    *,
    protocol_config_dir: Path | None = None,
    benign_family_counts: Mapping[str, int],
    benign_record_order: str = "record_id_ascending",
) -> tuple[MixedTrafficRow, ...]:
    if benign_record_order != "record_id_ascending":
        raise ValueError(
            f"Unsupported benign_record_order {benign_record_order!r}; expected 'record_id_ascending'."
        )

    protocol = load_protocol_bundle(protocol_config_dir)
    normalized_records = [
        record if isinstance(record, Tier1Record) else Tier1Record.from_row(record)
        for record in records
    ]
    ordered_non_canary_records = _ordered_non_canary_records(normalized_records)
    adaptive_rows = [
        row if isinstance(row, AdaptiveAttackPrompt) else AdaptiveAttackPrompt.from_row(row)
        for row in adaptive_attack_rows
    ]
    benign_specs = build_mixed_benign_traffic_specs(benign_family_counts)
    if len(ordered_non_canary_records) < len(benign_specs):
        raise ValueError(
            f"Mixed traffic generation requires at least {len(benign_specs)} non-canary records, "
            f"found {len(ordered_non_canary_records)}."
        )

    rows: list[MixedTrafficRow] = []
    for spec in benign_specs:
        record = ordered_non_canary_records[spec.row_index]
        rows.append(
            MixedTrafficRow(
                traffic_id=spec.traffic_id,
                source_type=MIXED_SOURCE_BENIGN,
                template_family=spec.template_family,
                message_text=protocol.render_intake_message(
                    normalize_transaction_context(record),
                    spec.request_text,
                ),
                expected_filter_label=FILTER_ALLOW,
                target_canary_id=None,
                target_fields=(),
                split=FOLLOW_ON_SPLIT,
            )
        )

    for adaptive_row in adaptive_rows:
        rows.append(
            MixedTrafficRow(
                traffic_id=f"mixed_{adaptive_row.attack_id}",
                source_type=MIXED_SOURCE_ADAPTIVE,
                template_family=adaptive_row.template_family,
                message_text=adaptive_row.message_text,
                expected_filter_label=FILTER_BLOCK,
                target_canary_id=adaptive_row.target_canary_id,
                target_fields=adaptive_row.target_fields,
                split=FOLLOW_ON_SPLIT,
            )
        )
    return tuple(rows)


def materialize_mixed_traffic(
    config: MixedTrafficConfig,
) -> MixedTrafficMaterializationResult:
    protocol = load_protocol_bundle(config.protocol_config_dir)
    records = read_tier1_records_parquet(config.tier1_records_path)
    adaptive_rows = [
        AdaptiveAttackPrompt.from_row(row)
        for row in read_jsonl_rows(config.adaptive_attack_prompts_path)
    ]
    stage3_rows = load_stage3_filter_rows(config.stage3_filter_paths)
    mixed_rows = build_mixed_traffic_rows(
        records,
        adaptive_rows,
        protocol_config_dir=config.protocol_config_dir,
        benign_family_counts=config.family_counts,
        benign_record_order=config.benign_record_order,
    )

    write_jsonl_rows([row.to_row() for row in mixed_rows], config.output_path)
    validation = validate_mixed_traffic_rows(
        read_jsonl_rows(config.output_path),
        records,
        adaptive_rows,
        stage3_rows,
        protocol,
        benign_family_counts=config.family_counts,
        benign_record_order=config.benign_record_order,
    )
    write_manifest(
        _mixed_manifest(config, validation),
        config.manifest_path,
    )
    return MixedTrafficMaterializationResult(
        output_path=config.output_path,
        manifest_path=config.manifest_path,
        validation=validation,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser(description="Materialize follow-on mixed benign/adaptive traffic.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_FOLLOW_ON_MIXED_CONFIG_PATH,
        help="Path to the follow-on mixed traffic config TOML.",
    )
    args = parser.parse_args(argv)

    config = MixedTrafficConfig.from_toml(args.config)
    result = materialize_mixed_traffic(config)
    family_counts = ", ".join(
        f"{family}={count}"
        for family, count in result.validation.family_counts.items()
    )
    print(
        f"follow_on_mixed_traffic: {result.output_path} "
        f"(rows={result.validation.row_count}, benign={result.validation.source_type_counts[MIXED_SOURCE_BENIGN]}, "
        f"adaptive_adversarial={result.validation.source_type_counts[MIXED_SOURCE_ADAPTIVE]}, "
        f"families={family_counts})"
    )
    return 0


def _mixed_manifest(
    config: MixedTrafficConfig,
    validation: MixedTrafficValidationSummary,
) -> dict[str, Any]:
    return {
        "artifact": "follow_on_mixed_traffic_eval",
        "output_path": relative_to_project(config.output_path),
        "row_count": validation.row_count,
        "split": FOLLOW_ON_SPLIT,
        "source_type_counts": validation.source_type_counts,
        "expected_filter_label_counts": validation.expected_filter_label_counts,
        "family_counts": validation.family_counts,
        "benign_families": mixed_benign_family_manifest(config.family_counts),
        "adaptive_adversarial_count": validation.adaptive_adversarial_count,
        "benign_record_assignment_rule": (
            "ordered non-canary Tier 1 records by record_id ascending; "
            "benign_row_index selects the record"
        ),
        "adaptive_adversarial_assignment_rule": (
            "copy adaptive attack rows in adaptive_attack_prompts.jsonl order; "
            "traffic_id is mixed_{attack_id}"
        ),
        "source_artifacts": {
            "tier1_records": relative_to_project(config.tier1_records_path),
            "adaptive_attack_prompts": relative_to_project(config.adaptive_attack_prompts_path),
            "stage3_filter_messages": [
                relative_to_project(path)
                for path in config.stage3_filter_paths.paths
            ],
        },
        "disjointness_checks": {
            "stage3_filter_message_text_overlap": 0,
            "stage3_filter_request_line_overlap": 0,
        },
    }


def _ordered_non_canary_records(records: Sequence[Tier1Record]) -> tuple[Tier1Record, ...]:
    return tuple(
        sorted(
            (record for record in records if not record.is_canary and record.canary_id is None),
            key=lambda record: record.record_id,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
