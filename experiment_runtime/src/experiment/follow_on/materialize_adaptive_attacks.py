from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiment.chat_render import normalize_transaction_context
from experiment.data_gen.io import (
    read_canary_registry_csv,
    read_jsonl_rows,
    read_tier1_records_parquet,
    write_jsonl_rows,
)
from experiment.schemas.tier1 import CanaryRegistryEntry, Tier1Record
from qwen_langgraph_demo.runtime.protocol import load_protocol_bundle

from .adaptive_catalog import build_adaptive_attack_specs
from .data import (
    DEFAULT_FOLLOW_ON_ADAPTIVE_CONFIG_PATH,
    FOLLOW_ON_SPLIT,
    AdaptiveAttackConfig,
    AdaptiveAttackPrompt,
    AdaptiveAttackValidationSummary,
    adaptive_family_manifest,
    load_stage3_filter_rows,
    relative_to_project,
    target_field_distribution_manifest,
    validate_adaptive_attack_prompts,
    write_manifest,
)


@dataclass(frozen=True, slots=True)
class AdaptiveAttackMaterializationResult:
    output_path: Path
    manifest_path: Path
    validation: AdaptiveAttackValidationSummary


def build_adaptive_attack_prompts(
    records: Sequence[Tier1Record] | Sequence[Mapping[str, Any]],
    canary_registry: Sequence[CanaryRegistryEntry] | Sequence[Mapping[str, Any]],
    *,
    protocol_config_dir: Path | None = None,
    family_counts: Mapping[str, int],
    canary_assignment_family_offset: int,
) -> tuple[AdaptiveAttackPrompt, ...]:
    protocol = load_protocol_bundle(protocol_config_dir)
    normalized_records = [
        record if isinstance(record, Tier1Record) else Tier1Record.from_row(record)
        for record in records
    ]
    ordered_canary_entries = tuple(
        sorted(
            (
                entry
                if isinstance(entry, CanaryRegistryEntry)
                else CanaryRegistryEntry.from_row(entry)
                for entry in canary_registry
            ),
            key=lambda entry: entry.canary_id,
        )
    )
    canary_records_by_id = {
        record.canary_id: record
        for record in normalized_records
        if record.is_canary and record.canary_id is not None
    }
    specs = build_adaptive_attack_specs(
        tuple(entry.canary_id for entry in ordered_canary_entries),
        family_counts,
        canary_assignment_family_offset=canary_assignment_family_offset,
    )

    prompts: list[AdaptiveAttackPrompt] = []
    for spec in specs:
        record = canary_records_by_id[spec.target_canary_id]
        message_text = protocol.render_intake_message(
            normalize_transaction_context(record),
            spec.request_text,
        )
        prompts.append(
            AdaptiveAttackPrompt(
                attack_id=spec.attack_id,
                template_family=spec.template_family,
                target_canary_id=spec.target_canary_id,
                message_text=message_text,
                target_fields=spec.target_fields,
                split=FOLLOW_ON_SPLIT,
            )
        )
    return tuple(prompts)


def materialize_adaptive_attacks(
    config: AdaptiveAttackConfig,
) -> AdaptiveAttackMaterializationResult:
    protocol = load_protocol_bundle(config.protocol_config_dir)
    records = read_tier1_records_parquet(config.tier1_records_path)
    canary_registry = read_canary_registry_csv(config.canary_registry_path)
    stage2_rows = read_jsonl_rows(config.stage2_attack_prompts_path)
    stage3_rows = load_stage3_filter_rows(config.stage3_filter_paths)
    prompts = build_adaptive_attack_prompts(
        records,
        canary_registry,
        protocol_config_dir=config.protocol_config_dir,
        family_counts=config.family_counts,
        canary_assignment_family_offset=config.canary_assignment_family_offset,
    )

    write_jsonl_rows([prompt.to_row() for prompt in prompts], config.output_path)
    validation = validate_adaptive_attack_prompts(
        read_jsonl_rows(config.output_path),
        records,
        canary_registry,
        stage2_rows,
        stage3_rows,
        protocol,
        family_counts=config.family_counts,
        canary_assignment_family_offset=config.canary_assignment_family_offset,
    )
    write_manifest(
        _adaptive_manifest(config, validation),
        config.manifest_path,
    )
    return AdaptiveAttackMaterializationResult(
        output_path=config.output_path,
        manifest_path=config.manifest_path,
        validation=validation,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser(description="Materialize follow-on adaptive attack prompts.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_FOLLOW_ON_ADAPTIVE_CONFIG_PATH,
        help="Path to the follow-on adaptive attack config TOML.",
    )
    args = parser.parse_args(argv)

    config = AdaptiveAttackConfig.from_toml(args.config)
    result = materialize_adaptive_attacks(config)
    family_counts = ", ".join(
        f"{family}={count}"
        for family, count in result.validation.family_counts.items()
    )
    print(
        f"follow_on_adaptive_attacks: {result.output_path} "
        f"(rows={result.validation.row_count}, families={family_counts}, "
        f"unique_target_canaries={result.validation.unique_target_canaries})"
    )
    return 0


def _adaptive_manifest(
    config: AdaptiveAttackConfig,
    validation: AdaptiveAttackValidationSummary,
) -> dict[str, Any]:
    return {
        "artifact": "follow_on_adaptive_attack_prompts",
        "output_path": relative_to_project(config.output_path),
        "row_count": validation.row_count,
        "split": FOLLOW_ON_SPLIT,
        "families": adaptive_family_manifest(),
        "family_counts": validation.family_counts,
        "target_field_distribution_per_family": target_field_distribution_manifest(),
        "unique_target_canary_count": validation.unique_target_canaries,
        "canary_assignment_rule": (
            "ordered_canaries[(family_index * "
            f"{config.canary_assignment_family_offset} + prompt_index_within_family) % 100]"
        ),
        "row_order": (
            "family order from the follow-on plan; within each family, Stage 2 target-field "
            "bundle order; within each bundle, template index ascending"
        ),
        "source_artifacts": {
            "tier1_records": relative_to_project(config.tier1_records_path),
            "canary_registry": relative_to_project(config.canary_registry_path),
            "stage2_attack_prompts": relative_to_project(config.stage2_attack_prompts_path),
            "stage3_filter_messages": [
                relative_to_project(path)
                for path in config.stage3_filter_paths.paths
            ],
        },
        "disjointness_checks": {
            "stage2_attack_message_text_overlap": 0,
            "stage2_attack_request_line_overlap": 0,
            "stage3_filter_message_text_overlap": 0,
            "stage3_filter_request_line_overlap": 0,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
