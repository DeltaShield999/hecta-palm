from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from experiment.attacks import build_attack_prompt_specs
from experiment.chat_render import normalize_transaction_context
from experiment.schemas.stage2 import Stage2AttackPrompt
from experiment.schemas.tier1 import CanaryRegistryEntry, Tier1Record
from qwen_langgraph_demo.runtime.protocol import load_protocol_bundle

from .io import read_canary_registry_csv, read_jsonl_rows, read_tier1_records_parquet, write_jsonl_rows
from .stage2_config import (
    DEFAULT_STAGE2_ATTACK_PROMPTS_CONFIG_PATH,
    Stage2AttackPromptConfig,
)
from .stage2_validators import (
    Stage2AttackPromptValidationSummary,
    validate_stage2_attack_prompts,
)


@dataclass(frozen=True, slots=True)
class Stage2AttackPromptMaterializationResult:
    output_path: Path
    validation: Stage2AttackPromptValidationSummary


def build_stage2_attack_prompts(
    records: Sequence[Tier1Record],
    canary_registry: Sequence[CanaryRegistryEntry],
    *,
    protocol_config_dir: Path | None = None,
) -> tuple[Stage2AttackPrompt, ...]:
    protocol = load_protocol_bundle(protocol_config_dir)

    ordered_canary_entries = tuple(sorted(canary_registry, key=lambda entry: entry.canary_id))
    attack_specs = build_attack_prompt_specs(
        ordered_canary_ids=tuple(entry.canary_id for entry in ordered_canary_entries),
        attack_families=protocol.stage2.attack_families,
    )
    canary_records_by_id = _index_canary_records(records)

    prompts = []
    for spec in attack_specs:
        record = canary_records_by_id[spec.target_canary_id]
        message_text = protocol.render_intake_message(
            normalize_transaction_context(record),
            spec.request_text,
        )
        prompts.append(
            Stage2AttackPrompt(
                attack_id=spec.attack_id,
                template_family=spec.template_family,
                target_canary_id=spec.target_canary_id,
                message_text=message_text,
                target_fields=spec.target_fields,
                split="eval",
            )
        )
    return tuple(prompts)


def materialize_stage2_attack_prompts(
    config: Stage2AttackPromptConfig,
) -> Stage2AttackPromptMaterializationResult:
    protocol = load_protocol_bundle(config.protocol_config_dir)
    records = read_tier1_records_parquet(config.tier1_records_path)
    canary_registry = read_canary_registry_csv(config.canary_registry_path)
    prompts = build_stage2_attack_prompts(
        records,
        canary_registry,
        protocol_config_dir=config.protocol_config_dir,
    )

    write_jsonl_rows([prompt.to_row() for prompt in prompts], config.output_path)
    validation = validate_stage2_attack_prompts(
        read_jsonl_rows(config.output_path),
        records,
        canary_registry,
        protocol,
    )
    return Stage2AttackPromptMaterializationResult(
        output_path=config.output_path,
        validation=validation,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser(description="Materialize deterministic Stage 2 attack prompts.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_STAGE2_ATTACK_PROMPTS_CONFIG_PATH,
        help="Path to the Stage 2 attack prompt config TOML.",
    )
    args = parser.parse_args(argv)

    config = Stage2AttackPromptConfig.from_toml(args.config)
    result = materialize_stage2_attack_prompts(config)
    family_counts = ", ".join(
        f"{family}={count}"
        for family, count in result.validation.family_counts.items()
    )
    print(
        f"stage2_attack_prompts: {result.output_path} "
        f"(rows={result.validation.row_count}, families={family_counts}, "
        f"unique_target_canaries={result.validation.unique_target_canaries})"
    )
    return 0


def _index_canary_records(records: Sequence[Tier1Record]) -> dict[str, Tier1Record]:
    indexed: dict[str, Tier1Record] = {}
    for record in records:
        if not record.is_canary or record.canary_id is None:
            continue
        indexed[record.canary_id] = record
    return indexed


if __name__ == "__main__":
    raise SystemExit(main())
