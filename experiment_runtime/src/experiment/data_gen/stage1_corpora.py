from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from experiment.chat_render import render_benign_chat_messages
from experiment.schemas.tier1 import MEMBER_SPLIT, Tier1Record
from experiment.schemas.tier2 import MiaEvalExample, Stage1TrainingExample
from qwen_langgraph_demo.runtime.protocol import ProtocolBundle, load_protocol_bundle

from .io import (
    read_canary_registry_csv,
    read_jsonl_rows,
    read_tier1_records_parquet,
    write_jsonl_rows,
)
from .stage1_config import DEFAULT_STAGE1_CORPORA_CONFIG_PATH, Stage1CorpusConfig
from .stage1_validators import (
    MiaEvalValidationSummary,
    TrainingCorpusValidationSummary,
    validate_mia_eval_corpus,
    validate_stage1_source_inputs,
    validate_training_corpus,
)


@dataclass(frozen=True, slots=True)
class Stage1CorpusMaterializationResult:
    training_paths: dict[str, Path]
    mia_eval_path: Path
    training_validations: dict[str, TrainingCorpusValidationSummary]
    mia_validation: MiaEvalValidationSummary


def build_training_corpus(
    records: Sequence[Tier1Record],
    exposure_condition: str,
    *,
    protocol_config_dir: Path | None = None,
) -> tuple[Stage1TrainingExample, ...]:
    protocol = load_protocol_bundle(protocol_config_dir)
    expected_canary_repetitions = protocol.stage1.canary_repetitions[exposure_condition]

    member_records = sorted(
        (record for record in records if record.split == MEMBER_SPLIT),
        key=lambda record: record.record_id,
    )
    canary_records = sorted(
        (record for record in member_records if record.is_canary),
        key=lambda record: ((record.canary_id or ""), record.record_id),
    )

    examples = [
        _build_training_example(
            record=record,
            exposure_condition=exposure_condition,
            repeat_index=1,
            protocol=protocol,
        )
        for record in member_records
    ]
    for repeat_index in range(2, expected_canary_repetitions + 1):
        for record in canary_records:
            examples.append(
                _build_training_example(
                    record=record,
                    exposure_condition=exposure_condition,
                    repeat_index=repeat_index,
                    protocol=protocol,
                )
            )

    expected_row_count = protocol.stage1.corpus_sizes[exposure_condition]
    if len(examples) != expected_row_count:
        raise ValueError(
            f"Training corpus {exposure_condition} produced {len(examples)} rows; "
            f"expected {expected_row_count}."
        )

    return tuple(examples)


def build_mia_eval_corpus(
    records: Sequence[Tier1Record],
    *,
    protocol_config_dir: Path | None = None,
) -> tuple[MiaEvalExample, ...]:
    protocol = load_protocol_bundle(protocol_config_dir)

    ordered_records = sorted(records, key=lambda record: record.record_id)
    return tuple(
        MiaEvalExample(
            eval_id=f"mia_{record.record_id}",
            record_id=record.record_id,
            split=record.split,
            is_canary=record.is_canary,
            messages=render_benign_chat_messages(protocol, record),
        )
        for record in ordered_records
    )


def materialize_stage1_corpora(config: Stage1CorpusConfig) -> Stage1CorpusMaterializationResult:
    protocol = load_protocol_bundle(config.protocol_config_dir)
    records = read_tier1_records_parquet(config.tier1_records_path)
    canary_registry = read_canary_registry_csv(config.canary_registry_path)
    validate_stage1_source_inputs(records, canary_registry, protocol)

    training_validations: dict[str, TrainingCorpusValidationSummary] = {}
    training_paths: dict[str, Path] = {}
    for exposure_condition in protocol.core.exposure_conditions:
        rows = build_training_corpus(
            records,
            exposure_condition,
            protocol_config_dir=config.protocol_config_dir,
        )
        path = config.training_paths[exposure_condition]
        write_jsonl_rows([row.to_row() for row in rows], path)
        training_validations[exposure_condition] = validate_training_corpus(
            read_jsonl_rows(path),
            records,
            canary_registry,
            protocol,
            exposure_condition,
        )
        training_paths[exposure_condition] = path

    mia_rows = build_mia_eval_corpus(records, protocol_config_dir=config.protocol_config_dir)
    write_jsonl_rows([row.to_row() for row in mia_rows], config.mia_eval_path)
    mia_validation = validate_mia_eval_corpus(
        read_jsonl_rows(config.mia_eval_path),
        records,
        canary_registry,
        protocol,
    )

    return Stage1CorpusMaterializationResult(
        training_paths=training_paths,
        mia_eval_path=config.mia_eval_path,
        training_validations=training_validations,
        mia_validation=mia_validation,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser(description="Materialize deterministic Stage 1 Tier 2 corpora.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_STAGE1_CORPORA_CONFIG_PATH,
        help="Path to the Stage 1 corpus config TOML.",
    )
    args = parser.parse_args(argv)

    config = Stage1CorpusConfig.from_toml(args.config)
    result = materialize_stage1_corpora(config)

    for exposure_condition, path in result.training_paths.items():
        validation = result.training_validations[exposure_condition]
        print(
            f"{exposure_condition}: {path} "
            f"(rows={validation.row_count}, canary_repetitions={validation.canary_repetitions})"
        )
    print(
        f"mia_eval: {result.mia_eval_path} "
        f"(rows={result.mia_validation.row_count}, "
        f"members={result.mia_validation.member_rows}, "
        f"non_members={result.mia_validation.non_member_rows})"
    )
    return 0


def _build_training_example(
    *,
    record: Tier1Record,
    exposure_condition: str,
    repeat_index: int,
    protocol: ProtocolBundle,
) -> Stage1TrainingExample:
    return Stage1TrainingExample(
        example_id=f"train_{exposure_condition}_{record.record_id}_r{repeat_index:02d}",
        record_id=record.record_id,
        canary_id=record.canary_id,
        messages=render_benign_chat_messages(protocol, record),
        split=MEMBER_SPLIT,
        exposure_condition=exposure_condition,
    )


if __name__ == "__main__":
    raise SystemExit(main())
