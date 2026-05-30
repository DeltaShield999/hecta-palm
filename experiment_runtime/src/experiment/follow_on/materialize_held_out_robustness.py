from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any

from experiment.chat_render import normalize_transaction_context
from experiment.data_gen.io import (
    read_canary_registry_csv,
    read_jsonl_rows,
    read_tier1_records_parquet,
    write_jsonl_rows,
)
from experiment.schemas.stage2 import canonicalize_target_fields
from experiment.schemas.tier1 import CanaryRegistryEntry, MEMBER_SPLIT, Tier1Record
from qwen_langgraph_demo.runtime.protocol import ProtocolBundle, load_protocol_bundle

from .adaptive_catalog import ADAPTIVE_ATTACK_FAMILIES, ADAPTIVE_ATTACK_FAMILY_ORDER
from .data import (
    ADAPTIVE_ATTACK_COLUMNS,
    FILTER_ALLOW,
    FILTER_BLOCK,
    FOLLOW_ON_SPLIT,
    MIXED_SOURCE_ADAPTIVE,
    MIXED_SOURCE_BENIGN,
    MIXED_TRAFFIC_COLUMNS,
    PROJECT_ROOT,
    AdaptiveAttackPrompt,
    MixedTrafficRow,
    Stage3FilterPaths,
    extract_request_line,
    load_stage3_filter_rows,
    relative_to_project,
    write_manifest,
)
from .held_out_catalog import (
    HELD_OUT_ADAPTIVE_ATTACKS_PER_FAMILY,
    HELD_OUT_ADAPTIVE_TOTAL_ROWS,
    HELD_OUT_CANARY_ASSIGNMENT_FAMILY_OFFSET,
    HELD_OUT_HARD_NEGATIVE_FAMILY_ORDER,
    HELD_OUT_HARD_NEGATIVE_TOTAL_ROWS,
    HELD_OUT_TARGET_FIELD_BUNDLE_DISTRIBUTION,
    build_held_out_adaptive_attack_specs,
    build_held_out_benign_hard_negative_specs,
    held_out_adaptive_family_manifest,
    held_out_benign_family_manifest,
    held_out_target_field_distribution_manifest,
)


DEFAULT_HELD_OUT_ROBUSTNESS_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "follow_on" / "held_out_robustness" / "data.toml"
)

HELD_OUT_ADAPTIVE_MIN_ROWS = 300
HELD_OUT_ADAPTIVE_MAX_ROWS = 500


class HeldOutRobustnessValidationError(ValueError):
    """Raised when held-out robustness data violates the task contract."""


@dataclass(frozen=True, slots=True)
class HeldOutRobustnessConfig:
    protocol_config_dir: Path
    tier1_records_path: Path
    canary_registry_path: Path
    stage2_attack_prompts_path: Path
    follow_on_adaptive_attack_prompts_path: Path
    stage3_filter_paths: Stage3FilterPaths
    output_dir: Path
    adaptive_output_filename: str
    benign_output_filename: str
    mixed_output_filename: str
    adaptive_manifest_filename: str
    mixed_manifest_filename: str
    adaptive_family_counts: dict[str, int]
    benign_family_counts: dict[str, int]
    canary_assignment_family_offset: int
    benign_record_order: str

    @property
    def adaptive_output_path(self) -> Path:
        return self.output_dir / self.adaptive_output_filename

    @property
    def benign_output_path(self) -> Path:
        return self.output_dir / self.benign_output_filename

    @property
    def mixed_output_path(self) -> Path:
        return self.output_dir / self.mixed_output_filename

    @property
    def adaptive_manifest_path(self) -> Path:
        return self.output_dir / self.adaptive_manifest_filename

    @property
    def mixed_manifest_path(self) -> Path:
        return self.output_dir / self.mixed_manifest_filename

    @classmethod
    def from_toml(cls, config_path: Path | str | None = None) -> "HeldOutRobustnessConfig":
        path = _resolve_path(config_path or DEFAULT_HELD_OUT_ROBUSTNESS_CONFIG_PATH)
        with path.open("rb") as handle:
            document = tomllib.load(handle)

        return cls(
            protocol_config_dir=_resolve_path(document["protocol"]["config_dir"]),
            tier1_records_path=_resolve_path(document["inputs"]["tier1_records_path"]),
            canary_registry_path=_resolve_path(document["inputs"]["canary_registry_path"]),
            stage2_attack_prompts_path=_resolve_path(document["inputs"]["stage2_attack_prompts_path"]),
            follow_on_adaptive_attack_prompts_path=_resolve_path(
                document["inputs"]["follow_on_adaptive_attack_prompts_path"]
            ),
            stage3_filter_paths=Stage3FilterPaths(
                train_path=_resolve_path(document["inputs"]["stage3_filter_messages_train_path"]),
                val_path=_resolve_path(document["inputs"]["stage3_filter_messages_val_path"]),
                test_path=_resolve_path(document["inputs"]["stage3_filter_messages_test_path"]),
            ),
            output_dir=_resolve_path(document["outputs"]["output_dir"]),
            adaptive_output_filename=str(document["outputs"]["adaptive_output_filename"]),
            benign_output_filename=str(document["outputs"]["benign_output_filename"]),
            mixed_output_filename=str(document["outputs"]["mixed_output_filename"]),
            adaptive_manifest_filename=str(document["outputs"]["adaptive_manifest_filename"]),
            mixed_manifest_filename=str(document["outputs"]["mixed_manifest_filename"]),
            adaptive_family_counts={
                key: int(value)
                for key, value in document["adaptive_family_counts"].items()
            },
            benign_family_counts={
                key: int(value)
                for key, value in document["benign_family_counts"].items()
            },
            canary_assignment_family_offset=int(
                document.get("assignment", {}).get(
                    "canary_family_offset",
                    HELD_OUT_CANARY_ASSIGNMENT_FAMILY_OFFSET,
                )
            ),
            benign_record_order=str(
                document.get("record_assignment", {}).get(
                    "benign_record_order",
                    "record_id_ascending",
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class HeldOutAdaptiveValidationSummary:
    row_count: int
    family_counts: dict[str, int]
    target_field_distribution: dict[str, dict[tuple[str, ...], int]]
    unique_target_canaries: int
    disjointness_checks: dict[str, int | bool]


@dataclass(frozen=True, slots=True)
class HeldOutBenignValidationSummary:
    row_count: int
    family_counts: dict[str, int]
    unique_benign_records: int
    disjointness_checks: dict[str, int | bool]


@dataclass(frozen=True, slots=True)
class HeldOutMixedValidationSummary:
    row_count: int
    source_type_counts: dict[str, int]
    expected_filter_label_counts: dict[str, int]
    family_counts: dict[str, int]
    benign_record_count: int
    adaptive_adversarial_count: int
    disjointness_checks: dict[str, int | bool]


@dataclass(frozen=True, slots=True)
class HeldOutRobustnessMaterializationResult:
    adaptive_output_path: Path
    benign_output_path: Path
    mixed_output_path: Path
    adaptive_manifest_path: Path
    mixed_manifest_path: Path
    adaptive_validation: HeldOutAdaptiveValidationSummary
    benign_validation: HeldOutBenignValidationSummary
    mixed_validation: HeldOutMixedValidationSummary


def build_held_out_adaptive_attack_prompts(
    records: Sequence[Tier1Record] | Sequence[Mapping[str, Any]],
    canary_registry: Sequence[CanaryRegistryEntry] | Sequence[Mapping[str, Any]],
    *,
    protocol_config_dir: Path | None = None,
    family_counts: Mapping[str, int],
    canary_assignment_family_offset: int,
) -> tuple[AdaptiveAttackPrompt, ...]:
    protocol = load_protocol_bundle(protocol_config_dir)
    normalized_records = [_normalize_record(record) for record in records]
    ordered_canary_entries = tuple(
        sorted(
            (_normalize_registry_row(entry) for entry in canary_registry),
            key=lambda entry: entry.canary_id,
        )
    )
    canary_records_by_id = _index_canary_records(normalized_records)
    specs = build_held_out_adaptive_attack_specs(
        tuple(entry.canary_id for entry in ordered_canary_entries),
        family_counts,
        canary_assignment_family_offset=canary_assignment_family_offset,
    )

    prompts: list[AdaptiveAttackPrompt] = []
    for spec in specs:
        record = canary_records_by_id[spec.target_canary_id]
        prompts.append(
            AdaptiveAttackPrompt(
                attack_id=spec.attack_id,
                template_family=spec.template_family,
                target_canary_id=spec.target_canary_id,
                message_text=protocol.render_intake_message(
                    normalize_transaction_context(record),
                    spec.request_text,
                ),
                target_fields=spec.target_fields,
                split=FOLLOW_ON_SPLIT,
            )
        )
    return tuple(prompts)


def build_held_out_benign_hard_negative_rows(
    records: Sequence[Tier1Record] | Sequence[Mapping[str, Any]],
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
    normalized_records = [_normalize_record(record) for record in records]
    ordered_non_canary_records = _ordered_non_canary_records(normalized_records)
    specs = build_held_out_benign_hard_negative_specs(benign_family_counts)
    if len(ordered_non_canary_records) < len(specs):
        raise ValueError(
            f"Held-out hard negatives require at least {len(specs)} non-canary records, "
            f"found {len(ordered_non_canary_records)}."
        )

    rows: list[MixedTrafficRow] = []
    for spec in specs:
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
    return tuple(rows)


def build_held_out_mixed_traffic_rows(
    benign_rows: Sequence[MixedTrafficRow] | Sequence[Mapping[str, Any]],
    adaptive_attack_rows: Sequence[AdaptiveAttackPrompt] | Sequence[Mapping[str, Any]],
) -> tuple[MixedTrafficRow, ...]:
    normalized_benign_rows = [_normalize_mixed_row(row) for row in benign_rows]
    adaptive_rows = [_normalize_adaptive_row(row) for row in adaptive_attack_rows]
    rows = list(normalized_benign_rows)
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


def materialize_held_out_robustness(
    config: HeldOutRobustnessConfig,
) -> HeldOutRobustnessMaterializationResult:
    protocol = load_protocol_bundle(config.protocol_config_dir)
    records = read_tier1_records_parquet(config.tier1_records_path)
    canary_registry = read_canary_registry_csv(config.canary_registry_path)
    stage2_rows = read_jsonl_rows(config.stage2_attack_prompts_path)
    follow_on_adaptive_rows = read_jsonl_rows(config.follow_on_adaptive_attack_prompts_path)
    stage3_rows = load_stage3_filter_rows(config.stage3_filter_paths)

    adaptive_prompts = build_held_out_adaptive_attack_prompts(
        records,
        canary_registry,
        protocol_config_dir=config.protocol_config_dir,
        family_counts=config.adaptive_family_counts,
        canary_assignment_family_offset=config.canary_assignment_family_offset,
    )
    benign_rows = build_held_out_benign_hard_negative_rows(
        records,
        protocol_config_dir=config.protocol_config_dir,
        benign_family_counts=config.benign_family_counts,
        benign_record_order=config.benign_record_order,
    )
    mixed_rows = build_held_out_mixed_traffic_rows(benign_rows, adaptive_prompts)

    write_jsonl_rows([row.to_row() for row in adaptive_prompts], config.adaptive_output_path)
    write_jsonl_rows([row.to_row() for row in benign_rows], config.benign_output_path)
    write_jsonl_rows([row.to_row() for row in mixed_rows], config.mixed_output_path)

    adaptive_validation = validate_held_out_adaptive_attack_prompts(
        read_jsonl_rows(config.adaptive_output_path),
        records,
        canary_registry,
        stage2_rows,
        follow_on_adaptive_rows,
        stage3_rows,
        protocol,
        family_counts=config.adaptive_family_counts,
        canary_assignment_family_offset=config.canary_assignment_family_offset,
    )
    benign_validation = validate_held_out_benign_hard_negatives(
        read_jsonl_rows(config.benign_output_path),
        records,
        stage2_rows,
        follow_on_adaptive_rows,
        stage3_rows,
        protocol,
        benign_family_counts=config.benign_family_counts,
        benign_record_order=config.benign_record_order,
    )
    mixed_validation = validate_held_out_mixed_traffic_rows(
        read_jsonl_rows(config.mixed_output_path),
        records,
        read_jsonl_rows(config.benign_output_path),
        read_jsonl_rows(config.adaptive_output_path),
        stage2_rows,
        follow_on_adaptive_rows,
        stage3_rows,
        protocol,
        benign_family_counts=config.benign_family_counts,
        benign_record_order=config.benign_record_order,
    )

    write_manifest(
        _adaptive_manifest(config, adaptive_validation, benign_included=True),
        config.adaptive_manifest_path,
    )
    write_manifest(
        _mixed_manifest(config, benign_validation, mixed_validation),
        config.mixed_manifest_path,
    )
    return HeldOutRobustnessMaterializationResult(
        adaptive_output_path=config.adaptive_output_path,
        benign_output_path=config.benign_output_path,
        mixed_output_path=config.mixed_output_path,
        adaptive_manifest_path=config.adaptive_manifest_path,
        mixed_manifest_path=config.mixed_manifest_path,
        adaptive_validation=adaptive_validation,
        benign_validation=benign_validation,
        mixed_validation=mixed_validation,
    )


def validate_held_out_adaptive_attack_prompts(
    rows: Sequence[AdaptiveAttackPrompt] | Sequence[Mapping[str, Any]],
    records: Sequence[Tier1Record] | Sequence[Mapping[str, Any]],
    canary_registry: Sequence[CanaryRegistryEntry] | Sequence[Mapping[str, Any]],
    stage2_attack_rows: Sequence[Mapping[str, Any]],
    follow_on_adaptive_rows: Sequence[Mapping[str, Any]],
    stage3_filter_rows: Sequence[Mapping[str, Any]],
    protocol: ProtocolBundle,
    *,
    family_counts: Mapping[str, int],
    canary_assignment_family_offset: int = HELD_OUT_CANARY_ASSIGNMENT_FAMILY_OFFSET,
) -> HeldOutAdaptiveValidationSummary:
    normalized_rows = [_normalize_adaptive_row(row) for row in rows]
    normalized_records = [_normalize_record(record) for record in records]
    normalized_registry = sorted(
        (_normalize_registry_row(entry) for entry in canary_registry),
        key=lambda entry: entry.canary_id,
    )
    ordered_canary_ids = tuple(entry.canary_id for entry in normalized_registry)
    expected_specs = build_held_out_adaptive_attack_specs(
        ordered_canary_ids,
        family_counts,
        canary_assignment_family_offset=canary_assignment_family_offset,
    )
    if not HELD_OUT_ADAPTIVE_MIN_ROWS <= len(normalized_rows) <= HELD_OUT_ADAPTIVE_MAX_ROWS:
        raise HeldOutRobustnessValidationError(
            f"Held-out adaptive rows must be in [{HELD_OUT_ADAPTIVE_MIN_ROWS}, "
            f"{HELD_OUT_ADAPTIVE_MAX_ROWS}], found {len(normalized_rows)}."
        )
    if len(normalized_rows) != len(expected_specs):
        raise HeldOutRobustnessValidationError(
            f"Held-out adaptive rows must contain exactly {len(expected_specs)} rows, "
            f"found {len(normalized_rows)}."
        )

    source_sets = _build_source_text_sets(
        stage2_attack_rows=stage2_attack_rows,
        follow_on_adaptive_rows=follow_on_adaptive_rows,
        stage3_filter_rows=stage3_filter_rows,
    )
    canary_records_by_id = _index_canary_records(normalized_records)
    seen_attack_ids: set[str] = set()
    seen_message_texts: set[str] = set()
    seen_request_lines: set[str] = set()
    family_counter: Counter[str] = Counter()
    bundle_counter: Counter[tuple[str, tuple[str, ...]]] = Counter()
    target_canaries: set[str] = set()
    all_source_attack_ids = source_sets["stage2_attack_ids"] | source_sets["follow_on_attack_ids"]

    for row_index, (raw_row, row, spec) in enumerate(
        zip(rows, normalized_rows, expected_specs, strict=True),
        start=1,
    ):
        row_mapping = _adaptive_row_mapping(raw_row)
        if tuple(row_mapping.keys()) != ADAPTIVE_ATTACK_COLUMNS:
            raise HeldOutRobustnessValidationError(
                f"Held-out adaptive row {row_index} must use schema {ADAPTIVE_ATTACK_COLUMNS}, "
                f"found {tuple(row_mapping.keys())}."
            )
        request_line = extract_request_line(row.message_text, row.attack_id)
        if row.attack_id in seen_attack_ids:
            raise HeldOutRobustnessValidationError(f"Duplicate held-out attack_id {row.attack_id}.")
        if row.attack_id in all_source_attack_ids:
            raise HeldOutRobustnessValidationError(
                f"Held-out attack_id {row.attack_id} overlaps a source attack id."
            )
        if row.message_text in seen_message_texts:
            raise HeldOutRobustnessValidationError(
                f"Held-out adaptive row {row.attack_id} duplicates a message_text."
            )
        if request_line in seen_request_lines:
            raise HeldOutRobustnessValidationError(
                f"Held-out adaptive row {row.attack_id} duplicates a Request line."
            )
        _assert_disjoint_text(
            row_id=row.attack_id,
            message_text=row.message_text,
            request_line=request_line,
            source_sets=source_sets,
        )
        if row.attack_id != spec.attack_id:
            raise HeldOutRobustnessValidationError(
                f"Held-out row {row_index} has attack_id {row.attack_id}; expected {spec.attack_id}."
            )
        if row.template_family != spec.template_family:
            raise HeldOutRobustnessValidationError(
                f"Held-out row {row.attack_id} has family {row.template_family}; "
                f"expected {spec.template_family}."
            )
        if row.target_canary_id != spec.target_canary_id:
            raise HeldOutRobustnessValidationError(
                f"Held-out row {row.attack_id} targets {row.target_canary_id}; "
                f"expected {spec.target_canary_id}."
            )
        if row.split != FOLLOW_ON_SPLIT:
            raise HeldOutRobustnessValidationError(
                f"Held-out row {row.attack_id} must use split={FOLLOW_ON_SPLIT}."
            )
        canonical_fields = _canonicalize_or_raise(row.target_fields, row.attack_id)
        if canonical_fields != spec.target_fields:
            raise HeldOutRobustnessValidationError(
                f"Held-out row {row.attack_id} has target_fields {canonical_fields}; "
                f"expected {spec.target_fields}."
            )
        canary_record = canary_records_by_id.get(row.target_canary_id)
        if canary_record is None:
            raise HeldOutRobustnessValidationError(
                f"Held-out row {row.attack_id} targets unknown canary {row.target_canary_id}."
            )
        if canary_record.split != MEMBER_SPLIT:
            raise HeldOutRobustnessValidationError(
                f"Held-out row {row.attack_id} must target a member canary."
            )
        if protocol.stage1.system_prompt in row.message_text:
            raise HeldOutRobustnessValidationError(
                f"Held-out row {row.attack_id} must not embed the system prompt."
            )
        expected_message_text = protocol.render_intake_message(
            normalize_transaction_context(canary_record),
            spec.request_text,
        )
        if row.message_text != expected_message_text:
            raise HeldOutRobustnessValidationError(
                f"Held-out row {row.attack_id} does not match the frozen intake wrapper."
            )

        seen_attack_ids.add(row.attack_id)
        seen_message_texts.add(row.message_text)
        seen_request_lines.add(request_line)
        family_counter[row.template_family] += 1
        bundle_counter[(row.template_family, canonical_fields)] += 1
        target_canaries.add(row.target_canary_id)

    expected_family_counts = {
        family_name: int(family_counts[family_slug])
        for family_name, family_slug in ADAPTIVE_ATTACK_FAMILIES
    }
    if dict(family_counter) != expected_family_counts:
        raise HeldOutRobustnessValidationError(
            f"Held-out adaptive family counts must equal {expected_family_counts}, "
            f"found {dict(family_counter)}."
        )
    expected_bundle_counts = {
        (family_name, tuple(target_fields)): count
        for family_name in ADAPTIVE_ATTACK_FAMILY_ORDER
        for target_fields, count in HELD_OUT_TARGET_FIELD_BUNDLE_DISTRIBUTION
    }
    if dict(bundle_counter) != expected_bundle_counts:
        raise HeldOutRobustnessValidationError(
            "Held-out adaptive target-field distribution does not match the task contract."
        )

    target_field_distribution = {
        family_name: {
            tuple(target_fields): bundle_counter[(family_name, tuple(target_fields))]
            for target_fields, _count in HELD_OUT_TARGET_FIELD_BUNDLE_DISTRIBUTION
        }
        for family_name in ADAPTIVE_ATTACK_FAMILY_ORDER
    }
    disjointness_checks = _adaptive_disjointness_checks(
        normalized_rows,
        source_sets,
        all_source_attack_ids=all_source_attack_ids,
    )
    return HeldOutAdaptiveValidationSummary(
        row_count=len(normalized_rows),
        family_counts=dict(family_counter),
        target_field_distribution=target_field_distribution,
        unique_target_canaries=len(target_canaries),
        disjointness_checks=disjointness_checks,
    )


def validate_held_out_benign_hard_negatives(
    rows: Sequence[MixedTrafficRow] | Sequence[Mapping[str, Any]],
    records: Sequence[Tier1Record] | Sequence[Mapping[str, Any]],
    stage2_attack_rows: Sequence[Mapping[str, Any]],
    follow_on_adaptive_rows: Sequence[Mapping[str, Any]],
    stage3_filter_rows: Sequence[Mapping[str, Any]],
    protocol: ProtocolBundle,
    *,
    benign_family_counts: Mapping[str, int],
    benign_record_order: str = "record_id_ascending",
) -> HeldOutBenignValidationSummary:
    if benign_record_order != "record_id_ascending":
        raise HeldOutRobustnessValidationError(
            f"Unsupported benign_record_order {benign_record_order!r}; expected 'record_id_ascending'."
        )

    normalized_rows = [_normalize_mixed_row(row) for row in rows]
    normalized_records = [_normalize_record(record) for record in records]
    expected_specs = build_held_out_benign_hard_negative_specs(benign_family_counts)
    if len(normalized_rows) != len(expected_specs):
        raise HeldOutRobustnessValidationError(
            f"Held-out benign hard negatives must contain {len(expected_specs)} rows, "
            f"found {len(normalized_rows)}."
        )
    ordered_non_canary_records = _ordered_non_canary_records(normalized_records)
    if len(ordered_non_canary_records) < len(expected_specs):
        raise HeldOutRobustnessValidationError(
            f"Held-out benign hard negatives need {len(expected_specs)} non-canary records, "
            f"found {len(ordered_non_canary_records)}."
        )

    source_sets = _build_source_text_sets(
        stage2_attack_rows=stage2_attack_rows,
        follow_on_adaptive_rows=follow_on_adaptive_rows,
        stage3_filter_rows=stage3_filter_rows,
    )
    seen_traffic_ids: set[str] = set()
    seen_message_texts: set[str] = set()
    seen_request_lines: set[str] = set()
    benign_record_ids: set[str] = set()
    family_counter: Counter[str] = Counter()

    for row_index, (raw_row, row, spec) in enumerate(
        zip(rows, normalized_rows, expected_specs, strict=True),
        start=1,
    ):
        row_mapping = _mixed_row_mapping(raw_row)
        if tuple(row_mapping.keys()) != MIXED_TRAFFIC_COLUMNS:
            raise HeldOutRobustnessValidationError(
                f"Held-out benign row {row_index} must use schema {MIXED_TRAFFIC_COLUMNS}, "
                f"found {tuple(row_mapping.keys())}."
            )
        request_line = extract_request_line(row.message_text, row.traffic_id)
        if row.traffic_id in seen_traffic_ids:
            raise HeldOutRobustnessValidationError(f"Duplicate held-out traffic_id {row.traffic_id}.")
        if row.message_text in seen_message_texts:
            raise HeldOutRobustnessValidationError(
                f"Held-out benign row {row.traffic_id} duplicates a message_text."
            )
        if request_line in seen_request_lines:
            raise HeldOutRobustnessValidationError(
                f"Held-out benign row {row.traffic_id} duplicates a Request line."
            )
        _assert_disjoint_text(
            row_id=row.traffic_id,
            message_text=row.message_text,
            request_line=request_line,
            source_sets=source_sets,
        )
        _assert_benign_request_contract(request_line, row.traffic_id)

        record = ordered_non_canary_records[spec.row_index]
        expected_message_text = protocol.render_intake_message(
            normalize_transaction_context(record),
            spec.request_text,
        )
        if record.is_canary or record.canary_id is not None:
            raise HeldOutRobustnessValidationError(
                f"Held-out benign row {row.traffic_id} was assigned a canary record."
            )
        if row.traffic_id != spec.traffic_id:
            raise HeldOutRobustnessValidationError(
                f"Held-out benign row {row_index} has traffic_id {row.traffic_id}; "
                f"expected {spec.traffic_id}."
            )
        if row.source_type != MIXED_SOURCE_BENIGN:
            raise HeldOutRobustnessValidationError(
                f"Held-out benign row {row.traffic_id} must use source_type={MIXED_SOURCE_BENIGN}."
            )
        if row.expected_filter_label != FILTER_ALLOW:
            raise HeldOutRobustnessValidationError(
                f"Held-out benign row {row.traffic_id} must be labeled {FILTER_ALLOW}."
            )
        if row.template_family != spec.template_family:
            raise HeldOutRobustnessValidationError(
                f"Held-out benign row {row.traffic_id} has family {row.template_family}; "
                f"expected {spec.template_family}."
            )
        if row.target_canary_id is not None or row.target_fields:
            raise HeldOutRobustnessValidationError(
                f"Held-out benign row {row.traffic_id} must not target canary fields."
            )
        if row.split != FOLLOW_ON_SPLIT:
            raise HeldOutRobustnessValidationError(
                f"Held-out benign row {row.traffic_id} must use split={FOLLOW_ON_SPLIT}."
            )
        if protocol.stage1.system_prompt in row.message_text:
            raise HeldOutRobustnessValidationError(
                f"Held-out benign row {row.traffic_id} must not embed the system prompt."
            )
        if row.message_text != expected_message_text:
            raise HeldOutRobustnessValidationError(
                f"Held-out benign row {row.traffic_id} does not match the frozen wrapper."
            )

        seen_traffic_ids.add(row.traffic_id)
        seen_message_texts.add(row.message_text)
        seen_request_lines.add(request_line)
        benign_record_ids.add(record.record_id)
        family_counter[row.template_family] += 1

    expected_family_counts = {
        family_slug: int(benign_family_counts[family_slug])
        for family_slug in HELD_OUT_HARD_NEGATIVE_FAMILY_ORDER
    }
    if dict(family_counter) != expected_family_counts:
        raise HeldOutRobustnessValidationError(
            f"Held-out benign family counts must equal {expected_family_counts}, "
            f"found {dict(family_counter)}."
        )
    if len(benign_record_ids) != len(expected_specs):
        raise HeldOutRobustnessValidationError(
            f"Held-out benign rows must use {len(expected_specs)} unique non-canary records."
        )

    disjointness_checks = _mixed_disjointness_checks(normalized_rows, source_sets)
    return HeldOutBenignValidationSummary(
        row_count=len(normalized_rows),
        family_counts=dict(family_counter),
        unique_benign_records=len(benign_record_ids),
        disjointness_checks=disjointness_checks,
    )


def validate_held_out_mixed_traffic_rows(
    rows: Sequence[MixedTrafficRow] | Sequence[Mapping[str, Any]],
    records: Sequence[Tier1Record] | Sequence[Mapping[str, Any]],
    benign_rows: Sequence[MixedTrafficRow] | Sequence[Mapping[str, Any]],
    adaptive_attack_rows: Sequence[AdaptiveAttackPrompt] | Sequence[Mapping[str, Any]],
    stage2_attack_rows: Sequence[Mapping[str, Any]],
    follow_on_adaptive_rows: Sequence[Mapping[str, Any]],
    stage3_filter_rows: Sequence[Mapping[str, Any]],
    protocol: ProtocolBundle,
    *,
    benign_family_counts: Mapping[str, int],
    benign_record_order: str = "record_id_ascending",
) -> HeldOutMixedValidationSummary:
    normalized_rows = [_normalize_mixed_row(row) for row in rows]
    normalized_benign_rows = [_normalize_mixed_row(row) for row in benign_rows]
    adaptive_rows = [_normalize_adaptive_row(row) for row in adaptive_attack_rows]
    expected_rows = build_held_out_mixed_traffic_rows(normalized_benign_rows, adaptive_rows)
    if len(normalized_rows) != len(expected_rows):
        raise HeldOutRobustnessValidationError(
            f"Held-out mixed traffic must contain {len(expected_rows)} rows, "
            f"found {len(normalized_rows)}."
        )

    source_sets = _build_source_text_sets(
        stage2_attack_rows=stage2_attack_rows,
        follow_on_adaptive_rows=follow_on_adaptive_rows,
        stage3_filter_rows=stage3_filter_rows,
    )
    seen_traffic_ids: set[str] = set()
    seen_message_texts: set[str] = set()
    source_type_counter: Counter[str] = Counter()
    expected_filter_label_counter: Counter[str] = Counter()
    family_counter: Counter[str] = Counter()

    for row_index, (raw_row, row, expected_row) in enumerate(
        zip(rows, normalized_rows, expected_rows, strict=True),
        start=1,
    ):
        row_mapping = _mixed_row_mapping(raw_row)
        if tuple(row_mapping.keys()) != MIXED_TRAFFIC_COLUMNS:
            raise HeldOutRobustnessValidationError(
                f"Held-out mixed row {row_index} must use schema {MIXED_TRAFFIC_COLUMNS}, "
                f"found {tuple(row_mapping.keys())}."
            )
        request_line = extract_request_line(row.message_text, row.traffic_id)
        if row.traffic_id in seen_traffic_ids:
            raise HeldOutRobustnessValidationError(f"Duplicate held-out mixed traffic_id {row.traffic_id}.")
        if row.message_text in seen_message_texts:
            raise HeldOutRobustnessValidationError(
                f"Held-out mixed row {row.traffic_id} duplicates a message_text."
            )
        _assert_disjoint_text(
            row_id=row.traffic_id,
            message_text=row.message_text,
            request_line=request_line,
            source_sets=source_sets,
        )
        if row != expected_row:
            raise HeldOutRobustnessValidationError(
                f"Held-out mixed row {row_index} does not match the deterministic expected row."
            )
        if protocol.stage1.system_prompt in row.message_text:
            raise HeldOutRobustnessValidationError(
                f"Held-out mixed row {row.traffic_id} must not embed the system prompt."
            )

        seen_traffic_ids.add(row.traffic_id)
        seen_message_texts.add(row.message_text)
        source_type_counter[row.source_type] += 1
        expected_filter_label_counter[row.expected_filter_label] += 1
        family_counter[row.template_family] += 1

    benign_summary = validate_held_out_benign_hard_negatives(
        normalized_benign_rows,
        records,
        stage2_attack_rows,
        follow_on_adaptive_rows,
        stage3_filter_rows,
        protocol,
        benign_family_counts=benign_family_counts,
        benign_record_order=benign_record_order,
    )
    expected_source_counts = {
        MIXED_SOURCE_BENIGN: HELD_OUT_HARD_NEGATIVE_TOTAL_ROWS,
        MIXED_SOURCE_ADAPTIVE: HELD_OUT_ADAPTIVE_TOTAL_ROWS,
    }
    if dict(source_type_counter) != expected_source_counts:
        raise HeldOutRobustnessValidationError(
            f"Held-out mixed source counts must equal {expected_source_counts}, "
            f"found {dict(source_type_counter)}."
        )
    expected_label_counts = {
        FILTER_ALLOW: HELD_OUT_HARD_NEGATIVE_TOTAL_ROWS,
        FILTER_BLOCK: HELD_OUT_ADAPTIVE_TOTAL_ROWS,
    }
    if dict(expected_filter_label_counter) != expected_label_counts:
        raise HeldOutRobustnessValidationError(
            f"Held-out mixed label counts must equal {expected_label_counts}, "
            f"found {dict(expected_filter_label_counter)}."
        )
    disjointness_checks = _mixed_disjointness_checks(normalized_rows, source_sets)
    return HeldOutMixedValidationSummary(
        row_count=len(normalized_rows),
        source_type_counts=dict(source_type_counter),
        expected_filter_label_counts=dict(expected_filter_label_counter),
        family_counts=dict(family_counter),
        benign_record_count=benign_summary.unique_benign_records,
        adaptive_adversarial_count=source_type_counter[MIXED_SOURCE_ADAPTIVE],
        disjointness_checks=disjointness_checks,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser(description="Materialize held-out follow-on robustness data.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_HELD_OUT_ROBUSTNESS_CONFIG_PATH,
        help="Path to the held-out robustness data config TOML.",
    )
    args = parser.parse_args(argv)

    config = HeldOutRobustnessConfig.from_toml(args.config)
    result = materialize_held_out_robustness(config)
    print(
        "follow_on_held_out_robustness: "
        f"adaptive={result.adaptive_output_path} ({result.adaptive_validation.row_count} rows), "
        f"benign={result.benign_output_path} ({result.benign_validation.row_count} rows), "
        f"mixed={result.mixed_output_path} ({result.mixed_validation.row_count} rows), "
        f"disjointness_passed={result.adaptive_validation.disjointness_checks['passed']}"
    )
    return 0


def _adaptive_manifest(
    config: HeldOutRobustnessConfig,
    validation: HeldOutAdaptiveValidationSummary,
    *,
    benign_included: bool,
) -> dict[str, Any]:
    return {
        "artifact": "held_out_adaptive_attack_prompts",
        "output_path": relative_to_project(config.adaptive_output_path),
        "row_count": validation.row_count,
        "split": FOLLOW_ON_SPLIT,
        "families": held_out_adaptive_family_manifest(),
        "family_counts": validation.family_counts,
        "target_field_distribution_per_family": {
            family_name: [
                {"target_fields": list(target_fields), "count": count}
                for target_fields, count in distribution.items()
            ]
            for family_name, distribution in validation.target_field_distribution.items()
        },
        "target_field_distribution_contract": held_out_target_field_distribution_manifest(),
        "unique_target_canary_count": validation.unique_target_canaries,
        "canary_assignment_rule": (
            "ordered_canaries[(family_index * "
            f"{config.canary_assignment_family_offset} + prompt_index_within_family) % 100]"
        ),
        "row_order": (
            "seven held-out family slugs in follow-on order; within each family, target-field "
            "bundle order; within each bundle, template index ascending"
        ),
        "benign_hard_negatives_included": benign_included,
        "source_artifacts": {
            "tier1_records": relative_to_project(config.tier1_records_path),
            "canary_registry": relative_to_project(config.canary_registry_path),
            "stage2_attack_prompts": relative_to_project(config.stage2_attack_prompts_path),
            "follow_on_adaptive_attack_prompts": relative_to_project(
                config.follow_on_adaptive_attack_prompts_path
            ),
            "stage3_filter_messages": [
                relative_to_project(path)
                for path in config.stage3_filter_paths.paths
            ],
        },
        "disjointness_checks": validation.disjointness_checks,
    }


def _mixed_manifest(
    config: HeldOutRobustnessConfig,
    benign_validation: HeldOutBenignValidationSummary,
    mixed_validation: HeldOutMixedValidationSummary,
) -> dict[str, Any]:
    return {
        "artifact": "held_out_mixed_traffic_eval",
        "benign_output_path": relative_to_project(config.benign_output_path),
        "mixed_output_path": relative_to_project(config.mixed_output_path),
        "row_count": mixed_validation.row_count,
        "split": FOLLOW_ON_SPLIT,
        "source_type_counts": mixed_validation.source_type_counts,
        "expected_filter_label_counts": mixed_validation.expected_filter_label_counts,
        "family_counts": mixed_validation.family_counts,
        "benign_family_counts": benign_validation.family_counts,
        "benign_families": held_out_benign_family_manifest(config.benign_family_counts),
        "benign_hard_negative_count": benign_validation.row_count,
        "benign_unique_record_count": benign_validation.unique_benign_records,
        "adaptive_adversarial_count": mixed_validation.adaptive_adversarial_count,
        "benign_record_assignment_rule": (
            "ordered non-canary Tier 1 records by record_id ascending; "
            "benign_row_index selects the record"
        ),
        "adaptive_adversarial_assignment_rule": (
            "copy held_out_adaptive_attack_prompts.jsonl rows in order; "
            "traffic_id is mixed_{attack_id}"
        ),
        "source_artifacts": {
            "tier1_records": relative_to_project(config.tier1_records_path),
            "held_out_adaptive_attack_prompts": relative_to_project(config.adaptive_output_path),
            "stage2_attack_prompts": relative_to_project(config.stage2_attack_prompts_path),
            "follow_on_adaptive_attack_prompts": relative_to_project(
                config.follow_on_adaptive_attack_prompts_path
            ),
            "stage3_filter_messages": [
                relative_to_project(path)
                for path in config.stage3_filter_paths.paths
            ],
        },
        "disjointness_checks": {
            "benign": benign_validation.disjointness_checks,
            "mixed": mixed_validation.disjointness_checks,
        },
    }


def _adaptive_disjointness_checks(
    rows: Sequence[AdaptiveAttackPrompt],
    source_sets: Mapping[str, set[str]],
    *,
    all_source_attack_ids: set[str],
) -> dict[str, int | bool]:
    message_texts = [row.message_text for row in rows]
    request_lines = [extract_request_line(row.message_text, row.attack_id) for row in rows]
    attack_ids = [row.attack_id for row in rows]
    checks: dict[str, int | bool] = {
        "internal_duplicate_attack_ids": len(attack_ids) - len(set(attack_ids)),
        "internal_duplicate_message_texts": len(message_texts) - len(set(message_texts)),
        "internal_duplicate_request_lines": len(request_lines) - len(set(request_lines)),
        "source_attack_id_overlap": len(set(attack_ids) & all_source_attack_ids),
        "stage2_attack_message_text_overlap": len(set(message_texts) & source_sets["stage2_message_texts"]),
        "stage2_attack_request_line_overlap": len(set(request_lines) & source_sets["stage2_request_lines"]),
        "follow_on_adaptive_message_text_overlap": len(set(message_texts) & source_sets["follow_on_message_texts"]),
        "follow_on_adaptive_request_line_overlap": len(set(request_lines) & source_sets["follow_on_request_lines"]),
        "existing_adaptive_request_line_catalog_overlap": len(
            set(request_lines) & source_sets["follow_on_request_lines"]
        ),
        "stage3_filter_message_text_overlap": len(set(message_texts) & source_sets["stage3_message_texts"]),
        "stage3_filter_request_line_overlap": len(set(request_lines) & source_sets["stage3_request_lines"]),
    }
    checks["passed"] = all(value == 0 for value in checks.values())
    return checks


def _mixed_disjointness_checks(
    rows: Sequence[MixedTrafficRow],
    source_sets: Mapping[str, set[str]],
) -> dict[str, int | bool]:
    message_texts = [row.message_text for row in rows]
    request_lines = [extract_request_line(row.message_text, row.traffic_id) for row in rows]
    traffic_ids = [row.traffic_id for row in rows]
    checks: dict[str, int | bool] = {
        "internal_duplicate_traffic_ids": len(traffic_ids) - len(set(traffic_ids)),
        "internal_duplicate_message_texts": len(message_texts) - len(set(message_texts)),
        "internal_duplicate_request_lines": len(request_lines) - len(set(request_lines)),
        "stage2_attack_message_text_overlap": len(set(message_texts) & source_sets["stage2_message_texts"]),
        "stage2_attack_request_line_overlap": len(set(request_lines) & source_sets["stage2_request_lines"]),
        "follow_on_adaptive_message_text_overlap": len(set(message_texts) & source_sets["follow_on_message_texts"]),
        "follow_on_adaptive_request_line_overlap": len(set(request_lines) & source_sets["follow_on_request_lines"]),
        "stage3_filter_message_text_overlap": len(set(message_texts) & source_sets["stage3_message_texts"]),
        "stage3_filter_request_line_overlap": len(set(request_lines) & source_sets["stage3_request_lines"]),
    }
    checks["passed"] = all(value == 0 for value in checks.values())
    return checks


def _build_source_text_sets(
    *,
    stage2_attack_rows: Sequence[Mapping[str, Any]],
    follow_on_adaptive_rows: Sequence[Mapping[str, Any]],
    stage3_filter_rows: Sequence[Mapping[str, Any]],
) -> dict[str, set[str]]:
    stage2_message_texts, stage2_request_lines = _message_texts_and_request_lines(
        stage2_attack_rows,
        "attack_id",
    )
    follow_on_message_texts, follow_on_request_lines = _message_texts_and_request_lines(
        follow_on_adaptive_rows,
        "attack_id",
    )
    stage3_message_texts, stage3_request_lines = _message_texts_and_request_lines(
        stage3_filter_rows,
        "message_id",
    )
    return {
        "stage2_message_texts": stage2_message_texts,
        "stage2_request_lines": stage2_request_lines,
        "stage2_attack_ids": {str(row["attack_id"]) for row in stage2_attack_rows},
        "follow_on_message_texts": follow_on_message_texts,
        "follow_on_request_lines": follow_on_request_lines,
        "follow_on_attack_ids": {str(row["attack_id"]) for row in follow_on_adaptive_rows},
        "stage3_message_texts": stage3_message_texts,
        "stage3_request_lines": stage3_request_lines,
    }


def _message_texts_and_request_lines(
    rows: Sequence[Mapping[str, Any]],
    identifier_key: str,
) -> tuple[set[str], set[str]]:
    message_texts: set[str] = set()
    request_lines: set[str] = set()
    for row_index, row in enumerate(rows, start=1):
        message_text = str(row["message_text"])
        row_identifier = str(row.get(identifier_key, row_index))
        message_texts.add(message_text)
        request_lines.add(extract_request_line(message_text, row_identifier))
    return message_texts, request_lines


def _assert_disjoint_text(
    *,
    row_id: str,
    message_text: str,
    request_line: str,
    source_sets: Mapping[str, set[str]],
) -> None:
    if message_text in source_sets["stage2_message_texts"] or request_line in source_sets["stage2_request_lines"]:
        raise HeldOutRobustnessValidationError(f"Held-out row {row_id} duplicates Stage 2 attack text.")
    if (
        message_text in source_sets["follow_on_message_texts"]
        or request_line in source_sets["follow_on_request_lines"]
    ):
        raise HeldOutRobustnessValidationError(
            f"Held-out row {row_id} duplicates current follow-on adaptive text."
        )
    if message_text in source_sets["stage3_message_texts"] or request_line in source_sets["stage3_request_lines"]:
        raise HeldOutRobustnessValidationError(f"Held-out row {row_id} duplicates Stage 3 filter text.")


def _assert_benign_request_contract(request_line: str, row_id: str) -> None:
    normalized = request_line.lower()
    forbidden_fragments = (
        "raw pii",
        "raw identity",
        "verbatim",
        "hidden prompt",
        "verify the customer name",
        "verify the date of birth",
        "verify the account number",
        "complete the customer name",
        "complete the date of birth",
        "complete the account number",
    )
    for fragment in forbidden_fragments:
        if fragment in normalized:
            raise HeldOutRobustnessValidationError(
                f"Held-out benign row {row_id} asks for disallowed identity disclosure: {fragment}."
            )


def _canonicalize_or_raise(target_fields: Sequence[str], row_id: str) -> tuple[str, ...]:
    try:
        return canonicalize_target_fields(target_fields)
    except ValueError as exc:
        raise HeldOutRobustnessValidationError(
            f"Held-out row {row_id} has invalid target_fields: {exc}"
        ) from exc


def _normalize_adaptive_row(row: AdaptiveAttackPrompt | Mapping[str, Any]) -> AdaptiveAttackPrompt:
    if isinstance(row, AdaptiveAttackPrompt):
        return row
    if not isinstance(row, Mapping):
        raise HeldOutRobustnessValidationError("Adaptive rows must be mappings.")
    for key in ADAPTIVE_ATTACK_COLUMNS:
        if key not in row:
            raise HeldOutRobustnessValidationError(f"Adaptive row is missing key {key}.")
    try:
        return AdaptiveAttackPrompt.from_row(row)
    except (KeyError, TypeError, ValueError) as exc:
        raise HeldOutRobustnessValidationError(str(exc)) from exc


def _normalize_mixed_row(row: MixedTrafficRow | Mapping[str, Any]) -> MixedTrafficRow:
    if isinstance(row, MixedTrafficRow):
        return row
    if not isinstance(row, Mapping):
        raise HeldOutRobustnessValidationError("Mixed rows must be mappings.")
    for key in MIXED_TRAFFIC_COLUMNS:
        if key not in row:
            raise HeldOutRobustnessValidationError(f"Mixed row is missing key {key}.")
    try:
        return MixedTrafficRow.from_row(row)
    except (KeyError, TypeError, ValueError) as exc:
        raise HeldOutRobustnessValidationError(str(exc)) from exc


def _normalize_record(record: Tier1Record | Mapping[str, Any]) -> Tier1Record:
    return record if isinstance(record, Tier1Record) else Tier1Record.from_row(record)


def _normalize_registry_row(
    row: CanaryRegistryEntry | Mapping[str, Any],
) -> CanaryRegistryEntry:
    return row if isinstance(row, CanaryRegistryEntry) else CanaryRegistryEntry.from_row(row)


def _adaptive_row_mapping(row: AdaptiveAttackPrompt | Mapping[str, Any]) -> Mapping[str, Any]:
    return row.to_row() if isinstance(row, AdaptiveAttackPrompt) else row


def _mixed_row_mapping(row: MixedTrafficRow | Mapping[str, Any]) -> Mapping[str, Any]:
    return row.to_row() if isinstance(row, MixedTrafficRow) else row


def _index_canary_records(records: Sequence[Tier1Record]) -> dict[str, Tier1Record]:
    return {
        record.canary_id: record
        for record in records
        if record.is_canary and record.canary_id is not None
    }


def _ordered_non_canary_records(records: Sequence[Tier1Record]) -> tuple[Tier1Record, ...]:
    record_ids = [record.record_id for record in records]
    if len(set(record_ids)) != len(record_ids):
        raise HeldOutRobustnessValidationError("Tier 1 record_id values must be unique.")
    return tuple(
        sorted(
            (record for record in records if not record.is_canary and record.canary_id is None),
            key=lambda record: record.record_id,
        )
    )


def _resolve_path(path_value: Path | str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
