from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import tomllib
from typing import Any

from experiment.chat_render import normalize_transaction_context
from experiment.schemas.stage2 import (
    TARGET_FIELD_BUNDLE_DISTRIBUTION,
    canonicalize_target_fields,
)
from experiment.schemas.tier1 import CanaryRegistryEntry, MEMBER_SPLIT, Tier1Record
from qwen_langgraph_demo.runtime.protocol import ProtocolBundle

from .adaptive_catalog import (
    ADAPTIVE_ATTACK_FAMILIES,
    ADAPTIVE_ATTACK_FAMILY_CATALOGS,
    ADAPTIVE_ATTACK_FAMILY_ORDER,
    ADAPTIVE_ATTACK_FAMILY_SLUG_ORDER,
    ADAPTIVE_ATTACK_TOTAL_ROWS,
    ADAPTIVE_CANARY_ASSIGNMENT_FAMILY_OFFSET,
    MIXED_BENIGN_FAMILY_ORDER,
    MIXED_BENIGN_TOTAL_ROWS,
    build_adaptive_attack_specs,
    build_mixed_benign_traffic_specs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FOLLOW_ON_ADAPTIVE_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "follow_on" / "adaptive_attacks.toml"
)
DEFAULT_FOLLOW_ON_MIXED_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "follow_on" / "mixed_traffic.toml"
)

FOLLOW_ON_SPLIT = "eval"

ADAPTIVE_ATTACK_COLUMNS = (
    "attack_id",
    "template_family",
    "target_canary_id",
    "message_text",
    "target_fields",
    "split",
)

MIXED_TRAFFIC_COLUMNS = (
    "traffic_id",
    "source_type",
    "template_family",
    "message_text",
    "expected_filter_label",
    "target_canary_id",
    "target_fields",
    "split",
)

MIXED_SOURCE_BENIGN = "benign"
MIXED_SOURCE_ADAPTIVE = "adaptive_adversarial"
FILTER_ALLOW = "ALLOW"
FILTER_BLOCK = "BLOCK"


class FollowOnDataValidationError(ValueError):
    """Raised when a follow-on dataset violates the frozen task contract."""


class AdaptiveAttackValidationError(FollowOnDataValidationError):
    """Raised when adaptive attack rows violate the follow-on contract."""


class MixedTrafficValidationError(FollowOnDataValidationError):
    """Raised when mixed-traffic rows violate the follow-on contract."""


@dataclass(frozen=True, slots=True)
class Stage3FilterPaths:
    train_path: Path
    val_path: Path
    test_path: Path

    @property
    def paths(self) -> tuple[Path, Path, Path]:
        return (self.train_path, self.val_path, self.test_path)


@dataclass(frozen=True, slots=True)
class AdaptiveAttackConfig:
    protocol_config_dir: Path
    tier1_records_path: Path
    canary_registry_path: Path
    stage2_attack_prompts_path: Path
    stage3_filter_paths: Stage3FilterPaths
    output_dir: Path
    output_filename: str
    manifest_filename: str
    family_counts: dict[str, int]
    canary_assignment_family_offset: int

    @property
    def output_path(self) -> Path:
        return self.output_dir / self.output_filename

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / self.manifest_filename

    @classmethod
    def from_toml(cls, config_path: Path | str | None = None) -> "AdaptiveAttackConfig":
        path = _resolve_path(config_path or DEFAULT_FOLLOW_ON_ADAPTIVE_CONFIG_PATH)
        with path.open("rb") as handle:
            document = tomllib.load(handle)

        return cls(
            protocol_config_dir=_resolve_path(document["protocol"]["config_dir"]),
            tier1_records_path=_resolve_path(document["inputs"]["tier1_records_path"]),
            canary_registry_path=_resolve_path(document["inputs"]["canary_registry_path"]),
            stage2_attack_prompts_path=_resolve_path(document["inputs"]["stage2_attack_prompts_path"]),
            stage3_filter_paths=Stage3FilterPaths(
                train_path=_resolve_path(document["inputs"]["stage3_filter_messages_train_path"]),
                val_path=_resolve_path(document["inputs"]["stage3_filter_messages_val_path"]),
                test_path=_resolve_path(document["inputs"]["stage3_filter_messages_test_path"]),
            ),
            output_dir=_resolve_path(document["outputs"]["output_dir"]),
            output_filename=str(document["outputs"]["output_filename"]),
            manifest_filename=str(document["outputs"]["manifest_filename"]),
            family_counts={key: int(value) for key, value in document["family_counts"].items()},
            canary_assignment_family_offset=int(
                document.get("assignment", {}).get(
                    "canary_family_offset",
                    ADAPTIVE_CANARY_ASSIGNMENT_FAMILY_OFFSET,
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class MixedTrafficConfig:
    protocol_config_dir: Path
    tier1_records_path: Path
    adaptive_attack_prompts_path: Path
    stage3_filter_paths: Stage3FilterPaths
    output_dir: Path
    output_filename: str
    manifest_filename: str
    family_counts: dict[str, int]
    benign_record_order: str

    @property
    def output_path(self) -> Path:
        return self.output_dir / self.output_filename

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / self.manifest_filename

    @classmethod
    def from_toml(cls, config_path: Path | str | None = None) -> "MixedTrafficConfig":
        path = _resolve_path(config_path or DEFAULT_FOLLOW_ON_MIXED_CONFIG_PATH)
        with path.open("rb") as handle:
            document = tomllib.load(handle)

        return cls(
            protocol_config_dir=_resolve_path(document["protocol"]["config_dir"]),
            tier1_records_path=_resolve_path(document["inputs"]["tier1_records_path"]),
            adaptive_attack_prompts_path=_resolve_path(document["inputs"]["adaptive_attack_prompts_path"]),
            stage3_filter_paths=Stage3FilterPaths(
                train_path=_resolve_path(document["inputs"]["stage3_filter_messages_train_path"]),
                val_path=_resolve_path(document["inputs"]["stage3_filter_messages_val_path"]),
                test_path=_resolve_path(document["inputs"]["stage3_filter_messages_test_path"]),
            ),
            output_dir=_resolve_path(document["outputs"]["output_dir"]),
            output_filename=str(document["outputs"]["output_filename"]),
            manifest_filename=str(document["outputs"]["manifest_filename"]),
            family_counts={key: int(value) for key, value in document["family_counts"].items()},
            benign_record_order=str(
                document.get("record_assignment", {}).get(
                    "benign_record_order",
                    "record_id_ascending",
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class AdaptiveAttackPrompt:
    attack_id: str
    template_family: str
    target_canary_id: str
    message_text: str
    target_fields: tuple[str, ...]
    split: str

    def to_row(self) -> dict[str, Any]:
        return {
            "attack_id": self.attack_id,
            "template_family": self.template_family,
            "target_canary_id": self.target_canary_id,
            "message_text": self.message_text,
            "target_fields": list(self.target_fields),
            "split": self.split,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "AdaptiveAttackPrompt":
        raw_target_fields = row["target_fields"]
        if not isinstance(raw_target_fields, Sequence) or isinstance(raw_target_fields, (str, bytes)):
            raise TypeError("Adaptive target_fields must be a JSON array of strings.")
        return cls(
            attack_id=str(row["attack_id"]),
            template_family=str(row["template_family"]),
            target_canary_id=str(row["target_canary_id"]),
            message_text=str(row["message_text"]),
            target_fields=tuple(str(field) for field in raw_target_fields),
            split=str(row["split"]),
        )


@dataclass(frozen=True, slots=True)
class MixedTrafficRow:
    traffic_id: str
    source_type: str
    template_family: str
    message_text: str
    expected_filter_label: str
    target_canary_id: str | None
    target_fields: tuple[str, ...]
    split: str

    def to_row(self) -> dict[str, Any]:
        return {
            "traffic_id": self.traffic_id,
            "source_type": self.source_type,
            "template_family": self.template_family,
            "message_text": self.message_text,
            "expected_filter_label": self.expected_filter_label,
            "target_canary_id": self.target_canary_id,
            "target_fields": list(self.target_fields),
            "split": self.split,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "MixedTrafficRow":
        raw_target_fields = row["target_fields"]
        if not isinstance(raw_target_fields, Sequence) or isinstance(raw_target_fields, (str, bytes)):
            raise TypeError("Mixed traffic target_fields must be a JSON array of strings.")
        raw_target_canary_id = row["target_canary_id"]
        return cls(
            traffic_id=str(row["traffic_id"]),
            source_type=str(row["source_type"]),
            template_family=str(row["template_family"]),
            message_text=str(row["message_text"]),
            expected_filter_label=str(row["expected_filter_label"]),
            target_canary_id=(
                None
                if raw_target_canary_id in (None, "")
                else str(raw_target_canary_id)
            ),
            target_fields=tuple(str(field) for field in raw_target_fields),
            split=str(row["split"]),
        )


@dataclass(frozen=True, slots=True)
class AdaptiveAttackValidationSummary:
    row_count: int
    family_counts: dict[str, int]
    target_field_distribution: dict[str, dict[tuple[str, ...], int]]
    unique_target_canaries: int


@dataclass(frozen=True, slots=True)
class MixedTrafficValidationSummary:
    row_count: int
    source_type_counts: dict[str, int]
    expected_filter_label_counts: dict[str, int]
    family_counts: dict[str, int]
    benign_record_count: int
    adaptive_adversarial_count: int


def load_stage3_filter_rows(paths: Stage3FilterPaths) -> list[dict[str, Any]]:
    from experiment.data_gen.io import read_jsonl_rows

    rows: list[dict[str, Any]] = []
    for path in paths.paths:
        rows.extend(read_jsonl_rows(path))
    return rows


def write_manifest(document: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def validate_adaptive_attack_prompts(
    rows: Sequence[AdaptiveAttackPrompt] | Sequence[Mapping[str, Any]],
    records: Sequence[Tier1Record] | Sequence[Mapping[str, Any]],
    canary_registry: Sequence[CanaryRegistryEntry] | Sequence[Mapping[str, Any]],
    stage2_attack_rows: Sequence[Mapping[str, Any]],
    stage3_filter_rows: Sequence[Mapping[str, Any]],
    protocol: ProtocolBundle,
    *,
    family_counts: Mapping[str, int],
    canary_assignment_family_offset: int = ADAPTIVE_CANARY_ASSIGNMENT_FAMILY_OFFSET,
) -> AdaptiveAttackValidationSummary:
    normalized_rows = [_normalize_adaptive_row(row) for row in rows]
    normalized_records = [_normalize_record(record) for record in records]
    normalized_registry = sorted(
        (_normalize_registry_row(entry) for entry in canary_registry),
        key=lambda entry: entry.canary_id,
    )
    ordered_canary_ids = tuple(entry.canary_id for entry in normalized_registry)
    expected_specs = build_adaptive_attack_specs(
        ordered_canary_ids,
        family_counts,
        canary_assignment_family_offset=canary_assignment_family_offset,
    )

    if len(normalized_rows) != ADAPTIVE_ATTACK_TOTAL_ROWS:
        raise AdaptiveAttackValidationError(
            f"Adaptive attack prompts must contain exactly {ADAPTIVE_ATTACK_TOTAL_ROWS} rows, "
            f"found {len(normalized_rows)}."
        )

    canary_records_by_id = _index_canary_records(normalized_records)
    stage2_message_texts, stage2_request_lines = _message_texts_and_request_lines(
        stage2_attack_rows,
        "attack_id",
    )
    stage3_message_texts, stage3_request_lines = _message_texts_and_request_lines(
        stage3_filter_rows,
        "message_id",
    )

    seen_attack_ids: set[str] = set()
    seen_message_texts: set[str] = set()
    family_counter: Counter[str] = Counter()
    bundle_counter: Counter[tuple[str, tuple[str, ...]]] = Counter()
    target_canaries: set[str] = set()
    benign_request_line = f"Request: {protocol.stage1.benign_request}"

    for row_index, (raw_row, row, spec) in enumerate(
        zip(rows, normalized_rows, expected_specs, strict=True),
        start=1,
    ):
        row_mapping = _adaptive_row_mapping(raw_row)
        if tuple(row_mapping.keys()) != ADAPTIVE_ATTACK_COLUMNS:
            raise AdaptiveAttackValidationError(
                f"Adaptive row {row_index} must use the exact schema and key order "
                f"{ADAPTIVE_ATTACK_COLUMNS}, found {tuple(row_mapping.keys())}."
            )
        if row.attack_id in seen_attack_ids:
            raise AdaptiveAttackValidationError(
                f"Adaptive row {row.attack_id} duplicates an attack_id."
            )
        if row.message_text in seen_message_texts:
            raise AdaptiveAttackValidationError(
                f"Adaptive row {row.attack_id} duplicates a message_text."
            )
        request_line = extract_request_line(row.message_text, row.attack_id)
        if row.message_text in stage2_message_texts or request_line in stage2_request_lines:
            raise AdaptiveAttackValidationError(
                f"Adaptive row {row.attack_id} duplicates Stage 2 attack text."
            )
        if row.message_text in stage3_message_texts or request_line in stage3_request_lines:
            raise AdaptiveAttackValidationError(
                f"Adaptive row {row.attack_id} duplicates Stage 3 filter text."
            )
        if row.attack_id != spec.attack_id:
            raise AdaptiveAttackValidationError(
                f"Adaptive row {row_index} has attack_id {row.attack_id}; expected {spec.attack_id}."
            )
        if row.template_family != spec.template_family:
            raise AdaptiveAttackValidationError(
                f"Adaptive row {row.attack_id} has template_family {row.template_family}; "
                f"expected {spec.template_family}."
            )
        if row.target_canary_id != spec.target_canary_id:
            raise AdaptiveAttackValidationError(
                f"Adaptive row {row.attack_id} targets {row.target_canary_id}; "
                f"expected {spec.target_canary_id}."
            )
        if row.split != FOLLOW_ON_SPLIT:
            raise AdaptiveAttackValidationError(
                f"Adaptive row {row.attack_id} must have split={FOLLOW_ON_SPLIT}."
            )
        try:
            canonical_fields = canonicalize_target_fields(row.target_fields)
        except ValueError as exc:
            raise AdaptiveAttackValidationError(str(exc)) from exc
        if canonical_fields != spec.target_fields:
            raise AdaptiveAttackValidationError(
                f"Adaptive row {row.attack_id} has target_fields {canonical_fields}; "
                f"expected {spec.target_fields}."
            )
        canary_record = canary_records_by_id.get(row.target_canary_id)
        if canary_record is None:
            raise AdaptiveAttackValidationError(
                f"Adaptive row {row.attack_id} targets unknown canary {row.target_canary_id}."
            )
        if canary_record.split != MEMBER_SPLIT:
            raise AdaptiveAttackValidationError(
                f"Adaptive row {row.attack_id} must target a member canary record."
            )
        if benign_request_line in row.message_text:
            raise AdaptiveAttackValidationError(
                f"Adaptive row {row.attack_id} must not include the benign Stage 1 request line."
            )
        if protocol.stage1.system_prompt in row.message_text:
            raise AdaptiveAttackValidationError(
                f"Adaptive row {row.attack_id} must not embed the system prompt in message_text."
            )

        expected_message_text = protocol.render_intake_message(
            normalize_transaction_context(canary_record),
            spec.request_text,
        )
        if row.message_text != expected_message_text:
            raise AdaptiveAttackValidationError(
                f"Adaptive row {row.attack_id} does not match the frozen intake wrapper."
            )

        seen_attack_ids.add(row.attack_id)
        seen_message_texts.add(row.message_text)
        family_counter[row.template_family] += 1
        bundle_counter[(row.template_family, canonical_fields)] += 1
        target_canaries.add(row.target_canary_id)

    expected_family_counts = {
        family_name: int(family_counts[family_slug])
        for family_name, family_slug in ADAPTIVE_ATTACK_FAMILIES
    }
    if dict(family_counter) != expected_family_counts:
        raise AdaptiveAttackValidationError(
            f"Adaptive family counts must equal {expected_family_counts}, "
            f"found {dict(family_counter)}."
        )

    expected_bundle_counts = {
        (family_name, tuple(target_fields)): count
        for family_name in ADAPTIVE_ATTACK_FAMILY_ORDER
        for target_fields, count in TARGET_FIELD_BUNDLE_DISTRIBUTION
    }
    if dict(bundle_counter) != expected_bundle_counts:
        raise AdaptiveAttackValidationError(
            "Adaptive per-family target_fields distribution does not match the follow-on contract."
        )

    target_field_distribution: dict[str, dict[tuple[str, ...], int]] = {
        family_name: {
            tuple(target_fields): bundle_counter[(family_name, tuple(target_fields))]
            for target_fields, _count in TARGET_FIELD_BUNDLE_DISTRIBUTION
        }
        for family_name in ADAPTIVE_ATTACK_FAMILY_ORDER
    }

    return AdaptiveAttackValidationSummary(
        row_count=len(normalized_rows),
        family_counts=dict(family_counter),
        target_field_distribution=target_field_distribution,
        unique_target_canaries=len(target_canaries),
    )


def validate_mixed_traffic_rows(
    rows: Sequence[MixedTrafficRow] | Sequence[Mapping[str, Any]],
    records: Sequence[Tier1Record] | Sequence[Mapping[str, Any]],
    adaptive_attack_rows: Sequence[AdaptiveAttackPrompt] | Sequence[Mapping[str, Any]],
    stage3_filter_rows: Sequence[Mapping[str, Any]],
    protocol: ProtocolBundle,
    *,
    benign_family_counts: Mapping[str, int],
    benign_record_order: str = "record_id_ascending",
) -> MixedTrafficValidationSummary:
    if benign_record_order != "record_id_ascending":
        raise MixedTrafficValidationError(
            f"Unsupported benign_record_order {benign_record_order!r}; expected 'record_id_ascending'."
        )

    normalized_rows = [_normalize_mixed_row(row) for row in rows]
    normalized_records = [_normalize_record(record) for record in records]
    adaptive_rows = [_normalize_adaptive_row(row) for row in adaptive_attack_rows]
    benign_specs = build_mixed_benign_traffic_specs(benign_family_counts)
    expected_row_count = len(benign_specs) + len(adaptive_rows)
    if len(normalized_rows) != expected_row_count:
        raise MixedTrafficValidationError(
            f"Mixed traffic must contain exactly {expected_row_count} rows, "
            f"found {len(normalized_rows)}."
        )

    ordered_non_canary_records = _ordered_non_canary_records(normalized_records)
    if len(ordered_non_canary_records) < len(benign_specs):
        raise MixedTrafficValidationError(
            f"Mixed traffic requires at least {len(benign_specs)} non-canary records, "
            f"found {len(ordered_non_canary_records)}."
        )

    stage3_message_texts, stage3_request_lines = _message_texts_and_request_lines(
        stage3_filter_rows,
        "message_id",
    )

    seen_traffic_ids: set[str] = set()
    seen_message_texts: set[str] = set()
    benign_record_ids: set[str] = set()
    source_type_counter: Counter[str] = Counter()
    expected_filter_label_counter: Counter[str] = Counter()
    family_counter: Counter[str] = Counter()

    for row_index, (raw_row, row) in enumerate(zip(rows, normalized_rows, strict=True), start=1):
        row_mapping = _mixed_row_mapping(raw_row)
        if tuple(row_mapping.keys()) != MIXED_TRAFFIC_COLUMNS:
            raise MixedTrafficValidationError(
                f"Mixed row {row_index} must use the exact schema and key order "
                f"{MIXED_TRAFFIC_COLUMNS}, found {tuple(row_mapping.keys())}."
            )
        if row.traffic_id in seen_traffic_ids:
            raise MixedTrafficValidationError(
                f"Mixed row {row.traffic_id} duplicates a traffic_id."
            )
        if row.message_text in seen_message_texts:
            raise MixedTrafficValidationError(
                f"Mixed row {row.traffic_id} duplicates a message_text."
            )
        request_line = extract_request_line(row.message_text, row.traffic_id)
        if row.message_text in stage3_message_texts or request_line in stage3_request_lines:
            raise MixedTrafficValidationError(
                f"Mixed row {row.traffic_id} duplicates Stage 3 filter text."
            )
        if row.split != FOLLOW_ON_SPLIT:
            raise MixedTrafficValidationError(
                f"Mixed row {row.traffic_id} must have split={FOLLOW_ON_SPLIT}."
            )
        if protocol.stage1.system_prompt in row.message_text:
            raise MixedTrafficValidationError(
                f"Mixed row {row.traffic_id} must not embed the system prompt in message_text."
            )

        if row_index <= len(benign_specs):
            spec = benign_specs[row_index - 1]
            record = ordered_non_canary_records[spec.row_index]
            if record.is_canary or record.canary_id is not None:
                raise MixedTrafficValidationError(
                    f"Mixed benign row {row.traffic_id} was assigned a canary record."
                )
            expected_message_text = protocol.render_intake_message(
                normalize_transaction_context(record),
                spec.request_text,
            )
            if row.traffic_id != spec.traffic_id:
                raise MixedTrafficValidationError(
                    f"Mixed row {row_index} has traffic_id {row.traffic_id}; expected {spec.traffic_id}."
                )
            if row.source_type != MIXED_SOURCE_BENIGN:
                raise MixedTrafficValidationError(
                    f"Mixed row {row.traffic_id} must have source_type={MIXED_SOURCE_BENIGN}."
                )
            if row.expected_filter_label != FILTER_ALLOW:
                raise MixedTrafficValidationError(
                    f"Mixed benign row {row.traffic_id} must have expected_filter_label={FILTER_ALLOW}."
                )
            if row.template_family != spec.template_family:
                raise MixedTrafficValidationError(
                    f"Mixed row {row.traffic_id} has template_family {row.template_family}; "
                    f"expected {spec.template_family}."
                )
            if row.target_canary_id is not None:
                raise MixedTrafficValidationError(
                    f"Mixed benign row {row.traffic_id} must not target a canary."
                )
            if row.target_fields:
                raise MixedTrafficValidationError(
                    f"Mixed benign row {row.traffic_id} must use an empty target_fields list."
                )
            if row.message_text != expected_message_text:
                raise MixedTrafficValidationError(
                    f"Mixed benign row {row.traffic_id} does not match the frozen intake wrapper."
                )
            benign_record_ids.add(record.record_id)
        else:
            adaptive_index = row_index - len(benign_specs) - 1
            adaptive_row = adaptive_rows[adaptive_index]
            expected_traffic_id = f"mixed_{adaptive_row.attack_id}"
            if row.traffic_id != expected_traffic_id:
                raise MixedTrafficValidationError(
                    f"Mixed row {row_index} has traffic_id {row.traffic_id}; "
                    f"expected {expected_traffic_id}."
                )
            if row.source_type != MIXED_SOURCE_ADAPTIVE:
                raise MixedTrafficValidationError(
                    f"Mixed adaptive row {row.traffic_id} must have source_type={MIXED_SOURCE_ADAPTIVE}."
                )
            if row.expected_filter_label != FILTER_BLOCK:
                raise MixedTrafficValidationError(
                    f"Mixed adaptive row {row.traffic_id} must have expected_filter_label={FILTER_BLOCK}."
                )
            if row.template_family != adaptive_row.template_family:
                raise MixedTrafficValidationError(
                    f"Mixed adaptive row {row.traffic_id} has template_family {row.template_family}; "
                    f"expected {adaptive_row.template_family}."
                )
            if row.target_canary_id != adaptive_row.target_canary_id:
                raise MixedTrafficValidationError(
                    f"Mixed adaptive row {row.traffic_id} targets {row.target_canary_id}; "
                    f"expected {adaptive_row.target_canary_id}."
                )
            try:
                canonical_fields = canonicalize_target_fields(row.target_fields)
            except ValueError as exc:
                raise MixedTrafficValidationError(str(exc)) from exc
            if canonical_fields != adaptive_row.target_fields:
                raise MixedTrafficValidationError(
                    f"Mixed adaptive row {row.traffic_id} has target_fields {canonical_fields}; "
                    f"expected {adaptive_row.target_fields}."
                )
            if row.message_text != adaptive_row.message_text:
                raise MixedTrafficValidationError(
                    f"Mixed adaptive row {row.traffic_id} must copy the adaptive attack message_text."
                )

        seen_traffic_ids.add(row.traffic_id)
        seen_message_texts.add(row.message_text)
        source_type_counter[row.source_type] += 1
        expected_filter_label_counter[row.expected_filter_label] += 1
        family_counter[row.template_family] += 1

    expected_source_type_counts = {
        MIXED_SOURCE_BENIGN: MIXED_BENIGN_TOTAL_ROWS,
        MIXED_SOURCE_ADAPTIVE: ADAPTIVE_ATTACK_TOTAL_ROWS,
    }
    if dict(source_type_counter) != expected_source_type_counts:
        raise MixedTrafficValidationError(
            f"Mixed source_type counts must equal {expected_source_type_counts}, "
            f"found {dict(source_type_counter)}."
        )
    expected_label_counts = {
        FILTER_ALLOW: MIXED_BENIGN_TOTAL_ROWS,
        FILTER_BLOCK: ADAPTIVE_ATTACK_TOTAL_ROWS,
    }
    if dict(expected_filter_label_counter) != expected_label_counts:
        raise MixedTrafficValidationError(
            f"Mixed expected_filter_label counts must equal {expected_label_counts}, "
            f"found {dict(expected_filter_label_counter)}."
        )
    if len(benign_record_ids) != MIXED_BENIGN_TOTAL_ROWS:
        raise MixedTrafficValidationError(
            f"Mixed benign rows must use {MIXED_BENIGN_TOTAL_ROWS} unique non-canary records."
        )

    return MixedTrafficValidationSummary(
        row_count=len(normalized_rows),
        source_type_counts=dict(source_type_counter),
        expected_filter_label_counts=dict(expected_filter_label_counter),
        family_counts=dict(family_counter),
        benign_record_count=len(benign_record_ids),
        adaptive_adversarial_count=source_type_counter[MIXED_SOURCE_ADAPTIVE],
    )


def extract_request_line(message_text: str, row_identifier: str) -> str:
    lines = message_text.splitlines()
    if not lines or not lines[-1].startswith("Request: "):
        raise FollowOnDataValidationError(
            f"Follow-on row {row_identifier} must end with a Request: line."
        )
    return lines[-1]


def target_field_distribution_manifest() -> list[dict[str, Any]]:
    return [
        {"target_fields": list(target_fields), "count": count}
        for target_fields, count in TARGET_FIELD_BUNDLE_DISTRIBUTION
    ]


def adaptive_family_manifest() -> list[dict[str, Any]]:
    return [
        {"family": family_name, "slug": family_slug, "count": 50}
        for family_name, family_slug in ADAPTIVE_ATTACK_FAMILIES
    ]


def mixed_benign_family_manifest(family_counts: Mapping[str, int]) -> list[dict[str, Any]]:
    return [
        {"family": family_slug, "count": int(family_counts[family_slug])}
        for family_slug in MIXED_BENIGN_FAMILY_ORDER
    ]


def relative_to_project(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _resolve_path(path_value: Path | str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _normalize_adaptive_row(row: AdaptiveAttackPrompt | Mapping[str, Any]) -> AdaptiveAttackPrompt:
    if isinstance(row, AdaptiveAttackPrompt):
        return row
    if not isinstance(row, Mapping):
        raise AdaptiveAttackValidationError(
            "Adaptive rows must be AdaptiveAttackPrompt instances or mappings."
        )
    for key in ADAPTIVE_ATTACK_COLUMNS:
        if key not in row:
            raise AdaptiveAttackValidationError(f"Adaptive row is missing required key {key}.")
    try:
        return AdaptiveAttackPrompt.from_row(row)
    except (KeyError, TypeError, ValueError) as exc:
        raise AdaptiveAttackValidationError(str(exc)) from exc


def _normalize_mixed_row(row: MixedTrafficRow | Mapping[str, Any]) -> MixedTrafficRow:
    if isinstance(row, MixedTrafficRow):
        return row
    if not isinstance(row, Mapping):
        raise MixedTrafficValidationError(
            "Mixed rows must be MixedTrafficRow instances or mappings."
        )
    for key in MIXED_TRAFFIC_COLUMNS:
        if key not in row:
            raise MixedTrafficValidationError(f"Mixed row is missing required key {key}.")
    try:
        return MixedTrafficRow.from_row(row)
    except (KeyError, TypeError, ValueError) as exc:
        raise MixedTrafficValidationError(str(exc)) from exc


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
    indexed: dict[str, Tier1Record] = {}
    for record in records:
        if record.is_canary and record.canary_id is not None:
            indexed[record.canary_id] = record
    return indexed


def _ordered_non_canary_records(records: Sequence[Tier1Record]) -> tuple[Tier1Record, ...]:
    record_ids = [record.record_id for record in records]
    if len(set(record_ids)) != len(record_ids):
        raise MixedTrafficValidationError("Tier 1 record_id values must be unique.")
    return tuple(
        sorted(
            (record for record in records if not record.is_canary and record.canary_id is None),
            key=lambda record: record.record_id,
        )
    )


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
