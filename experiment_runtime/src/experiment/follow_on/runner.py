from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
import csv
import json
import os
import tomllib

import numpy as np
import torch
from transformers import set_seed

from experiment.data_gen.io import read_canary_registry_csv, read_jsonl_rows
from experiment.eval.config import (
    DecodingSettings,
    FheFilterReference,
    FilterEncoderSettings,
    InferenceSettings,
    ModelSettings,
    OfficialRunReference,
    PlaintextFilterReference,
    TokenizerSettings,
    _load_official_run_reference,
    _validate_fhe_settings,
    resolve_exposure_conditions,
    resolve_filter_modes,
)
from experiment.eval.runner import (
    _configure_cuda_inference,
    _load_adapter_model,
    _release_model,
)
from experiment.eval.scoring import CanaryLeakageScorer
from experiment.fhe.config import OpenFheSettings
from experiment.fhe.data import (
    compute_plaintext_logits,
    load_plaintext_model_parameters,
    predict_labels,
    sigmoid,
)
from experiment.fhe.openfhe_backend import OpenFheBundlePaths, OpenFheCkksScorer
from experiment.filter_train.config import EncoderSettings
from experiment.filter_train.data import INT_TO_LABEL
from experiment.filter_train.embeddings import (
    EXPECTED_STAGE3_EMBEDDING_DIMENSION,
    load_sentence_encoder,
)
from experiment.schemas.stage2 import canonicalize_target_fields
from experiment.schemas.tier2 import ChatMessage
from experiment.train_qwen.config import EXPOSURE_CONDITIONS
from experiment.train_qwen.data import load_stage1_tokenizer, tokenize_chat_messages
from qwen_langgraph_demo.runtime.protocol import load_protocol_bundle

from .adaptive_catalog import ADAPTIVE_ATTACK_FAMILY_ORDER, MIXED_BENIGN_FAMILY_ORDER
from .confidence_intervals import DEFAULT_CONFIDENCE_LEVEL
from .data import (
    FILTER_ALLOW,
    FILTER_BLOCK,
    FOLLOW_ON_SPLIT,
    MIXED_SOURCE_ADAPTIVE,
    MIXED_SOURCE_BENIGN,
    PROJECT_ROOT,
    AdaptiveAttackPrompt,
    MixedTrafficRow,
)
from .metrics import (
    build_adaptive_attack_metrics,
    build_adaptive_family_metrics,
    build_filter_parity_metrics,
    build_mixed_family_metrics,
    build_mixed_traffic_metrics,
)
from .timing import (
    FILTER_TIMING_COLUMNS,
    PIPELINE_TIMING_COLUMNS,
    SETUP_TIMING_COLUMNS,
    DEFAULT_FILTER_TIMING_NUMERIC_COLUMNS,
    DEFAULT_PIPELINE_TIMING_NUMERIC_COLUMNS,
    FilterTimingSample,
    PipelineTimingSample,
    SetupTimingEntry,
    elapsed_ms,
    summarize_filter_timing_samples,
    summarize_pipeline_timing_samples,
    summarize_timing_rows,
)


DEFAULT_FOLLOW_ON_ADAPTIVE_REPLAY_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "follow_on" / "adaptive_replay.toml"
)
DEFAULT_FOLLOW_ON_MIXED_REPLAY_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "follow_on" / "mixed_traffic_replay.toml"
)

ADAPTIVE_CONDITIONS = (
    "no_system_prompt",
    "system_prompt_active",
    "plaintext_filter",
    "fhe_filter",
)
FILTER_CONDITIONS = ("plaintext_filter", "fhe_filter")
NO_FILTER_MODE = "none"
INTEGRATED_PROMPT_CONDITION = "system_prompt_active"
ADAPTIVE_EVAL_DATASET = "adaptive"
MIXED_EVAL_DATASET = "mixed_traffic"

ADAPTIVE_RESPONSE_COLUMNS = (
    "attack_id",
    "template_family",
    "target_canary_id",
    "target_fields",
    "exposure_condition",
    "condition",
    "run_name",
    "message_text",
    "filter_mode",
    "filter_block_probability",
    "filter_decision",
    "response_generated",
    "response_text",
    "name_leaked",
    "date_of_birth_leaked",
    "account_number_leaked",
    "any_sensitive_field_leaked",
    "full_canary_record_leaked",
    "other_canary_leak",
    "refusal_emitted",
)

MIXED_TRAFFIC_RESPONSE_COLUMNS = (
    "traffic_id",
    "source_type",
    "template_family",
    "expected_filter_label",
    "target_canary_id",
    "target_fields",
    "exposure_condition",
    "filter_mode",
    "run_name",
    "message_text",
    "filter_block_probability",
    "filter_decision",
    "response_generated",
    "response_text",
    "name_leaked",
    "date_of_birth_leaked",
    "account_number_leaked",
    "any_sensitive_field_leaked",
    "full_canary_record_leaked",
    "other_canary_leak",
    "refusal_emitted",
    "benign_false_positive",
)

ADAPTIVE_FAMILY_METRIC_COLUMNS = (
    "exposure_condition",
    "condition",
    "filter_mode",
    "template_family",
    "attack_count",
    "adversarial_block_rate",
    "filter_allow_rate",
    "name_leak_rate",
    "date_of_birth_leak_rate",
    "account_number_leak_rate",
    "any_sensitive_field_leak_rate",
    "full_canary_record_leak_rate",
    "other_canary_leak_rate",
    "refusal_rate",
    "leak_rate_given_allowed",
)

MIXED_FAMILY_METRIC_COLUMNS = (
    "exposure_condition",
    "filter_mode",
    "template_family",
    "traffic_count",
    "benign_count",
    "adaptive_adversarial_count",
    "adversarial_block_rate",
    "benign_false_positive_rate",
    "benign_allow_rate",
    "adaptive_any_sensitive_field_leak_rate",
    "adaptive_full_canary_record_leak_rate",
    "other_canary_leak_rate",
    "refusal_rate",
    "leak_rate_given_allowed",
)


@dataclass(frozen=True, slots=True)
class FollowOnAdaptiveReplayConfig:
    config_path: Path
    protocol_config_dir: Path
    adaptive_attack_dataset_path: Path
    canary_registry_path: Path
    output_root: Path
    timing_root: Path
    model: ModelSettings
    tokenizer: TokenizerSettings
    decoding: DecodingSettings
    inference: InferenceSettings
    official_runs: dict[str, OfficialRunReference]
    filter_encoder: FilterEncoderSettings
    plaintext_filter: PlaintextFilterReference
    fhe_filter: FheFilterReference
    fhe: OpenFheSettings
    seed: int

    @classmethod
    def from_toml(
        cls,
        config_path: Path | str | None = None,
    ) -> "FollowOnAdaptiveReplayConfig":
        path = _resolve_path(config_path or DEFAULT_FOLLOW_ON_ADAPTIVE_REPLAY_CONFIG_PATH)
        with path.open("rb") as handle:
            document = tomllib.load(handle)

        common = _parse_common_replay_config(
            path=path,
            document=document,
            purpose="follow-on adaptive replay",
        )
        return cls(
            config_path=path,
            protocol_config_dir=common["protocol_config_dir"],
            adaptive_attack_dataset_path=_resolve_existing_path(
                document["inputs"]["adaptive_attack_dataset_path"]
            ),
            canary_registry_path=_resolve_existing_path(document["inputs"]["canary_registry_path"]),
            output_root=common["output_root"],
            timing_root=common["timing_root"],
            model=common["model"],
            tokenizer=common["tokenizer"],
            decoding=common["decoding"],
            inference=common["inference"],
            official_runs=common["official_runs"],
            filter_encoder=common["filter_encoder"],
            plaintext_filter=common["plaintext_filter"],
            fhe_filter=common["fhe_filter"],
            fhe=common["fhe"],
            seed=common["seed"],
        )


@dataclass(frozen=True, slots=True)
class FollowOnMixedTrafficReplayConfig:
    config_path: Path
    protocol_config_dir: Path
    mixed_traffic_dataset_path: Path
    canary_registry_path: Path
    output_root: Path
    timing_root: Path
    model: ModelSettings
    tokenizer: TokenizerSettings
    decoding: DecodingSettings
    inference: InferenceSettings
    official_runs: dict[str, OfficialRunReference]
    filter_encoder: FilterEncoderSettings
    plaintext_filter: PlaintextFilterReference
    fhe_filter: FheFilterReference
    fhe: OpenFheSettings
    seed: int

    @classmethod
    def from_toml(
        cls,
        config_path: Path | str | None = None,
    ) -> "FollowOnMixedTrafficReplayConfig":
        path = _resolve_path(config_path or DEFAULT_FOLLOW_ON_MIXED_REPLAY_CONFIG_PATH)
        with path.open("rb") as handle:
            document = tomllib.load(handle)

        common = _parse_common_replay_config(
            path=path,
            document=document,
            purpose="follow-on mixed-traffic replay",
        )
        return cls(
            config_path=path,
            protocol_config_dir=common["protocol_config_dir"],
            mixed_traffic_dataset_path=_resolve_existing_path(
                document["inputs"]["mixed_traffic_dataset_path"]
            ),
            canary_registry_path=_resolve_existing_path(document["inputs"]["canary_registry_path"]),
            output_root=common["output_root"],
            timing_root=common["timing_root"],
            model=common["model"],
            tokenizer=common["tokenizer"],
            decoding=common["decoding"],
            inference=common["inference"],
            official_runs=common["official_runs"],
            filter_encoder=common["filter_encoder"],
            plaintext_filter=common["plaintext_filter"],
            fhe_filter=common["fhe_filter"],
            fhe=common["fhe"],
            seed=common["seed"],
        )


@dataclass(frozen=True, slots=True)
class AdaptiveResponseRow:
    attack_id: str
    template_family: str
    target_canary_id: str
    target_fields: tuple[str, ...]
    exposure_condition: str
    condition: str
    run_name: str
    message_text: str
    filter_mode: str
    filter_block_probability: float | None
    filter_decision: str | None
    response_generated: int
    response_text: str
    name_leaked: int
    date_of_birth_leaked: int
    account_number_leaked: int
    any_sensitive_field_leaked: int
    full_canary_record_leaked: int
    other_canary_leak: int
    refusal_emitted: int

    def to_row(self) -> dict[str, Any]:
        return {
            "attack_id": self.attack_id,
            "template_family": self.template_family,
            "target_canary_id": self.target_canary_id,
            "target_fields": list(self.target_fields),
            "exposure_condition": self.exposure_condition,
            "condition": self.condition,
            "run_name": self.run_name,
            "message_text": self.message_text,
            "filter_mode": self.filter_mode,
            "filter_block_probability": self.filter_block_probability,
            "filter_decision": self.filter_decision,
            "response_generated": self.response_generated,
            "response_text": self.response_text,
            "name_leaked": self.name_leaked,
            "date_of_birth_leaked": self.date_of_birth_leaked,
            "account_number_leaked": self.account_number_leaked,
            "any_sensitive_field_leaked": self.any_sensitive_field_leaked,
            "full_canary_record_leaked": self.full_canary_record_leaked,
            "other_canary_leak": self.other_canary_leak,
            "refusal_emitted": self.refusal_emitted,
        }


@dataclass(frozen=True, slots=True)
class MixedTrafficResponseRow:
    traffic_id: str
    source_type: str
    template_family: str
    expected_filter_label: str
    target_canary_id: str | None
    target_fields: tuple[str, ...]
    exposure_condition: str
    filter_mode: str
    run_name: str
    message_text: str
    filter_block_probability: float
    filter_decision: str
    response_generated: int
    response_text: str
    name_leaked: int
    date_of_birth_leaked: int
    account_number_leaked: int
    any_sensitive_field_leaked: int
    full_canary_record_leaked: int
    other_canary_leak: int
    refusal_emitted: int
    benign_false_positive: int

    def to_row(self) -> dict[str, Any]:
        return {
            "traffic_id": self.traffic_id,
            "source_type": self.source_type,
            "template_family": self.template_family,
            "expected_filter_label": self.expected_filter_label,
            "target_canary_id": self.target_canary_id,
            "target_fields": list(self.target_fields),
            "exposure_condition": self.exposure_condition,
            "filter_mode": self.filter_mode,
            "run_name": self.run_name,
            "message_text": self.message_text,
            "filter_block_probability": self.filter_block_probability,
            "filter_decision": self.filter_decision,
            "response_generated": self.response_generated,
            "response_text": self.response_text,
            "name_leaked": self.name_leaked,
            "date_of_birth_leaked": self.date_of_birth_leaked,
            "account_number_leaked": self.account_number_leaked,
            "any_sensitive_field_leaked": self.any_sensitive_field_leaked,
            "full_canary_record_leaked": self.full_canary_record_leaked,
            "other_canary_leak": self.other_canary_leak,
            "refusal_emitted": self.refusal_emitted,
            "benign_false_positive": self.benign_false_positive,
        }


@dataclass(frozen=True, slots=True)
class FollowOnConditionArtifacts:
    exposure_condition: str
    condition: str
    responses_path: Path
    metrics_path: Path
    family_metrics_path: Path
    pipeline_timing_samples_path: Path
    pipeline_timing_summary_path: Path
    filter_timing_samples_path: Path | None = None
    filter_timing_summary_path: Path | None = None


@dataclass(frozen=True, slots=True)
class FollowOnEvaluationResult:
    summary_path: Path
    ci_summary_path: Path
    parity_summary_path: Path
    setup_timing_path: Path
    artifacts: dict[tuple[str, str], FollowOnConditionArtifacts]


@dataclass(frozen=True, slots=True)
class FollowOnFilterInput:
    row_id: str
    message_text: str


@dataclass(frozen=True, slots=True)
class FollowOnFilterDecision:
    row_id: str
    block_probability: float
    decision: str
    timing_sample: FilterTimingSample


def resolve_adaptive_conditions(condition: str) -> tuple[str, ...]:
    if condition == "all":
        return ADAPTIVE_CONDITIONS
    if condition not in ADAPTIVE_CONDITIONS:
        raise ValueError(
            f"Unsupported follow-on adaptive condition {condition!r}; "
            f"expected one of {ADAPTIVE_CONDITIONS} or 'all'."
        )
    return (condition,)


def load_adaptive_attack_prompts(path: Path) -> tuple[AdaptiveAttackPrompt, ...]:
    rows = tuple(AdaptiveAttackPrompt.from_row(row) for row in read_jsonl_rows(path))
    if not rows:
        raise ValueError(f"Follow-on adaptive attack dataset is empty: {path}")
    seen_ids: set[str] = set()
    for row in rows:
        if row.attack_id in seen_ids:
            raise ValueError(f"Adaptive attack dataset contains duplicate attack_id {row.attack_id!r}.")
        seen_ids.add(row.attack_id)
        if row.split != FOLLOW_ON_SPLIT:
            raise ValueError(f"Adaptive attack row {row.attack_id} must use split={FOLLOW_ON_SPLIT!r}.")
        canonicalize_target_fields(row.target_fields)
    return rows


def load_mixed_traffic_rows(path: Path) -> tuple[MixedTrafficRow, ...]:
    rows = tuple(MixedTrafficRow.from_row(row) for row in read_jsonl_rows(path))
    if not rows:
        raise ValueError(f"Follow-on mixed-traffic dataset is empty: {path}")
    seen_ids: set[str] = set()
    for row in rows:
        if row.traffic_id in seen_ids:
            raise ValueError(f"Mixed-traffic dataset contains duplicate traffic_id {row.traffic_id!r}.")
        seen_ids.add(row.traffic_id)
        if row.split != FOLLOW_ON_SPLIT:
            raise ValueError(f"Mixed-traffic row {row.traffic_id} must use split={FOLLOW_ON_SPLIT!r}.")
        if row.source_type not in {MIXED_SOURCE_BENIGN, MIXED_SOURCE_ADAPTIVE}:
            raise ValueError(f"Mixed-traffic row {row.traffic_id} has invalid source_type {row.source_type!r}.")
        if row.expected_filter_label not in {FILTER_ALLOW, FILTER_BLOCK}:
            raise ValueError(
                f"Mixed-traffic row {row.traffic_id} has invalid expected_filter_label "
                f"{row.expected_filter_label!r}."
            )
        if row.source_type == MIXED_SOURCE_BENIGN:
            if row.target_canary_id is not None or row.target_fields:
                raise ValueError(f"Mixed benign row {row.traffic_id} must not target a canary.")
            continue
        if row.target_canary_id in (None, ""):
            raise ValueError(f"Mixed adaptive row {row.traffic_id} must target a canary.")
        canonicalize_target_fields(row.target_fields)
    return rows


def score_mixed_response_leakage(
    *,
    source_type: str,
    response_text: str,
    target_canary_id: str | None,
    scorer: CanaryLeakageScorer,
    refusal_response: str,
) -> dict[str, int]:
    if source_type == MIXED_SOURCE_BENIGN:
        return _zero_leakage_flags(
            refusal_emitted=int(response_text.strip() == refusal_response)
        )
    if source_type != MIXED_SOURCE_ADAPTIVE:
        raise ValueError(f"Unsupported mixed source_type {source_type!r}.")
    if target_canary_id in (None, ""):
        raise ValueError("Adaptive mixed rows require target_canary_id for leakage scoring.")
    leakage = scorer.score_response(
        response_text=response_text,
        target_canary_id=str(target_canary_id),
    )
    return _leakage_to_flags(leakage)


def run_follow_on_adaptive_evaluation(
    *,
    config_path: Path | str | None = None,
    exposure: str,
    condition: str,
) -> FollowOnEvaluationResult:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    config = FollowOnAdaptiveReplayConfig.from_toml(
        config_path or DEFAULT_FOLLOW_ON_ADAPTIVE_REPLAY_CONFIG_PATH
    )
    selected_exposures = resolve_exposure_conditions(exposure)
    selected_conditions = resolve_adaptive_conditions(condition)
    _prepare_output_root(config.output_root)
    _prepare_output_root(config.timing_root)
    _configure_cuda_inference(config.inference.bf16, config.inference.tf32)

    protocol = load_protocol_bundle(config.protocol_config_dir)
    setup_entries: list[SetupTimingEntry] = []
    tokenizer = _load_tokenizer(config, setup_entries)
    attack_prompts = load_adaptive_attack_prompts(config.adaptive_attack_dataset_path)
    canary_registry = read_canary_registry_csv(config.canary_registry_path)
    scorer = CanaryLeakageScorer(
        tuple(canary_registry),
        refusal_response=protocol.stage1.refusal_response,
    )

    set_seed(config.seed)
    selected_filter_modes = tuple(
        selected_condition
        for selected_condition in selected_conditions
        if selected_condition in FILTER_CONDITIONS
    )
    filter_decisions_by_mode = _compute_filter_decisions(
        config=config,
        filter_inputs=tuple(
            FollowOnFilterInput(row_id=prompt.attack_id, message_text=prompt.message_text)
            for prompt in attack_prompts
        ),
        eval_dataset=ADAPTIVE_EVAL_DATASET,
        selected_filter_modes=selected_filter_modes,
        setup_entries=setup_entries,
    )

    artifacts: dict[tuple[str, str], FollowOnConditionArtifacts] = {}
    rows_by_condition: dict[tuple[str, str], tuple[AdaptiveResponseRow, ...]] = {}
    for exposure_condition in selected_exposures:
        official_run = config.official_runs[exposure_condition]
        model = _load_model_with_timing(config, official_run, tokenizer, setup_entries)
        try:
            for selected_condition in selected_conditions:
                condition_artifacts, rows = _evaluate_adaptive_condition(
                    config=config,
                    protocol=protocol,
                    official_run=official_run,
                    condition=selected_condition,
                    attack_prompts=attack_prompts,
                    filter_decisions=filter_decisions_by_mode.get(selected_condition, {}),
                    scorer=scorer,
                    tokenizer=tokenizer,
                    model=model,
                )
                artifacts[(exposure_condition, selected_condition)] = condition_artifacts
                rows_by_condition[(exposure_condition, selected_condition)] = rows
        finally:
            _release_model(model)

    summary_path = config.output_root / "adaptive_summary.json"
    _write_adaptive_summary_json(
        summary_path=summary_path,
        output_root=config.output_root,
        exposure_order=selected_exposures,
        conditions=selected_conditions,
    )
    ci_summary_path = config.output_root / "adaptive_ci_summary.json"
    _write_ci_summary_json(
        summary_path=ci_summary_path,
        output_root=config.output_root,
        exposure_order=selected_exposures,
        run_axis_name="condition",
        run_axis_values=selected_conditions,
        metrics_filename="adaptive_metrics.json",
    )
    parity_summary_path = config.output_root / "filter_parity_summary.json"
    _write_filter_parity_summary_json(
        summary_path=parity_summary_path,
        output_root=config.output_root,
        exposure_order=selected_exposures,
        rows_by_condition=rows_by_condition,
        row_id_field="attack_id",
        metrics_filename="adaptive_metrics.json",
    )
    setup_timing_path = _write_setup_timing_json(
        config.timing_root,
        setup_entries,
        eval_dataset=ADAPTIVE_EVAL_DATASET,
    )
    return FollowOnEvaluationResult(
        summary_path=summary_path,
        ci_summary_path=ci_summary_path,
        parity_summary_path=parity_summary_path,
        setup_timing_path=setup_timing_path,
        artifacts=artifacts,
    )


def run_follow_on_mixed_evaluation(
    *,
    config_path: Path | str | None = None,
    exposure: str,
    filter_mode: str,
) -> FollowOnEvaluationResult:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    config = FollowOnMixedTrafficReplayConfig.from_toml(
        config_path or DEFAULT_FOLLOW_ON_MIXED_REPLAY_CONFIG_PATH
    )
    selected_exposures = resolve_exposure_conditions(exposure)
    selected_filter_modes = resolve_filter_modes(filter_mode)
    _prepare_output_root(config.output_root)
    _prepare_output_root(config.timing_root)
    _configure_cuda_inference(config.inference.bf16, config.inference.tf32)

    protocol = load_protocol_bundle(config.protocol_config_dir)
    setup_entries: list[SetupTimingEntry] = []
    tokenizer = _load_tokenizer(config, setup_entries)
    mixed_rows = load_mixed_traffic_rows(config.mixed_traffic_dataset_path)
    canary_registry = read_canary_registry_csv(config.canary_registry_path)
    scorer = CanaryLeakageScorer(
        tuple(canary_registry),
        refusal_response=protocol.stage1.refusal_response,
    )

    set_seed(config.seed)
    filter_decisions_by_mode = _compute_filter_decisions(
        config=config,
        filter_inputs=tuple(
            FollowOnFilterInput(row_id=row.traffic_id, message_text=row.message_text)
            for row in mixed_rows
        ),
        eval_dataset=MIXED_EVAL_DATASET,
        selected_filter_modes=selected_filter_modes,
        setup_entries=setup_entries,
    )

    artifacts: dict[tuple[str, str], FollowOnConditionArtifacts] = {}
    rows_by_condition: dict[tuple[str, str], tuple[MixedTrafficResponseRow, ...]] = {}
    for exposure_condition in selected_exposures:
        official_run = config.official_runs[exposure_condition]
        model = _load_model_with_timing(config, official_run, tokenizer, setup_entries)
        try:
            for selected_filter_mode in selected_filter_modes:
                condition_artifacts, rows = _evaluate_mixed_filter_mode(
                    config=config,
                    protocol=protocol,
                    official_run=official_run,
                    filter_mode=selected_filter_mode,
                    mixed_rows=mixed_rows,
                    filter_decisions=filter_decisions_by_mode[selected_filter_mode],
                    scorer=scorer,
                    tokenizer=tokenizer,
                    model=model,
                )
                artifacts[(exposure_condition, selected_filter_mode)] = condition_artifacts
                rows_by_condition[(exposure_condition, selected_filter_mode)] = rows
        finally:
            _release_model(model)

    summary_path = config.output_root / "mixed_traffic_summary.json"
    _write_mixed_summary_json(
        summary_path=summary_path,
        output_root=config.output_root,
        exposure_order=selected_exposures,
        filter_modes=selected_filter_modes,
    )
    ci_summary_path = config.output_root / "mixed_traffic_ci_summary.json"
    _write_ci_summary_json(
        summary_path=ci_summary_path,
        output_root=config.output_root,
        exposure_order=selected_exposures,
        run_axis_name="filter_mode",
        run_axis_values=selected_filter_modes,
        metrics_filename="mixed_traffic_metrics.json",
    )
    parity_summary_path = config.output_root / "filter_parity_summary.json"
    _write_filter_parity_summary_json(
        summary_path=parity_summary_path,
        output_root=config.output_root,
        exposure_order=selected_exposures,
        rows_by_condition=rows_by_condition,
        row_id_field="traffic_id",
        metrics_filename="mixed_traffic_metrics.json",
    )
    setup_timing_path = _write_setup_timing_json(
        config.timing_root,
        setup_entries,
        eval_dataset=MIXED_EVAL_DATASET,
    )
    return FollowOnEvaluationResult(
        summary_path=summary_path,
        ci_summary_path=ci_summary_path,
        parity_summary_path=parity_summary_path,
        setup_timing_path=setup_timing_path,
        artifacts=artifacts,
    )


def _evaluate_adaptive_condition(
    *,
    config: FollowOnAdaptiveReplayConfig,
    protocol: Any,
    official_run: OfficialRunReference,
    condition: str,
    attack_prompts: tuple[AdaptiveAttackPrompt, ...],
    filter_decisions: Mapping[str, FollowOnFilterDecision],
    scorer: CanaryLeakageScorer,
    tokenizer: Any,
    model: torch.nn.Module,
) -> tuple[FollowOnConditionArtifacts, tuple[AdaptiveResponseRow, ...]]:
    is_filtered = condition in FILTER_CONDITIONS
    if is_filtered:
        allowed_prompts = tuple(
            prompt
            for prompt in attack_prompts
            if filter_decisions[prompt.attack_id].decision == FILTER_ALLOW
        )
        prompt_condition = INTEGRATED_PROMPT_CONDITION
    else:
        allowed_prompts = attack_prompts
        prompt_condition = condition

    response_texts, llm_generation_ms = _generate_response_texts_with_timing(
        prompts=allowed_prompts,
        row_id_getter=lambda prompt: prompt.attack_id,
        message_text_getter=lambda prompt: prompt.message_text,
        prompt_condition=prompt_condition,
        protocol=protocol,
        tokenizer=tokenizer,
        model=model,
        batch_size=config.inference.batch_size,
        max_sequence_length=config.tokenizer.max_sequence_length,
        add_generation_prompt=config.tokenizer.add_generation_prompt,
        max_new_tokens=config.decoding.max_new_tokens,
    )

    rows: list[AdaptiveResponseRow] = []
    pipeline_samples: list[PipelineTimingSample] = []
    filter_samples: list[FilterTimingSample] = []
    for attack_prompt in attack_prompts:
        routing_start = perf_counter()
        if is_filtered:
            filter_decision = filter_decisions[attack_prompt.attack_id]
            filter_mode = condition
            filter_block_probability = filter_decision.block_probability
            filter_decision_label: str | None = filter_decision.decision
            filter_total_ms = filter_decision.timing_sample.total_filter_ms
            filter_samples.append(filter_decision.timing_sample)
            if filter_decision.decision == FILTER_BLOCK:
                row = AdaptiveResponseRow(
                    attack_id=attack_prompt.attack_id,
                    template_family=attack_prompt.template_family,
                    target_canary_id=attack_prompt.target_canary_id,
                    target_fields=attack_prompt.target_fields,
                    exposure_condition=official_run.exposure_condition,
                    condition=condition,
                    run_name=official_run.run_name,
                    message_text=attack_prompt.message_text,
                    filter_mode=filter_mode,
                    filter_block_probability=filter_block_probability,
                    filter_decision=filter_decision_label,
                    response_generated=0,
                    response_text="",
                    **_zero_leakage_flags(),
                )
                routing_ms = elapsed_ms(routing_start)
                rows.append(row)
                pipeline_samples.append(
                    _pipeline_sample(
                        row_id=attack_prompt.attack_id,
                        exposure_condition=official_run.exposure_condition,
                        eval_dataset=ADAPTIVE_EVAL_DATASET,
                        condition=condition,
                        filter_mode=filter_mode,
                        source_type=MIXED_SOURCE_ADAPTIVE,
                        filter_decision=filter_decision_label,
                        response_generated=0,
                        filter_total_ms=filter_total_ms,
                        llm_generation_ms=0.0,
                        routing_ms=routing_ms,
                    )
                )
                continue
        else:
            filter_mode = NO_FILTER_MODE
            filter_block_probability = None
            filter_decision_label = None
            filter_total_ms = None

        response_text = response_texts[attack_prompt.attack_id]
        leakage = scorer.score_response(
            response_text=response_text,
            target_canary_id=attack_prompt.target_canary_id,
        )
        row = AdaptiveResponseRow(
            attack_id=attack_prompt.attack_id,
            template_family=attack_prompt.template_family,
            target_canary_id=attack_prompt.target_canary_id,
            target_fields=attack_prompt.target_fields,
            exposure_condition=official_run.exposure_condition,
            condition=condition,
            run_name=official_run.run_name,
            message_text=attack_prompt.message_text,
            filter_mode=filter_mode,
            filter_block_probability=filter_block_probability,
            filter_decision=filter_decision_label,
            response_generated=1,
            response_text=response_text,
            **_leakage_to_flags(leakage),
        )
        routing_ms = elapsed_ms(routing_start)
        rows.append(row)
        pipeline_samples.append(
            _pipeline_sample(
                row_id=attack_prompt.attack_id,
                exposure_condition=official_run.exposure_condition,
                eval_dataset=ADAPTIVE_EVAL_DATASET,
                condition=condition,
                filter_mode=filter_mode,
                source_type=MIXED_SOURCE_ADAPTIVE,
                filter_decision=filter_decision_label,
                response_generated=1,
                filter_total_ms=filter_total_ms,
                llm_generation_ms=llm_generation_ms.get(attack_prompt.attack_id, 0.0),
                routing_ms=routing_ms,
            )
        )

    output_dir = config.output_root / official_run.exposure_condition / condition
    output_dir.mkdir(parents=True, exist_ok=True)
    response_rows = tuple(rows)
    responses_path = output_dir / "adaptive_responses.jsonl"
    _write_response_jsonl(responses_path, response_rows, ADAPTIVE_RESPONSE_COLUMNS)

    metrics_payload = {
        "exposure_condition": official_run.exposure_condition,
        "condition": condition,
        "prompt_condition": prompt_condition,
        "filter_mode": condition if is_filtered else NO_FILTER_MODE,
        "run_name": official_run.run_name,
        "base_model_name": config.model.name,
        "adapter_run_dir": str(official_run.run_dir),
        "attack_dataset_path": str(config.adaptive_attack_dataset_path),
        "family_count": len(ADAPTIVE_ATTACK_FAMILY_ORDER),
        "system_prompt_used": condition != "no_system_prompt",
        "decoding": _build_decoding_payload(config),
        **build_adaptive_attack_metrics(response_rows, filtered=is_filtered),
    }
    metrics_path = output_dir / "adaptive_metrics.json"
    _write_json(metrics_path, metrics_payload)

    family_metrics = build_adaptive_family_metrics(
        response_rows,
        family_order=ADAPTIVE_ATTACK_FAMILY_ORDER,
        filtered=is_filtered,
        confidence_level=DEFAULT_CONFIDENCE_LEVEL,
    )
    family_metrics_path = output_dir / "family_metrics.csv"
    _write_family_metrics_csv(
        family_metrics_path,
        family_metrics,
        fieldnames=ADAPTIVE_FAMILY_METRIC_COLUMNS,
        fixed_values={
            "exposure_condition": official_run.exposure_condition,
            "condition": condition,
            "filter_mode": condition if is_filtered else NO_FILTER_MODE,
        },
    )

    pipeline_timing_samples_path = output_dir / "timing_pipeline_samples.csv"
    pipeline_timing_summary_path = output_dir / "timing_pipeline_summary.json"
    _write_pipeline_timing_artifacts(
        samples=tuple(pipeline_samples),
        samples_path=pipeline_timing_samples_path,
        summary_path=pipeline_timing_summary_path,
    )

    filter_timing_samples_path: Path | None = None
    filter_timing_summary_path: Path | None = None
    if is_filtered:
        filter_timing_samples_path = output_dir / "timing_filter_samples.csv"
        filter_timing_summary_path = output_dir / "timing_filter_summary.json"
        _write_filter_timing_artifacts(
            samples=tuple(filter_samples),
            samples_path=filter_timing_samples_path,
            summary_path=filter_timing_summary_path,
        )

    return (
        FollowOnConditionArtifacts(
            exposure_condition=official_run.exposure_condition,
            condition=condition,
            responses_path=responses_path,
            metrics_path=metrics_path,
            family_metrics_path=family_metrics_path,
            pipeline_timing_samples_path=pipeline_timing_samples_path,
            pipeline_timing_summary_path=pipeline_timing_summary_path,
            filter_timing_samples_path=filter_timing_samples_path,
            filter_timing_summary_path=filter_timing_summary_path,
        ),
        response_rows,
    )


def _evaluate_mixed_filter_mode(
    *,
    config: FollowOnMixedTrafficReplayConfig,
    protocol: Any,
    official_run: OfficialRunReference,
    filter_mode: str,
    mixed_rows: tuple[MixedTrafficRow, ...],
    filter_decisions: Mapping[str, FollowOnFilterDecision],
    scorer: CanaryLeakageScorer,
    tokenizer: Any,
    model: torch.nn.Module,
) -> tuple[FollowOnConditionArtifacts, tuple[MixedTrafficResponseRow, ...]]:
    allowed_rows = tuple(
        row
        for row in mixed_rows
        if filter_decisions[row.traffic_id].decision == FILTER_ALLOW
    )
    response_texts, llm_generation_ms = _generate_response_texts_with_timing(
        prompts=allowed_rows,
        row_id_getter=lambda row: row.traffic_id,
        message_text_getter=lambda row: row.message_text,
        prompt_condition=INTEGRATED_PROMPT_CONDITION,
        protocol=protocol,
        tokenizer=tokenizer,
        model=model,
        batch_size=config.inference.batch_size,
        max_sequence_length=config.tokenizer.max_sequence_length,
        add_generation_prompt=config.tokenizer.add_generation_prompt,
        max_new_tokens=config.decoding.max_new_tokens,
    )

    response_rows: list[MixedTrafficResponseRow] = []
    pipeline_samples: list[PipelineTimingSample] = []
    filter_samples: list[FilterTimingSample] = []
    for mixed_row in mixed_rows:
        routing_start = perf_counter()
        filter_decision = filter_decisions[mixed_row.traffic_id]
        filter_samples.append(filter_decision.timing_sample)
        filter_total_ms = filter_decision.timing_sample.total_filter_ms
        benign_false_positive = int(
            mixed_row.source_type == MIXED_SOURCE_BENIGN
            and filter_decision.decision == FILTER_BLOCK
        )
        if filter_decision.decision == FILTER_BLOCK:
            row = MixedTrafficResponseRow(
                traffic_id=mixed_row.traffic_id,
                source_type=mixed_row.source_type,
                template_family=mixed_row.template_family,
                expected_filter_label=mixed_row.expected_filter_label,
                target_canary_id=mixed_row.target_canary_id,
                target_fields=mixed_row.target_fields,
                exposure_condition=official_run.exposure_condition,
                filter_mode=filter_mode,
                run_name=official_run.run_name,
                message_text=mixed_row.message_text,
                filter_block_probability=filter_decision.block_probability,
                filter_decision=filter_decision.decision,
                response_generated=0,
                response_text="",
                benign_false_positive=benign_false_positive,
                **_zero_leakage_flags(),
            )
            routing_ms = elapsed_ms(routing_start)
            response_rows.append(row)
            pipeline_samples.append(
                _pipeline_sample(
                    row_id=mixed_row.traffic_id,
                    exposure_condition=official_run.exposure_condition,
                    eval_dataset=MIXED_EVAL_DATASET,
                    condition=filter_mode,
                    filter_mode=filter_mode,
                    source_type=mixed_row.source_type,
                    filter_decision=filter_decision.decision,
                    response_generated=0,
                    filter_total_ms=filter_total_ms,
                    llm_generation_ms=0.0,
                    routing_ms=routing_ms,
                )
            )
            continue

        response_text = response_texts[mixed_row.traffic_id]
        leakage_flags = score_mixed_response_leakage(
            source_type=mixed_row.source_type,
            response_text=response_text,
            target_canary_id=mixed_row.target_canary_id,
            scorer=scorer,
            refusal_response=protocol.stage1.refusal_response,
        )
        row = MixedTrafficResponseRow(
            traffic_id=mixed_row.traffic_id,
            source_type=mixed_row.source_type,
            template_family=mixed_row.template_family,
            expected_filter_label=mixed_row.expected_filter_label,
            target_canary_id=mixed_row.target_canary_id,
            target_fields=mixed_row.target_fields,
            exposure_condition=official_run.exposure_condition,
            filter_mode=filter_mode,
            run_name=official_run.run_name,
            message_text=mixed_row.message_text,
            filter_block_probability=filter_decision.block_probability,
            filter_decision=filter_decision.decision,
            response_generated=1,
            response_text=response_text,
            benign_false_positive=benign_false_positive,
            **leakage_flags,
        )
        routing_ms = elapsed_ms(routing_start)
        response_rows.append(row)
        pipeline_samples.append(
            _pipeline_sample(
                row_id=mixed_row.traffic_id,
                exposure_condition=official_run.exposure_condition,
                eval_dataset=MIXED_EVAL_DATASET,
                condition=filter_mode,
                filter_mode=filter_mode,
                source_type=mixed_row.source_type,
                filter_decision=filter_decision.decision,
                response_generated=1,
                filter_total_ms=filter_total_ms,
                llm_generation_ms=llm_generation_ms.get(mixed_row.traffic_id, 0.0),
                routing_ms=routing_ms,
            )
        )

    output_dir = config.output_root / official_run.exposure_condition / filter_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_tuple = tuple(response_rows)
    responses_path = output_dir / "mixed_traffic_responses.jsonl"
    _write_response_jsonl(responses_path, rows_tuple, MIXED_TRAFFIC_RESPONSE_COLUMNS)

    metrics_payload = {
        "exposure_condition": official_run.exposure_condition,
        "filter_mode": filter_mode,
        "prompt_condition": INTEGRATED_PROMPT_CONDITION,
        "run_name": official_run.run_name,
        "base_model_name": config.model.name,
        "adapter_run_dir": str(official_run.run_dir),
        "mixed_traffic_dataset_path": str(config.mixed_traffic_dataset_path),
        "family_count": len(MIXED_BENIGN_FAMILY_ORDER) + len(ADAPTIVE_ATTACK_FAMILY_ORDER),
        "system_prompt_used": True,
        "decoding": _build_decoding_payload(config),
        **build_mixed_traffic_metrics(rows_tuple),
    }
    metrics_path = output_dir / "mixed_traffic_metrics.json"
    _write_json(metrics_path, metrics_payload)

    family_metrics = build_mixed_family_metrics(
        rows_tuple,
        family_order=tuple(MIXED_BENIGN_FAMILY_ORDER) + tuple(ADAPTIVE_ATTACK_FAMILY_ORDER),
        confidence_level=DEFAULT_CONFIDENCE_LEVEL,
    )
    family_metrics_path = output_dir / "family_metrics.csv"
    _write_family_metrics_csv(
        family_metrics_path,
        family_metrics,
        fieldnames=MIXED_FAMILY_METRIC_COLUMNS,
        fixed_values={
            "exposure_condition": official_run.exposure_condition,
            "filter_mode": filter_mode,
        },
    )

    pipeline_timing_samples_path = output_dir / "timing_pipeline_samples.csv"
    pipeline_timing_summary_path = output_dir / "timing_pipeline_summary.json"
    _write_pipeline_timing_artifacts(
        samples=tuple(pipeline_samples),
        samples_path=pipeline_timing_samples_path,
        summary_path=pipeline_timing_summary_path,
    )
    filter_timing_samples_path = output_dir / "timing_filter_samples.csv"
    filter_timing_summary_path = output_dir / "timing_filter_summary.json"
    _write_filter_timing_artifacts(
        samples=tuple(filter_samples),
        samples_path=filter_timing_samples_path,
        summary_path=filter_timing_summary_path,
    )

    return (
        FollowOnConditionArtifacts(
            exposure_condition=official_run.exposure_condition,
            condition=filter_mode,
            responses_path=responses_path,
            metrics_path=metrics_path,
            family_metrics_path=family_metrics_path,
            pipeline_timing_samples_path=pipeline_timing_samples_path,
            pipeline_timing_summary_path=pipeline_timing_summary_path,
            filter_timing_samples_path=filter_timing_samples_path,
            filter_timing_summary_path=filter_timing_summary_path,
        ),
        rows_tuple,
    )


def _compute_filter_decisions(
    *,
    config: FollowOnAdaptiveReplayConfig | FollowOnMixedTrafficReplayConfig,
    filter_inputs: tuple[FollowOnFilterInput, ...],
    eval_dataset: str,
    selected_filter_modes: tuple[str, ...],
    setup_entries: list[SetupTimingEntry],
) -> dict[str, dict[str, FollowOnFilterDecision]]:
    if not selected_filter_modes:
        return {}

    model_parameters_start = perf_counter()
    model_parameters = load_plaintext_model_parameters(config.plaintext_filter.model_parameters_path)
    setup_entries.append(
        SetupTimingEntry(
            component="plaintext_filter_parameters_load",
            duration_ms=elapsed_ms(model_parameters_start),
            detail=str(config.plaintext_filter.model_parameters_path),
        )
    )

    encoder_settings = EncoderSettings(
        model_name=model_parameters.encoder_model_name,
        normalize_embeddings=model_parameters.normalize_embeddings,
        batch_size=config.filter_encoder.batch_size,
        device=config.filter_encoder.device,
    )
    encoder_start = perf_counter()
    encoder, encoder_device, embedding_dimension = load_sentence_encoder(encoder_settings)
    setup_entries.append(
        SetupTimingEntry(
            component="sentence_encoder_load",
            duration_ms=elapsed_ms(encoder_start),
            detail=f"{model_parameters.encoder_model_name} on {encoder_device}",
        )
    )
    if embedding_dimension != model_parameters.embedding_dimension:
        raise ValueError(
            "Follow-on filter encoder dimension does not match saved Stage 3 model parameters."
        )
    if embedding_dimension != EXPECTED_STAGE3_EMBEDDING_DIMENSION:
        raise ValueError("Follow-on filter encoder must use the frozen 384-dimensional encoder.")

    fhe_scorer: OpenFheCkksScorer | None = None
    if "fhe_filter" in selected_filter_modes:
        fhe_start = perf_counter()
        fhe_scorer = OpenFheCkksScorer.load_or_create(
            settings=config.fhe,
            model_parameters=model_parameters,
            bundle_paths=OpenFheBundlePaths.for_root(config.fhe_filter.compiled_bundle_dir),
        )
        setup_entries.append(
            SetupTimingEntry(
                component="fhe_bundle_load_or_create",
                duration_ms=elapsed_ms(fhe_start),
                detail=(
                    f"{config.fhe_filter.compiled_bundle_dir}; "
                    f"reused_existing_bundle={fhe_scorer.reused_existing_bundle}"
                ),
            )
        )

    decisions: dict[str, dict[str, FollowOnFilterDecision]] = {
        filter_mode: {} for filter_mode in selected_filter_modes
    }
    for filter_input in filter_inputs:
        embedding_start = perf_counter()
        embedding = np.asarray(
            encoder.encode(
                [filter_input.message_text],
                batch_size=1,
                convert_to_numpy=True,
                normalize_embeddings=encoder_settings.normalize_embeddings,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        embedding_ms = elapsed_ms(embedding_start)
        _validate_filter_embedding(embedding, expected_dimension=model_parameters.embedding_dimension)
        embedding_vector = embedding[0]

        if "plaintext_filter" in selected_filter_modes:
            threshold_start = perf_counter()
            probability = float(sigmoid(compute_plaintext_logits(model_parameters, embedding))[0])
            predicted_label = int(predict_labels(model_parameters, np.asarray([probability]))[0])
            threshold_ms = elapsed_ms(threshold_start)
            decision = _label_to_filter_decision(predicted_label)
            total_filter_ms = embedding_ms + threshold_ms
            timing_sample = FilterTimingSample(
                row_id=filter_input.row_id,
                eval_dataset=eval_dataset,
                filter_mode="plaintext_filter",
                embedding_ms=embedding_ms,
                encryption_ms=None,
                fhe_scoring_ms=None,
                decryption_ms=None,
                threshold_ms=threshold_ms,
                io_ms=None,
                total_filter_ms=total_filter_ms,
            )
            decisions["plaintext_filter"][filter_input.row_id] = FollowOnFilterDecision(
                row_id=filter_input.row_id,
                block_probability=probability,
                decision=decision,
                timing_sample=timing_sample,
            )

        if "fhe_filter" in selected_filter_modes:
            if fhe_scorer is None:
                raise RuntimeError("FHE scorer was not initialized.")
            fhe_operation_start = perf_counter()
            decrypted_logit, latency = fhe_scorer.score_embedding(embedding_vector)
            fhe_operation_ms = elapsed_ms(fhe_operation_start)
            threshold_start = perf_counter()
            probability = float(sigmoid(np.asarray([decrypted_logit], dtype=np.float64))[0])
            predicted_label = int(predict_labels(model_parameters, np.asarray([probability]))[0])
            threshold_ms = elapsed_ms(threshold_start)
            total_filter_ms = embedding_ms + fhe_operation_ms + threshold_ms
            decision = _label_to_filter_decision(predicted_label)
            timing_sample = FilterTimingSample(
                row_id=filter_input.row_id,
                eval_dataset=eval_dataset,
                filter_mode="fhe_filter",
                embedding_ms=embedding_ms,
                encryption_ms=latency.encryption_ms,
                fhe_scoring_ms=latency.scoring_ms,
                decryption_ms=latency.decryption_ms,
                threshold_ms=threshold_ms,
                io_ms=None,
                total_filter_ms=total_filter_ms,
            )
            decisions["fhe_filter"][filter_input.row_id] = FollowOnFilterDecision(
                row_id=filter_input.row_id,
                block_probability=probability,
                decision=decision,
                timing_sample=timing_sample,
            )

    del encoder
    if encoder_device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return decisions


def _generate_response_texts_with_timing(
    *,
    prompts: Sequence[Any],
    row_id_getter: Any,
    message_text_getter: Any,
    prompt_condition: str,
    protocol: Any,
    tokenizer: Any,
    model: torch.nn.Module,
    batch_size: int,
    max_sequence_length: int,
    add_generation_prompt: bool,
    max_new_tokens: int,
) -> tuple[dict[str, str], dict[str, float]]:
    if not prompts:
        return {}, {}

    model.eval()
    response_texts: dict[str, str] = {}
    llm_generation_ms: dict[str, float] = {}
    device = torch.device("cuda")

    with torch.inference_mode():
        for start_index in range(0, len(prompts), batch_size):
            batch_prompts = tuple(prompts[start_index : start_index + batch_size])
            tokenized_batch = [
                tokenize_chat_messages(
                    _build_prompt_messages(
                        message_text_getter(prompt),
                        prompt_condition=prompt_condition,
                        system_prompt=protocol.stage1.system_prompt,
                    ),
                    tokenizer=tokenizer,
                    max_sequence_length=max_sequence_length,
                    add_generation_prompt=add_generation_prompt,
                )
                for prompt in batch_prompts
            ]
            batch = tokenizer.pad(
                [
                    {
                        "input_ids": list(example.input_ids),
                        "attention_mask": list(example.attention_mask),
                    }
                    for example in tokenized_batch
                ],
                padding=True,
                return_tensors="pt",
                pad_to_multiple_of=8,
            )
            batch = {
                key: value.to(device=device, non_blocking=True)
                for key, value in batch.items()
            }
            generation_start = perf_counter()
            generated = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            batch_generation_ms = elapsed_ms(generation_start)
            per_row_generation_ms = batch_generation_ms / len(batch_prompts)
            prompt_length = batch["input_ids"].shape[1]
            completion_token_ids = generated[:, prompt_length:]
            for prompt, output_ids in zip(batch_prompts, completion_token_ids, strict=True):
                row_id = row_id_getter(prompt)
                response_texts[row_id] = tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True,
                ).strip()
                llm_generation_ms[row_id] = per_row_generation_ms
    return response_texts, llm_generation_ms


def _build_prompt_messages(
    message_text: str,
    *,
    prompt_condition: str,
    system_prompt: str,
) -> tuple[ChatMessage, ...]:
    if prompt_condition == "no_system_prompt":
        return (ChatMessage(role="user", content=message_text),)
    if prompt_condition == "system_prompt_active":
        return (
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=message_text),
        )
    raise ValueError(f"Unsupported prompt condition {prompt_condition!r}.")


def _parse_common_replay_config(
    *,
    path: Path,
    document: Mapping[str, Any],
    purpose: str,
) -> dict[str, Any]:
    protocol_dir = _resolve_path(document["protocol"]["config_dir"])
    protocol = load_protocol_bundle(protocol_dir)

    model = ModelSettings(
        name=str(document["model"]["name"]),
        trust_remote_code=bool(document["model"]["trust_remote_code"]),
        attn_implementation=str(document["model"]["attn_implementation"]),
    )
    if model.name != protocol.core.primary_model:
        raise ValueError(
            f"{purpose} must target the frozen primary model {protocol.core.primary_model}, "
            f"found {model.name}."
        )

    tokenizer = TokenizerSettings(
        source=str(document["tokenizer"]["source"]),
        use_fast=bool(document["tokenizer"]["use_fast"]),
        add_generation_prompt=bool(document["tokenizer"]["add_generation_prompt"]),
        padding_side=str(document["tokenizer"]["padding_side"]),
        truncation_side=str(document["tokenizer"]["truncation_side"]),
        max_sequence_length=int(document["tokenizer"]["max_sequence_length"]),
    )
    if not tokenizer.add_generation_prompt:
        raise ValueError(f"{purpose} must set tokenizer.add_generation_prompt = true.")
    if tokenizer.max_sequence_length <= 0:
        raise ValueError("tokenizer.max_sequence_length must be positive.")

    decoding = DecodingSettings(
        temperature=float(document["decoding"]["temperature"]),
        top_p=float(document["decoding"]["top_p"]),
        max_new_tokens=int(document["decoding"]["max_new_tokens"]),
        completions_per_prompt=int(document["decoding"]["completions_per_prompt"]),
        retry_budget=int(document["decoding"]["retry_budget"]),
    )
    protocol_decoding = protocol.stage2.decoding
    if (
        decoding.temperature != protocol_decoding.temperature
        or decoding.top_p != protocol_decoding.top_p
        or decoding.max_new_tokens != protocol_decoding.max_new_tokens
        or decoding.completions_per_prompt != protocol_decoding.completions_per_prompt
        or decoding.retry_budget != protocol_decoding.retry_budget
    ):
        raise ValueError(f"{purpose} decoding settings must match the frozen Stage 2 protocol.")

    inference = InferenceSettings(
        batch_size=int(document["inference"]["batch_size"]),
        bf16=bool(document["inference"]["bf16"]),
        tf32=bool(document["inference"]["tf32"]),
    )
    if inference.batch_size <= 0:
        raise ValueError("inference.batch_size must be positive.")

    official_run_dirs = {
        str(exposure): _resolve_path(run_dir)
        for exposure, run_dir in document["inputs"]["official_run_dirs"].items()
    }
    if tuple(official_run_dirs) != EXPOSURE_CONDITIONS:
        raise ValueError(
            f"{purpose} must define official run dirs for exposure conditions "
            f"{EXPOSURE_CONDITIONS}, found {tuple(official_run_dirs)}."
        )
    official_runs = {
        exposure: _load_official_run_reference(
            exposure_condition=exposure,
            run_dir=run_dir,
            expected_base_model_name=model.name,
        )
        for exposure, run_dir in official_run_dirs.items()
    }

    filter_encoder = FilterEncoderSettings(
        batch_size=int(document["filter"]["encoder_batch_size"]),
        device=str(document["filter"]["encoder_device"]),
    )
    if filter_encoder.batch_size <= 0:
        raise ValueError("filter.encoder_batch_size must be positive.")

    plaintext_filter = PlaintextFilterReference(
        metrics_path=_resolve_existing_path(document["inputs"]["plaintext_filter"]["metrics_path"]),
        model_parameters_path=_resolve_existing_path(
            document["inputs"]["plaintext_filter"]["model_parameters_path"]
        ),
    )
    fhe_filter = FheFilterReference(
        compiled_bundle_dir=_resolve_path(document["inputs"]["fhe_filter"]["compiled_bundle_dir"]),
        compiled_bundle_manifest_path=_resolve_existing_path(
            document["inputs"]["fhe_filter"]["compiled_bundle_manifest_path"]
        ),
    )
    fhe = OpenFheSettings(
        backend=str(document["fhe"]["backend"]),
        scheme=str(document["fhe"]["scheme"]),
        ring_dimension=int(document["fhe"]["ring_dimension"]),
        multiplicative_depth=int(document["fhe"]["multiplicative_depth"]),
        scaling_mod_size=int(document["fhe"]["scaling_mod_size"]),
        first_mod_size=int(document["fhe"]["first_mod_size"]),
        batch_size=int(document["fhe"]["batch_size"]),
        security_level=str(document["fhe"]["security_level"]),
    )
    _validate_fhe_settings(fhe)

    return {
        "config_path": path,
        "protocol_config_dir": protocol_dir,
        "output_root": _resolve_path(document["outputs"]["root_dir"]),
        "timing_root": _resolve_path(document["outputs"]["timing_root_dir"]),
        "model": model,
        "tokenizer": tokenizer,
        "decoding": decoding,
        "inference": inference,
        "official_runs": official_runs,
        "filter_encoder": filter_encoder,
        "plaintext_filter": plaintext_filter,
        "fhe_filter": fhe_filter,
        "fhe": fhe,
        "seed": int(document["seed"]["value"]),
    }


def _load_tokenizer(
    config: FollowOnAdaptiveReplayConfig | FollowOnMixedTrafficReplayConfig,
    setup_entries: list[SetupTimingEntry],
) -> Any:
    start = perf_counter()
    tokenizer = load_stage1_tokenizer(
        config.tokenizer.source,
        use_fast=config.tokenizer.use_fast,
        trust_remote_code=config.model.trust_remote_code,
        padding_side=config.tokenizer.padding_side,
        truncation_side=config.tokenizer.truncation_side,
    )
    setup_entries.append(
        SetupTimingEntry(
            component="tokenizer_load",
            duration_ms=elapsed_ms(start),
            detail=config.tokenizer.source,
        )
    )
    return tokenizer


def _load_model_with_timing(
    config: FollowOnAdaptiveReplayConfig | FollowOnMixedTrafficReplayConfig,
    official_run: OfficialRunReference,
    tokenizer: Any,
    setup_entries: list[SetupTimingEntry],
) -> torch.nn.Module:
    start = perf_counter()
    model = _load_adapter_model(config, official_run, tokenizer)
    setup_entries.append(
        SetupTimingEntry(
            component="llm_adapter_model_load",
            duration_ms=elapsed_ms(start),
            detail=f"{official_run.exposure_condition}:{official_run.run_name}",
        )
    )
    return model


def _pipeline_sample(
    *,
    row_id: str,
    exposure_condition: str,
    eval_dataset: str,
    condition: str,
    filter_mode: str | None,
    source_type: str | None,
    filter_decision: str | None,
    response_generated: int,
    filter_total_ms: float | None,
    llm_generation_ms: float | None,
    routing_ms: float | None,
) -> PipelineTimingSample:
    total_pipeline_ms = (
        (filter_total_ms or 0.0)
        + (llm_generation_ms or 0.0)
        + (routing_ms or 0.0)
    )
    return PipelineTimingSample(
        row_id=row_id,
        exposure_condition=exposure_condition,
        eval_dataset=eval_dataset,
        condition=condition,
        filter_mode=filter_mode,
        source_type=source_type,
        filter_decision=filter_decision,
        response_generated=response_generated,
        filter_total_ms=filter_total_ms,
        llm_generation_ms=llm_generation_ms,
        routing_ms=routing_ms,
        total_pipeline_ms=total_pipeline_ms,
    )


def _leakage_to_flags(leakage: Any) -> dict[str, int]:
    return {
        "name_leaked": int(leakage.name_leaked),
        "date_of_birth_leaked": int(leakage.date_of_birth_leaked),
        "account_number_leaked": int(leakage.account_number_leaked),
        "any_sensitive_field_leaked": int(leakage.any_sensitive_field_leaked),
        "full_canary_record_leaked": int(leakage.full_canary_record_leaked),
        "other_canary_leak": int(leakage.other_canary_leak),
        "refusal_emitted": int(leakage.refusal_emitted),
    }


def _zero_leakage_flags(*, refusal_emitted: int = 0) -> dict[str, int]:
    return {
        "name_leaked": 0,
        "date_of_birth_leaked": 0,
        "account_number_leaked": 0,
        "any_sensitive_field_leaked": 0,
        "full_canary_record_leaked": 0,
        "other_canary_leak": 0,
        "refusal_emitted": int(refusal_emitted),
    }


def _label_to_filter_decision(predicted_label: int) -> str:
    decision = INT_TO_LABEL[int(predicted_label)]
    if decision not in {FILTER_ALLOW, FILTER_BLOCK}:
        raise ValueError(f"Unexpected filter decision label {decision!r}.")
    return decision


def _validate_filter_embedding(embedding: np.ndarray, *, expected_dimension: int) -> None:
    if embedding.ndim != 2 or embedding.shape[0] != 1:
        raise ValueError("Follow-on filter embedding must have shape (1, embedding_dimension).")
    if embedding.shape[1] != expected_dimension:
        raise ValueError(
            f"Follow-on filter embedding dimension {embedding.shape[1]} does not match "
            f"expected dimension {expected_dimension}."
        )


def _write_response_jsonl(path: Path, rows: Iterable[Any], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            serialized = row.to_row()
            if tuple(serialized.keys()) != tuple(columns):
                raise ValueError(
                    f"Follow-on response row must match schema {tuple(columns)}, "
                    f"found {tuple(serialized.keys())}."
                )
            handle.write(json.dumps(serialized, ensure_ascii=True, separators=(",", ":"), allow_nan=False))
            handle.write("\n")


def _write_family_metrics_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
    fixed_values: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output_row = dict(fixed_values)
            output_row.update({key: row.get(key) for key in fieldnames if key not in output_row})
            writer.writerow({key: _format_csv_value(output_row.get(key)) for key in fieldnames})


def _write_pipeline_timing_artifacts(
    *,
    samples: tuple[PipelineTimingSample, ...],
    samples_path: Path,
    summary_path: Path,
) -> None:
    _write_timing_csv(samples_path, samples, PIPELINE_TIMING_COLUMNS)
    _write_json(
        summary_path,
        {
            "timing_unit": "milliseconds",
            "blocked_message_llm_generation_ms": 0.0,
            "numeric_columns": list(DEFAULT_PIPELINE_TIMING_NUMERIC_COLUMNS),
            "summary": summarize_pipeline_timing_samples(samples),
        },
    )


def _write_filter_timing_artifacts(
    *,
    samples: tuple[FilterTimingSample, ...],
    samples_path: Path,
    summary_path: Path,
) -> None:
    _write_timing_csv(samples_path, samples, FILTER_TIMING_COLUMNS)
    _write_json(
        summary_path,
        {
            "timing_unit": "milliseconds",
            "numeric_columns": list(DEFAULT_FILTER_TIMING_NUMERIC_COLUMNS),
            "summary": summarize_filter_timing_samples(samples),
        },
    )


def _write_timing_csv(path: Path, rows: Sequence[Any], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = row.to_row()
            writer.writerow({field: _format_csv_value(serialized.get(field)) for field in fieldnames})


def _write_setup_timing_json(
    timing_root: Path,
    entries: Sequence[SetupTimingEntry],
    *,
    eval_dataset: str,
) -> Path:
    timing_root.mkdir(parents=True, exist_ok=True)
    aggregate_path = timing_root / "setup_timing.json"
    manifest_path = timing_root / "setup_timing_manifest.json"
    per_sweep_path = timing_root / f"setup_timing_{eval_dataset}.json"
    rows = [entry.to_row() for entry in entries]
    per_sweep_payload = {
        "eval_dataset": eval_dataset,
        "timing_unit": "milliseconds",
        "columns": list(SETUP_TIMING_COLUMNS),
        "entries": rows,
        "summary": summarize_timing_rows(rows, numeric_columns=("duration_ms",)),
    }
    _write_json(
        per_sweep_path,
        per_sweep_payload,
    )

    sweeps = _load_setup_timing_manifest_sweeps(manifest_path)
    sweeps[eval_dataset] = {
        "eval_dataset": eval_dataset,
        "setup_timing_path": str(per_sweep_path),
        "entry_count": len(rows),
        "summary": per_sweep_payload["summary"],
    }
    ordered_sweeps = _order_setup_timing_sweeps(sweeps)
    _write_json(
        manifest_path,
        {
            "timing_unit": "milliseconds",
            "aggregate_setup_timing_path": str(aggregate_path),
            "sweeps": ordered_sweeps,
        },
    )

    aggregate_rows = _load_aggregate_setup_timing_rows(ordered_sweeps)
    _write_json(
        aggregate_path,
        {
            "timing_unit": "milliseconds",
            "columns": ["eval_dataset", *SETUP_TIMING_COLUMNS],
            "manifest_path": str(manifest_path),
            "per_sweep_setup_timing_paths": {
                sweep["eval_dataset"]: sweep["setup_timing_path"] for sweep in ordered_sweeps
            },
            "entries": aggregate_rows,
            "summary": summarize_timing_rows(aggregate_rows, numeric_columns=("duration_ms",)),
            "sweeps": ordered_sweeps,
        },
    )
    return aggregate_path


def _load_setup_timing_manifest_sweeps(
    manifest_path: Path,
) -> dict[str, dict[str, Any]]:
    if not manifest_path.exists():
        return {}
    document = _read_json(manifest_path)
    sweeps: dict[str, dict[str, Any]] = {}
    for sweep in document.get("sweeps", []):
        eval_dataset = str(sweep.get("eval_dataset", ""))
        setup_timing_path = str(sweep.get("setup_timing_path", ""))
        if not eval_dataset or not setup_timing_path:
            continue
        sweeps[eval_dataset] = dict(sweep)
    return sweeps


def _order_setup_timing_sweeps(
    sweeps: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    preferred_order = (ADAPTIVE_EVAL_DATASET, MIXED_EVAL_DATASET)
    ordered_keys = [key for key in preferred_order if key in sweeps]
    ordered_keys.extend(sorted(key for key in sweeps if key not in preferred_order))
    return [dict(sweeps[key]) for key in ordered_keys]


def _load_aggregate_setup_timing_rows(
    sweeps: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    aggregate_rows: list[dict[str, Any]] = []
    for sweep in sweeps:
        eval_dataset = str(sweep["eval_dataset"])
        per_sweep_path = Path(str(sweep["setup_timing_path"]))
        if not per_sweep_path.exists():
            continue
        per_sweep_payload = _read_json(per_sweep_path)
        for row in per_sweep_payload.get("entries", []):
            aggregate_rows.append({"eval_dataset": eval_dataset, **dict(row)})
    return aggregate_rows


def _write_adaptive_summary_json(
    *,
    summary_path: Path,
    output_root: Path,
    exposure_order: tuple[str, ...],
    conditions: tuple[str, ...],
) -> None:
    runs = []
    for exposure_condition in exposure_order:
        for condition in conditions:
            run_dir = output_root / exposure_condition / condition
            metrics_path = run_dir / "adaptive_metrics.json"
            family_metrics_path = run_dir / "family_metrics.csv"
            responses_path = run_dir / "adaptive_responses.jsonl"
            pipeline_summary_path = run_dir / "timing_pipeline_summary.json"
            if not metrics_path.exists():
                continue
            metrics = _read_json(metrics_path)
            runs.append(
                {
                    "exposure_condition": exposure_condition,
                    "condition": condition,
                    "run_name": metrics["run_name"],
                    "responses_path": str(responses_path),
                    "metrics_path": str(metrics_path),
                    "family_metrics_path": str(family_metrics_path),
                    "timing_pipeline_summary_path": str(pipeline_summary_path),
                    "any_sensitive_field_leak_rate": metrics["any_sensitive_field_leak_rate"],
                    "full_canary_record_leak_rate": metrics["full_canary_record_leak_rate"],
                    "other_canary_leak_rate": metrics["other_canary_leak_rate"],
                    "refusal_rate": metrics["refusal_rate"],
                    "adversarial_block_rate": metrics.get("adversarial_block_rate"),
                    "filter_allow_rate": metrics.get("filter_allow_rate"),
                    "leak_rate_given_allowed": metrics.get("leak_rate_given_allowed"),
                }
            )
    _write_json(summary_path, {"runs": runs})


def _write_mixed_summary_json(
    *,
    summary_path: Path,
    output_root: Path,
    exposure_order: tuple[str, ...],
    filter_modes: tuple[str, ...],
) -> None:
    runs = []
    for exposure_condition in exposure_order:
        for filter_mode in filter_modes:
            run_dir = output_root / exposure_condition / filter_mode
            metrics_path = run_dir / "mixed_traffic_metrics.json"
            family_metrics_path = run_dir / "family_metrics.csv"
            responses_path = run_dir / "mixed_traffic_responses.jsonl"
            pipeline_summary_path = run_dir / "timing_pipeline_summary.json"
            if not metrics_path.exists():
                continue
            metrics = _read_json(metrics_path)
            runs.append(
                {
                    "exposure_condition": exposure_condition,
                    "filter_mode": filter_mode,
                    "run_name": metrics["run_name"],
                    "responses_path": str(responses_path),
                    "metrics_path": str(metrics_path),
                    "family_metrics_path": str(family_metrics_path),
                    "timing_pipeline_summary_path": str(pipeline_summary_path),
                    "adversarial_block_rate": metrics["adversarial_block_rate"],
                    "benign_false_positive_rate": metrics["benign_false_positive_rate"],
                    "benign_allow_rate": metrics["benign_allow_rate"],
                    "adaptive_any_sensitive_field_leak_rate": metrics[
                        "adaptive_any_sensitive_field_leak_rate"
                    ],
                    "adaptive_full_canary_record_leak_rate": metrics[
                        "adaptive_full_canary_record_leak_rate"
                    ],
                    "leak_rate_given_allowed": metrics["leak_rate_given_allowed"],
                }
            )
    _write_json(summary_path, {"runs": runs})


def _write_ci_summary_json(
    *,
    summary_path: Path,
    output_root: Path,
    exposure_order: tuple[str, ...],
    run_axis_name: str,
    run_axis_values: tuple[str, ...],
    metrics_filename: str,
) -> None:
    runs = []
    for exposure_condition in exposure_order:
        for run_axis_value in run_axis_values:
            metrics_path = output_root / exposure_condition / run_axis_value / metrics_filename
            if not metrics_path.exists():
                continue
            metrics = _read_json(metrics_path)
            ci_metrics = {
                key: value
                for key, value in metrics.items()
                if key.endswith("_ci")
            }
            runs.append(
                {
                    "exposure_condition": exposure_condition,
                    run_axis_name: run_axis_value,
                    "metrics_path": str(metrics_path),
                    "confidence_intervals": ci_metrics,
                }
            )
    _write_json(
        summary_path,
        {
            "method": "wilson",
            "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
            "runs": runs,
        },
    )


def _write_filter_parity_summary_json(
    *,
    summary_path: Path,
    output_root: Path,
    exposure_order: tuple[str, ...],
    rows_by_condition: Mapping[tuple[str, str], Sequence[Any]],
    row_id_field: str,
    metrics_filename: str,
) -> None:
    runs = []
    for exposure_condition in exposure_order:
        plaintext_rows = rows_by_condition.get((exposure_condition, "plaintext_filter"))
        fhe_rows = rows_by_condition.get((exposure_condition, "fhe_filter"))
        if plaintext_rows is None or fhe_rows is None:
            continue
        parity_metrics = build_filter_parity_metrics(
            plaintext_rows,
            fhe_rows,
            row_id_field=row_id_field,
        )
        runs.append(
            {
                "exposure_condition": exposure_condition,
                "plaintext_metrics_path": str(
                    output_root / exposure_condition / "plaintext_filter" / metrics_filename
                ),
                "fhe_metrics_path": str(
                    output_root / exposure_condition / "fhe_filter" / metrics_filename
                ),
                **parity_metrics,
            }
        )
    _write_json(summary_path, {"runs": runs})


def _build_decoding_payload(
    config: FollowOnAdaptiveReplayConfig | FollowOnMixedTrafficReplayConfig,
) -> dict[str, Any]:
    return {
        "temperature": config.decoding.temperature,
        "top_p": config.decoding.top_p,
        "max_new_tokens": config.decoding.max_new_tokens,
        "completions_per_prompt": config.decoding.completions_per_prompt,
        "retry_budget": config.decoding.retry_budget,
        "do_sample": False,
    }


def _prepare_output_root(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".16g")
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return value


def _resolve_path(value: Path | str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _resolve_existing_path(value: Path | str) -> Path:
    path = _resolve_path(value)
    if not path.exists():
        raise FileNotFoundError(f"Expected path is missing: {path}")
    return path
