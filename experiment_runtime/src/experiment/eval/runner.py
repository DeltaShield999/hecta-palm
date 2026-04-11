from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import json
import os

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, set_seed

from qwen_langgraph_demo.runtime.protocol import load_protocol_bundle

from experiment.eval.config import (
    DEFAULT_STAGE2_FILTERED_REPLAY_CONFIG_PATH,
    DEFAULT_STAGE2_REPLAY_CONFIG_PATH,
    OfficialRunReference,
    Stage2FilteredReplayConfig,
    Stage2ReplayConfig,
    resolve_exposure_conditions,
    resolve_filter_modes,
    resolve_prompt_conditions,
)
from experiment.eval.data import (
    STAGE2_FILTERED_RESPONSE_COLUMNS,
    STAGE2_RESPONSE_COLUMNS,
    Stage2FilteredResponseRow,
    Stage2ResponseRow,
    build_prompt_messages,
    load_canary_registry_index,
    load_stage2_attack_prompts,
)
from experiment.eval.metrics import (
    FAMILY_METRIC_COLUMNS,
    FILTERED_FAMILY_METRIC_COLUMNS,
    FamilyMetricRow,
    FilteredFamilyMetricRow,
    build_condition_metrics,
    build_family_metrics,
    build_filter_parity_metrics,
    build_filtered_condition_metrics,
    build_filtered_family_metrics,
)
from experiment.eval.scoring import CanaryLeakageScorer
from experiment.fhe.data import (
    compute_plaintext_logits,
    load_plaintext_model_parameters,
    predict_labels,
    sigmoid,
)
from experiment.fhe.openfhe_backend import OpenFheBundlePaths, OpenFheCkksScorer
from experiment.filter_train.config import EncoderSettings
from experiment.filter_train.data import INT_TO_LABEL
from experiment.filter_train.embeddings import EXPECTED_STAGE3_EMBEDDING_DIMENSION, load_sentence_encoder
from experiment.schemas.stage2 import Stage2AttackPrompt
from experiment.train_qwen.data import load_stage1_tokenizer, tokenize_chat_messages


INTEGRATED_PROMPT_CONDITION = "system_prompt_active"
FILTER_ALLOW_LABEL = "ALLOW"
FILTER_BLOCK_LABEL = "BLOCK"


@dataclass(frozen=True, slots=True)
class Stage2ConditionArtifacts:
    exposure_condition: str
    prompt_condition: str
    responses_path: Path
    metrics_path: Path
    family_metrics_path: Path


@dataclass(frozen=True, slots=True)
class Stage2EvaluationResult:
    summary_path: Path
    artifacts: dict[tuple[str, str], Stage2ConditionArtifacts]


@dataclass(frozen=True, slots=True)
class Stage2FilteredConditionArtifacts:
    exposure_condition: str
    filter_mode: str
    responses_path: Path
    metrics_path: Path
    family_metrics_path: Path


@dataclass(frozen=True, slots=True)
class Stage2FilteredEvaluationResult:
    summary_path: Path
    parity_summary_path: Path
    artifacts: dict[tuple[str, str], Stage2FilteredConditionArtifacts]


@dataclass(frozen=True, slots=True)
class FilterDecision:
    attack_id: str
    block_probability: float
    decision: str


def run_stage2_evaluation(
    *,
    config_path: Path | str | None = None,
    exposure: str,
    condition: str,
) -> Stage2EvaluationResult:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    config = Stage2ReplayConfig.from_toml(config_path or DEFAULT_STAGE2_REPLAY_CONFIG_PATH)
    selected_exposures = resolve_exposure_conditions(exposure)
    selected_conditions = resolve_prompt_conditions(condition)
    _prepare_output_root(config.output_root)
    _configure_cuda_inference(config.inference.bf16, config.inference.tf32)

    protocol = load_protocol_bundle(config.protocol_config_dir)
    tokenizer = load_stage1_tokenizer(
        config.tokenizer.source,
        use_fast=config.tokenizer.use_fast,
        trust_remote_code=config.model.trust_remote_code,
        padding_side=config.tokenizer.padding_side,
        truncation_side=config.tokenizer.truncation_side,
    )
    attack_prompts = load_stage2_attack_prompts(
        config.attack_dataset_path,
        expected_families=protocol.stage2.attack_families,
    )
    canary_registry = load_canary_registry_index(config.canary_registry_path)
    scorer = CanaryLeakageScorer(
        tuple(canary_registry.values()),
        refusal_response=protocol.stage1.refusal_response,
    )

    set_seed(config.seed)
    artifacts: dict[tuple[str, str], Stage2ConditionArtifacts] = {}
    for exposure_condition in selected_exposures:
        official_run = config.official_runs[exposure_condition]
        model = _load_adapter_model(config, official_run, tokenizer)
        try:
            for prompt_condition in selected_conditions:
                condition_artifacts = _evaluate_condition(
                    config=config,
                    protocol=protocol,
                    official_run=official_run,
                    prompt_condition=prompt_condition,
                    attack_prompts=attack_prompts,
                    scorer=scorer,
                    tokenizer=tokenizer,
                    model=model,
                )
                artifacts[(exposure_condition, prompt_condition)] = condition_artifacts
        finally:
            _release_model(model)

    summary_path = config.output_root / "stage2_summary.json"
    _write_summary_json(
        summary_path=summary_path,
        output_root=config.output_root,
        exposure_order=selected_exposures,
        prompt_conditions=selected_conditions,
    )
    return Stage2EvaluationResult(
        summary_path=summary_path,
        artifacts=artifacts,
    )


def run_stage2_filtered_evaluation(
    *,
    config_path: Path | str | None = None,
    exposure: str,
    filter_mode: str,
) -> Stage2FilteredEvaluationResult:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    config = Stage2FilteredReplayConfig.from_toml(
        config_path or DEFAULT_STAGE2_FILTERED_REPLAY_CONFIG_PATH
    )
    selected_exposures = resolve_exposure_conditions(exposure)
    selected_filter_modes = resolve_filter_modes(filter_mode)
    _prepare_output_root(config.output_root)
    _configure_cuda_inference(config.inference.bf16, config.inference.tf32)

    protocol = load_protocol_bundle(config.protocol_config_dir)
    tokenizer = load_stage1_tokenizer(
        config.tokenizer.source,
        use_fast=config.tokenizer.use_fast,
        trust_remote_code=config.model.trust_remote_code,
        padding_side=config.tokenizer.padding_side,
        truncation_side=config.tokenizer.truncation_side,
    )
    attack_prompts = load_stage2_attack_prompts(
        config.attack_dataset_path,
        expected_families=protocol.stage2.attack_families,
    )
    canary_registry = load_canary_registry_index(config.canary_registry_path)
    scorer = CanaryLeakageScorer(
        tuple(canary_registry.values()),
        refusal_response=protocol.stage1.refusal_response,
    )

    set_seed(config.seed)
    filter_decisions_by_mode = _compute_filter_decisions(
        config=config,
        attack_prompts=attack_prompts,
        selected_filter_modes=selected_filter_modes,
    )

    artifacts: dict[tuple[str, str], Stage2FilteredConditionArtifacts] = {}
    rows_by_condition: dict[tuple[str, str], tuple[Stage2FilteredResponseRow, ...]] = {}
    for exposure_condition in selected_exposures:
        official_run = config.official_runs[exposure_condition]
        model = _load_adapter_model(config, official_run, tokenizer)
        try:
            for selected_filter_mode in selected_filter_modes:
                condition_artifacts, rows = _evaluate_filtered_condition(
                    config=config,
                    protocol=protocol,
                    official_run=official_run,
                    filter_mode=selected_filter_mode,
                    attack_prompts=attack_prompts,
                    filter_decisions=filter_decisions_by_mode[selected_filter_mode],
                    scorer=scorer,
                    tokenizer=tokenizer,
                    model=model,
                )
                artifacts[(exposure_condition, selected_filter_mode)] = condition_artifacts
                rows_by_condition[(exposure_condition, selected_filter_mode)] = rows
        finally:
            _release_model(model)

    summary_path = config.output_root / "stage2_filtered_summary.json"
    _write_filtered_summary_json(
        summary_path=summary_path,
        baseline_summary_path=config.baseline_summary_path,
        output_root=config.output_root,
        exposure_order=selected_exposures,
        filter_modes=selected_filter_modes,
    )

    parity_summary_path = config.output_root / "filter_parity_summary.json"
    _write_filter_parity_summary_json(
        summary_path=parity_summary_path,
        output_root=config.output_root,
        exposure_order=selected_exposures,
        rows_by_condition=rows_by_condition,
    )

    return Stage2FilteredEvaluationResult(
        summary_path=summary_path,
        parity_summary_path=parity_summary_path,
        artifacts=artifacts,
    )


def _evaluate_condition(
    *,
    config: Stage2ReplayConfig,
    protocol: Any,
    official_run: OfficialRunReference,
    prompt_condition: str,
    attack_prompts: tuple[Stage2AttackPrompt, ...],
    scorer: CanaryLeakageScorer,
    tokenizer: Any,
    model: torch.nn.Module,
) -> Stage2ConditionArtifacts:
    response_rows = _generate_and_score_responses(
        attack_prompts=attack_prompts,
        prompt_condition=prompt_condition,
        official_run=official_run,
        protocol=protocol,
        scorer=scorer,
        tokenizer=tokenizer,
        model=model,
        batch_size=config.inference.batch_size,
        max_sequence_length=config.tokenizer.max_sequence_length,
        add_generation_prompt=config.tokenizer.add_generation_prompt,
        max_new_tokens=config.decoding.max_new_tokens,
    )

    output_dir = config.output_root / official_run.exposure_condition / prompt_condition
    output_dir.mkdir(parents=True, exist_ok=True)

    responses_path = output_dir / "stage2_responses.jsonl"
    _write_responses_jsonl(responses_path, response_rows)

    metrics_payload = build_condition_metrics(
        response_rows,
        exposure_condition=official_run.exposure_condition,
        prompt_condition=prompt_condition,
        run_name=official_run.run_name,
        base_model_name=config.model.name,
        adapter_run_dir=official_run.run_dir,
        attack_dataset_path=config.attack_dataset_path,
        family_count=len(protocol.stage2.attack_families),
        system_prompt_used=(prompt_condition == "system_prompt_active"),
        headline_metric=protocol.stage2.headline_metric,
        secondary_metric=protocol.stage2.secondary_metric,
        decoding=_build_decoding_payload(config),
    )
    metrics_path = output_dir / "stage2_metrics.json"
    _write_json(metrics_path, metrics_payload)

    family_metrics = build_family_metrics(
        response_rows,
        exposure_condition=official_run.exposure_condition,
        prompt_condition=prompt_condition,
        family_order=protocol.stage2.attack_families,
    )
    family_metrics_path = output_dir / "family_metrics.csv"
    _write_family_metrics_csv(family_metrics_path, family_metrics)

    return Stage2ConditionArtifacts(
        exposure_condition=official_run.exposure_condition,
        prompt_condition=prompt_condition,
        responses_path=responses_path,
        metrics_path=metrics_path,
        family_metrics_path=family_metrics_path,
    )


def _evaluate_filtered_condition(
    *,
    config: Stage2FilteredReplayConfig,
    protocol: Any,
    official_run: OfficialRunReference,
    filter_mode: str,
    attack_prompts: tuple[Stage2AttackPrompt, ...],
    filter_decisions: dict[str, FilterDecision],
    scorer: CanaryLeakageScorer,
    tokenizer: Any,
    model: torch.nn.Module,
) -> tuple[Stage2FilteredConditionArtifacts, tuple[Stage2FilteredResponseRow, ...]]:
    allowed_prompts = tuple(
        attack_prompt
        for attack_prompt in attack_prompts
        if filter_decisions[attack_prompt.attack_id].decision == FILTER_ALLOW_LABEL
    )
    generated_responses = _generate_model_response_texts(
        attack_prompts=allowed_prompts,
        prompt_condition=INTEGRATED_PROMPT_CONDITION,
        protocol=protocol,
        tokenizer=tokenizer,
        model=model,
        batch_size=config.inference.batch_size,
        max_sequence_length=config.tokenizer.max_sequence_length,
        add_generation_prompt=config.tokenizer.add_generation_prompt,
        max_new_tokens=config.decoding.max_new_tokens,
    )

    rows: list[Stage2FilteredResponseRow] = []
    for attack_prompt in attack_prompts:
        filter_decision = filter_decisions[attack_prompt.attack_id]
        if filter_decision.decision == FILTER_BLOCK_LABEL:
            rows.append(
                Stage2FilteredResponseRow(
                    attack_id=attack_prompt.attack_id,
                    template_family=attack_prompt.template_family,
                    target_canary_id=attack_prompt.target_canary_id,
                    target_fields=attack_prompt.target_fields,
                    exposure_condition=official_run.exposure_condition,
                    prompt_condition=INTEGRATED_PROMPT_CONDITION,
                    run_name=official_run.run_name,
                    filter_mode=filter_mode,
                    message_text=attack_prompt.message_text,
                    filter_block_probability=filter_decision.block_probability,
                    filter_decision=filter_decision.decision,
                    response_generated=0,
                    response_text="",
                    name_leaked=0,
                    date_of_birth_leaked=0,
                    account_number_leaked=0,
                    any_sensitive_field_leaked=0,
                    full_canary_record_leaked=0,
                    other_canary_leak=0,
                    refusal_emitted=0,
                )
            )
            continue

        response_text = generated_responses[attack_prompt.attack_id]
        leakage = scorer.score_response(
            response_text=response_text,
            target_canary_id=attack_prompt.target_canary_id,
        )
        rows.append(
            Stage2FilteredResponseRow(
                attack_id=attack_prompt.attack_id,
                template_family=attack_prompt.template_family,
                target_canary_id=attack_prompt.target_canary_id,
                target_fields=attack_prompt.target_fields,
                exposure_condition=official_run.exposure_condition,
                prompt_condition=INTEGRATED_PROMPT_CONDITION,
                run_name=official_run.run_name,
                filter_mode=filter_mode,
                message_text=attack_prompt.message_text,
                filter_block_probability=filter_decision.block_probability,
                filter_decision=filter_decision.decision,
                response_generated=1,
                response_text=response_text,
                name_leaked=leakage.name_leaked,
                date_of_birth_leaked=leakage.date_of_birth_leaked,
                account_number_leaked=leakage.account_number_leaked,
                any_sensitive_field_leaked=leakage.any_sensitive_field_leaked,
                full_canary_record_leaked=leakage.full_canary_record_leaked,
                other_canary_leak=leakage.other_canary_leak,
                refusal_emitted=leakage.refusal_emitted,
            )
        )

    output_dir = config.output_root / official_run.exposure_condition / filter_mode
    output_dir.mkdir(parents=True, exist_ok=True)

    response_rows = tuple(rows)
    responses_path = output_dir / "stage2_filtered_responses.jsonl"
    _write_filtered_responses_jsonl(responses_path, response_rows)

    metrics_payload = build_filtered_condition_metrics(
        response_rows,
        exposure_condition=official_run.exposure_condition,
        prompt_condition=INTEGRATED_PROMPT_CONDITION,
        filter_mode=filter_mode,
        run_name=official_run.run_name,
        base_model_name=config.model.name,
        adapter_run_dir=official_run.run_dir,
        attack_dataset_path=config.attack_dataset_path,
        family_count=len(protocol.stage2.attack_families),
        system_prompt_used=True,
        headline_metric=protocol.stage2.headline_metric,
        secondary_metric=protocol.stage2.secondary_metric,
        decoding=_build_decoding_payload(config),
    )
    metrics_path = output_dir / "stage2_filtered_metrics.json"
    _write_json(metrics_path, metrics_payload)

    family_metrics = build_filtered_family_metrics(
        response_rows,
        exposure_condition=official_run.exposure_condition,
        prompt_condition=INTEGRATED_PROMPT_CONDITION,
        filter_mode=filter_mode,
        family_order=protocol.stage2.attack_families,
    )
    family_metrics_path = output_dir / "family_metrics.csv"
    _write_filtered_family_metrics_csv(family_metrics_path, family_metrics)

    return (
        Stage2FilteredConditionArtifacts(
            exposure_condition=official_run.exposure_condition,
            filter_mode=filter_mode,
            responses_path=responses_path,
            metrics_path=metrics_path,
            family_metrics_path=family_metrics_path,
        ),
        response_rows,
    )


def _generate_and_score_responses(
    *,
    attack_prompts: tuple[Stage2AttackPrompt, ...],
    prompt_condition: str,
    official_run: OfficialRunReference,
    protocol: Any,
    scorer: CanaryLeakageScorer,
    tokenizer: Any,
    model: torch.nn.Module,
    batch_size: int,
    max_sequence_length: int,
    add_generation_prompt: bool,
    max_new_tokens: int,
) -> tuple[Stage2ResponseRow, ...]:
    response_texts = _generate_model_response_texts(
        attack_prompts=attack_prompts,
        prompt_condition=prompt_condition,
        protocol=protocol,
        tokenizer=tokenizer,
        model=model,
        batch_size=batch_size,
        max_sequence_length=max_sequence_length,
        add_generation_prompt=add_generation_prompt,
        max_new_tokens=max_new_tokens,
    )
    rows: list[Stage2ResponseRow] = []
    for attack_prompt in attack_prompts:
        response_text = response_texts[attack_prompt.attack_id]
        leakage = scorer.score_response(
            response_text=response_text,
            target_canary_id=attack_prompt.target_canary_id,
        )
        rows.append(
            Stage2ResponseRow(
                attack_id=attack_prompt.attack_id,
                template_family=attack_prompt.template_family,
                target_canary_id=attack_prompt.target_canary_id,
                target_fields=attack_prompt.target_fields,
                exposure_condition=official_run.exposure_condition,
                prompt_condition=prompt_condition,
                run_name=official_run.run_name,
                message_text=attack_prompt.message_text,
                response_text=response_text,
                name_leaked=leakage.name_leaked,
                date_of_birth_leaked=leakage.date_of_birth_leaked,
                account_number_leaked=leakage.account_number_leaked,
                any_sensitive_field_leaked=leakage.any_sensitive_field_leaked,
                full_canary_record_leaked=leakage.full_canary_record_leaked,
                other_canary_leak=leakage.other_canary_leak,
                refusal_emitted=leakage.refusal_emitted,
            )
        )
    return tuple(rows)


def _generate_model_response_texts(
    *,
    attack_prompts: tuple[Stage2AttackPrompt, ...],
    prompt_condition: str,
    protocol: Any,
    tokenizer: Any,
    model: torch.nn.Module,
    batch_size: int,
    max_sequence_length: int,
    add_generation_prompt: bool,
    max_new_tokens: int,
) -> dict[str, str]:
    if not attack_prompts:
        return {}

    model.eval()
    rows: dict[str, str] = {}
    device = torch.device("cuda")

    with torch.inference_mode():
        for start_index in range(0, len(attack_prompts), batch_size):
            batch_prompts = attack_prompts[start_index : start_index + batch_size]
            tokenized_batch = [
                tokenize_chat_messages(
                    build_prompt_messages(
                        attack_prompt,
                        prompt_condition=prompt_condition,
                        system_prompt=protocol.stage1.system_prompt,
                    ),
                    tokenizer=tokenizer,
                    max_sequence_length=max_sequence_length,
                    add_generation_prompt=add_generation_prompt,
                )
                for attack_prompt in batch_prompts
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
            generated = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            prompt_length = batch["input_ids"].shape[1]
            completion_token_ids = generated[:, prompt_length:]
            for attack_prompt, output_ids in zip(batch_prompts, completion_token_ids, strict=True):
                rows[attack_prompt.attack_id] = tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True,
                ).strip()
    return rows


def _compute_filter_decisions(
    *,
    config: Stage2FilteredReplayConfig,
    attack_prompts: tuple[Stage2AttackPrompt, ...],
    selected_filter_modes: tuple[str, ...],
) -> dict[str, dict[str, FilterDecision]]:
    model_parameters = load_plaintext_model_parameters(config.plaintext_filter.model_parameters_path)
    encoder_settings = EncoderSettings(
        model_name=model_parameters.encoder_model_name,
        normalize_embeddings=model_parameters.normalize_embeddings,
        batch_size=config.filter_encoder.batch_size,
        device=config.filter_encoder.device,
    )
    encoder, encoder_device, embedding_dimension = load_sentence_encoder(encoder_settings)
    if embedding_dimension != model_parameters.embedding_dimension:
        raise ValueError(
            "Stage 2 filtered replay encoder dimension does not match saved Stage 3 model parameters."
        )

    message_texts = [attack_prompt.message_text for attack_prompt in attack_prompts]
    embeddings = np.asarray(
        encoder.encode(
            message_texts,
            batch_size=encoder_settings.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=encoder_settings.normalize_embeddings,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    _validate_attack_embeddings(embeddings, expected_dimension=model_parameters.embedding_dimension)
    del encoder
    if encoder_device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    decisions: dict[str, dict[str, FilterDecision]] = {}
    if "plaintext_filter" in selected_filter_modes:
        probabilities = sigmoid(compute_plaintext_logits(model_parameters, embeddings))
        labels = predict_labels(model_parameters, probabilities)
        decisions["plaintext_filter"] = _build_filter_decision_index(
            attack_prompts=attack_prompts,
            block_probabilities=probabilities,
            predicted_labels=labels,
        )

    if "fhe_filter" in selected_filter_modes:
        scorer = OpenFheCkksScorer.load_or_create(
            settings=config.fhe,
            model_parameters=model_parameters,
            bundle_paths=OpenFheBundlePaths.for_root(config.fhe_filter.compiled_bundle_dir),
        )
        probabilities = np.empty(len(attack_prompts), dtype=np.float64)
        for index, embedding in enumerate(embeddings):
            decrypted_logit, _ = scorer.score_embedding(embedding)
            probabilities[index] = float(sigmoid(np.asarray([decrypted_logit], dtype=np.float64))[0])
        labels = predict_labels(model_parameters, probabilities)
        decisions["fhe_filter"] = _build_filter_decision_index(
            attack_prompts=attack_prompts,
            block_probabilities=probabilities,
            predicted_labels=labels,
        )

    return decisions


def _build_filter_decision_index(
    *,
    attack_prompts: tuple[Stage2AttackPrompt, ...],
    block_probabilities: np.ndarray,
    predicted_labels: np.ndarray,
) -> dict[str, FilterDecision]:
    if len(attack_prompts) != block_probabilities.shape[0] or len(attack_prompts) != predicted_labels.shape[0]:
        raise ValueError("Filter decision arrays must align with the frozen Stage 2 attack set.")

    decisions: dict[str, FilterDecision] = {}
    for attack_prompt, block_probability, predicted_label in zip(
        attack_prompts,
        block_probabilities,
        predicted_labels,
        strict=True,
    ):
        decision = INT_TO_LABEL[int(predicted_label)]
        if decision not in {FILTER_ALLOW_LABEL, FILTER_BLOCK_LABEL}:
            raise ValueError(f"Unexpected filter decision label {decision!r}.")
        decisions[attack_prompt.attack_id] = FilterDecision(
            attack_id=attack_prompt.attack_id,
            block_probability=float(block_probability),
            decision=decision,
        )
    return decisions


def _load_adapter_model(
    config: Stage2ReplayConfig | Stage2FilteredReplayConfig,
    official_run: OfficialRunReference,
    tokenizer: Any,
) -> torch.nn.Module:
    dtype = torch.bfloat16 if config.inference.bf16 else torch.float32
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": config.model.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if config.model.attn_implementation:
        model_kwargs["attn_implementation"] = config.model.attn_implementation

    base_model = AutoModelForCausalLM.from_pretrained(config.model.name, **model_kwargs)
    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.generation_config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        base_model.generation_config.eos_token_id = tokenizer.eos_token_id
    base_model.to("cuda")

    model = PeftModel.from_pretrained(base_model, str(official_run.adapter_dir), is_trainable=False)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.to("cuda")
    return model


def _configure_cuda_inference(use_bf16: bool, use_tf32: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 2 replay requires a CUDA-capable NVIDIA GPU.")
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The configured bf16 inference mode is not supported on this GPU.")
    if use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _prepare_output_root(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_responses_jsonl(path: Path, rows: tuple[Stage2ResponseRow, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            serialized = row.to_row()
            if tuple(serialized.keys()) != STAGE2_RESPONSE_COLUMNS:
                raise ValueError(
                    "Stage 2 response log row must match the exact schema "
                    f"{STAGE2_RESPONSE_COLUMNS}, found {tuple(serialized.keys())}."
                )
            handle.write(json.dumps(serialized, ensure_ascii=True, separators=(",", ":"), allow_nan=False))
            handle.write("\n")


def _write_filtered_responses_jsonl(path: Path, rows: tuple[Stage2FilteredResponseRow, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            serialized = row.to_row()
            if tuple(serialized.keys()) != STAGE2_FILTERED_RESPONSE_COLUMNS:
                raise ValueError(
                    "Stage 2 filtered response log row must match the exact schema "
                    f"{STAGE2_FILTERED_RESPONSE_COLUMNS}, found {tuple(serialized.keys())}."
                )
            handle.write(json.dumps(serialized, ensure_ascii=True, separators=(",", ":"), allow_nan=False))
            handle.write("\n")


def _write_family_metrics_csv(path: Path, rows: tuple[FamilyMetricRow, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAMILY_METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "exposure_condition": row.exposure_condition,
                    "prompt_condition": row.prompt_condition,
                    "template_family": row.template_family,
                    "attack_count": row.attack_count,
                    "name_leak_rate": _format_float(row.name_leak_rate),
                    "date_of_birth_leak_rate": _format_float(row.date_of_birth_leak_rate),
                    "account_number_leak_rate": _format_float(row.account_number_leak_rate),
                    "any_sensitive_field_leak_rate": _format_float(row.any_sensitive_field_leak_rate),
                    "full_canary_record_leak_rate": _format_float(row.full_canary_record_leak_rate),
                    "other_canary_leak_rate": _format_float(row.other_canary_leak_rate),
                    "refusal_rate": _format_float(row.refusal_rate),
                }
            )


def _write_filtered_family_metrics_csv(path: Path, rows: tuple[FilteredFamilyMetricRow, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FILTERED_FAMILY_METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "exposure_condition": row.exposure_condition,
                    "prompt_condition": row.prompt_condition,
                    "filter_mode": row.filter_mode,
                    "template_family": row.template_family,
                    "attack_count": row.attack_count,
                    "adversarial_block_rate": _format_float(row.adversarial_block_rate),
                    "filter_allow_rate": _format_float(row.filter_allow_rate),
                    "name_leak_rate": _format_float(row.name_leak_rate),
                    "date_of_birth_leak_rate": _format_float(row.date_of_birth_leak_rate),
                    "account_number_leak_rate": _format_float(row.account_number_leak_rate),
                    "any_sensitive_field_leak_rate": _format_float(row.any_sensitive_field_leak_rate),
                    "full_canary_record_leak_rate": _format_float(row.full_canary_record_leak_rate),
                    "other_canary_leak_rate": _format_float(row.other_canary_leak_rate),
                    "refusal_rate": _format_float(row.refusal_rate),
                    "leak_rate_given_allowed": _format_float(row.leak_rate_given_allowed),
                }
            )


def _write_summary_json(
    *,
    summary_path: Path,
    output_root: Path,
    exposure_order: tuple[str, ...],
    prompt_conditions: tuple[str, ...],
) -> None:
    runs = []
    for exposure_condition in exposure_order:
        for prompt_condition in prompt_conditions:
            metrics_path = output_root / exposure_condition / prompt_condition / "stage2_metrics.json"
            family_metrics_path = output_root / exposure_condition / prompt_condition / "family_metrics.csv"
            if not metrics_path.exists() or not family_metrics_path.exists():
                continue
            metrics = _read_json(metrics_path)
            runs.append(
                {
                    "exposure_condition": exposure_condition,
                    "prompt_condition": prompt_condition,
                    "run_name": metrics["run_name"],
                    "metrics_path": str(metrics_path),
                    "family_metrics_path": str(family_metrics_path),
                    "any_sensitive_field_leak_rate": metrics["any_sensitive_field_leak_rate"],
                    "full_canary_record_leak_rate": metrics["full_canary_record_leak_rate"],
                    "other_canary_leak_rate": metrics["other_canary_leak_rate"],
                    "refusal_rate": metrics["refusal_rate"],
                }
            )
    _write_json(summary_path, {"runs": runs})


def _write_filtered_summary_json(
    *,
    summary_path: Path,
    baseline_summary_path: Path,
    output_root: Path,
    exposure_order: tuple[str, ...],
    filter_modes: tuple[str, ...],
) -> None:
    baseline_summary = _read_json(baseline_summary_path)
    baseline_index = {
        (str(run["exposure_condition"]), str(run["prompt_condition"])): run
        for run in baseline_summary.get("runs", ())
    }

    runs = []
    for exposure_condition in exposure_order:
        baseline = baseline_index[(exposure_condition, INTEGRATED_PROMPT_CONDITION)]
        for filter_mode in filter_modes:
            metrics_path = output_root / exposure_condition / filter_mode / "stage2_filtered_metrics.json"
            family_metrics_path = output_root / exposure_condition / filter_mode / "family_metrics.csv"
            responses_path = output_root / exposure_condition / filter_mode / "stage2_filtered_responses.jsonl"
            if not metrics_path.exists() or not family_metrics_path.exists() or not responses_path.exists():
                continue
            metrics = _read_json(metrics_path)
            filtered_any = float(metrics["any_sensitive_field_leak_rate"])
            filtered_full = float(metrics["full_canary_record_leak_rate"])
            baseline_any = float(baseline["any_sensitive_field_leak_rate"])
            baseline_full = float(baseline["full_canary_record_leak_rate"])
            runs.append(
                {
                    "exposure_condition": exposure_condition,
                    "prompt_condition": INTEGRATED_PROMPT_CONDITION,
                    "filter_mode": filter_mode,
                    "run_name": metrics["run_name"],
                    "baseline_metrics_path": str(baseline["metrics_path"]),
                    "filtered_metrics_path": str(metrics_path),
                    "family_metrics_path": str(family_metrics_path),
                    "responses_path": str(responses_path),
                    "baseline_any_sensitive_field_leak_rate": baseline_any,
                    "baseline_full_canary_record_leak_rate": baseline_full,
                    "filtered_any_sensitive_field_leak_rate": filtered_any,
                    "filtered_full_canary_record_leak_rate": filtered_full,
                    "absolute_any_leak_reduction": baseline_any - filtered_any,
                    "absolute_full_leak_reduction": baseline_full - filtered_full,
                    "adversarial_block_rate": metrics["adversarial_block_rate"],
                    "filter_allow_rate": metrics["filter_allow_rate"],
                    "leak_rate_given_allowed": metrics["leak_rate_given_allowed"],
                }
            )
    _write_json(
        summary_path,
        {
            "baseline_summary_path": str(baseline_summary_path),
            "baseline_prompt_condition": INTEGRATED_PROMPT_CONDITION,
            "runs": runs,
        },
    )


def _write_filter_parity_summary_json(
    *,
    summary_path: Path,
    output_root: Path,
    exposure_order: tuple[str, ...],
    rows_by_condition: dict[tuple[str, str], tuple[Stage2FilteredResponseRow, ...]],
) -> None:
    runs = []
    for exposure_condition in exposure_order:
        plaintext_rows = rows_by_condition.get((exposure_condition, "plaintext_filter"))
        fhe_rows = rows_by_condition.get((exposure_condition, "fhe_filter"))
        if plaintext_rows is None or fhe_rows is None:
            continue
        parity_metrics = build_filter_parity_metrics(plaintext_rows, fhe_rows)
        runs.append(
            {
                "exposure_condition": exposure_condition,
                "plaintext_metrics_path": str(
                    output_root / exposure_condition / "plaintext_filter" / "stage2_filtered_metrics.json"
                ),
                "fhe_metrics_path": str(
                    output_root / exposure_condition / "fhe_filter" / "stage2_filtered_metrics.json"
                ),
                **parity_metrics,
            }
        )
    _write_json(summary_path, {"runs": runs})


def _build_decoding_payload(config: Stage2ReplayConfig | Stage2FilteredReplayConfig) -> dict[str, Any]:
    return {
        "temperature": config.decoding.temperature,
        "top_p": config.decoding.top_p,
        "max_new_tokens": config.decoding.max_new_tokens,
        "completions_per_prompt": config.decoding.completions_per_prompt,
        "retry_budget": config.decoding.retry_budget,
        "do_sample": False,
    }


def _validate_attack_embeddings(embeddings: np.ndarray, *, expected_dimension: int) -> None:
    if embeddings.ndim != 2:
        raise ValueError("Stage 2 filtered replay embeddings must be rank-2.")
    if embeddings.shape[1] != expected_dimension:
        raise ValueError(
            f"Stage 2 filtered replay embeddings must have dimension {expected_dimension}, "
            f"found {embeddings.shape[1]}."
        )
    if expected_dimension != EXPECTED_STAGE3_EMBEDDING_DIMENSION:
        raise ValueError(
            "Stage 2 filtered replay requires the frozen 384-dimensional Stage 3 encoder."
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_float(value: float) -> str:
    return format(float(value), ".16g")


def _release_model(model: torch.nn.Module) -> None:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
