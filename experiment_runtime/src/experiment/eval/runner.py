from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, set_seed

from qwen_langgraph_demo.runtime.protocol import load_protocol_bundle

from experiment.eval.config import (
    DEFAULT_STAGE2_REPLAY_CONFIG_PATH,
    OfficialRunReference,
    Stage2ReplayConfig,
    resolve_exposure_conditions,
    resolve_prompt_conditions,
)
from experiment.eval.data import STAGE2_RESPONSE_COLUMNS, Stage2ResponseRow, build_prompt_messages, load_canary_registry_index, load_stage2_attack_prompts
from experiment.eval.metrics import FAMILY_METRIC_COLUMNS, FamilyMetricRow, build_condition_metrics, build_family_metrics
from experiment.eval.scoring import CanaryLeakageScorer
from experiment.schemas.stage2 import Stage2AttackPrompt
from experiment.train_qwen.data import load_stage1_tokenizer, tokenize_chat_messages


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

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 2 replay requires a CUDA-capable NVIDIA GPU.")
    if config.inference.bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The configured bf16 inference mode is not supported on this GPU.")
    if config.inference.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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
                    canary_registry=canary_registry,
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
        exposure_order=tuple(config.official_runs),
        prompt_conditions=selected_conditions,
    )
    return Stage2EvaluationResult(
        summary_path=summary_path,
        artifacts=artifacts,
    )


def _evaluate_condition(
    *,
    config: Stage2ReplayConfig,
    protocol: Any,
    official_run: OfficialRunReference,
    prompt_condition: str,
    attack_prompts: tuple[Stage2AttackPrompt, ...],
    canary_registry: dict[str, Any],
    scorer: CanaryLeakageScorer,
    tokenizer: Any,
    model: torch.nn.Module,
) -> Stage2ConditionArtifacts:
    response_rows = _generate_and_score_responses(
        attack_prompts=attack_prompts,
        prompt_condition=prompt_condition,
        official_run=official_run,
        protocol=protocol,
        canary_registry=canary_registry,
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

    decoding_payload = {
        "temperature": config.decoding.temperature,
        "top_p": config.decoding.top_p,
        "max_new_tokens": config.decoding.max_new_tokens,
        "completions_per_prompt": config.decoding.completions_per_prompt,
        "retry_budget": config.decoding.retry_budget,
        "do_sample": False,
    }
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
        decoding=decoding_payload,
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


def _generate_and_score_responses(
    *,
    attack_prompts: tuple[Stage2AttackPrompt, ...],
    prompt_condition: str,
    official_run: OfficialRunReference,
    protocol: Any,
    canary_registry: dict[str, Any],
    scorer: CanaryLeakageScorer,
    tokenizer: Any,
    model: torch.nn.Module,
    batch_size: int,
    max_sequence_length: int,
    add_generation_prompt: bool,
    max_new_tokens: int,
) -> tuple[Stage2ResponseRow, ...]:
    model.eval()
    rows: list[Stage2ResponseRow] = []
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
                if attack_prompt.target_canary_id not in canary_registry:
                    raise KeyError(
                        f"Attack prompt {attack_prompt.attack_id} targeted missing canary "
                        f"{attack_prompt.target_canary_id!r}."
                    )
                response_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
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


def _load_adapter_model(
    config: Stage2ReplayConfig,
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
