from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import inspect
from pathlib import Path
from typing import Any
import json
import os
import platform
import sys

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .config import (
    DEFAULT_STAGE1_TRAIN_CONFIG_PATH,
    Stage1RunConfig,
    Stage1TrainConfig,
    render_toml_document,
    resolve_run_config,
)
from .data import FullSequenceDataCollator, prepare_training_dataset


@dataclass(frozen=True, slots=True)
class Stage1TrainingResult:
    run_dir: Path
    adapter_dir: Path
    tokenizer_dir: Path
    metrics_path: Path
    metadata_path: Path
    environment_path: Path
    resolved_config_path: Path
    train_metrics: dict[str, Any]
    train_examples: int


def run_stage1_training(
    *,
    config_path: Path | str | None = None,
    exposure_condition: str,
    run_name: str | None = None,
    smoke: bool = False,
) -> Stage1TrainingResult:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    config = Stage1TrainConfig.from_toml(config_path or DEFAULT_STAGE1_TRAIN_CONFIG_PATH)
    run_config = resolve_run_config(
        config,
        config_path=config_path or DEFAULT_STAGE1_TRAIN_CONFIG_PATH,
        exposure_condition=exposure_condition,
        run_name=run_name,
        smoke=smoke,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 1 training requires a CUDA-capable NVIDIA GPU.")
    if run_config.training.bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The configured bf16 training mode is not supported on this GPU.")

    _prepare_run_directory(run_config.run_dir)
    resolved_config_path = run_config.run_dir / "resolved_config.toml"
    resolved_config_path.write_text(render_toml_document(run_config.to_document()), encoding="utf-8")

    start_time = _utc_now()
    tokenizer = _load_tokenizer(run_config)
    prepared_dataset = prepare_training_dataset(
        run_config.corpus_path,
        exposure_condition=run_config.exposure_condition,
        tokenizer=tokenizer,
        max_sequence_length=run_config.tokenizer.max_sequence_length,
        add_generation_prompt=run_config.tokenizer.add_generation_prompt,
        max_examples=run_config.training.max_train_examples,
    )

    if run_config.training.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    set_seed(run_config.seed)
    model = _load_model(run_config, tokenizer)
    trainable_summary = _summarize_parameters(model)

    trainer = Trainer(
        **_build_trainer_kwargs(
            model=model,
            run_config=run_config,
            train_dataset=prepared_dataset.dataset,
            tokenizer=tokenizer,
        )
    )

    train_result = trainer.train()
    trainer.save_state()
    trainer_state_path = run_config.run_dir / "trainer_state.json"
    trainer.state.save_to_json(str(trainer_state_path))

    adapter_dir = run_config.run_dir / "adapter_model"
    tokenizer_dir = run_config.run_dir / "tokenizer"
    trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(tokenizer_dir)

    train_metrics = dict(train_result.metrics)
    train_metrics["train_examples"] = prepared_dataset.summary.example_count
    train_metrics["sequence_length_min"] = prepared_dataset.summary.min_sequence_length
    train_metrics["sequence_length_max"] = prepared_dataset.summary.max_sequence_length
    train_metrics["sequence_length_mean"] = prepared_dataset.summary.mean_sequence_length
    metrics_path = run_config.run_dir / "train_metrics.json"
    _write_json(metrics_path, train_metrics)

    end_time = _utc_now()
    metadata_path = run_config.run_dir / "run_metadata.json"
    _write_json(
        metadata_path,
        {
            "run_name": run_config.run_name,
            "run_dir": str(run_config.run_dir),
            "trainer_output_dir": str(_trainer_output_dir(run_config)),
            "exposure_condition": run_config.exposure_condition,
            "smoke_enabled": run_config.smoke_enabled,
            "corpus_path": str(run_config.corpus_path),
            "train_examples": prepared_dataset.summary.example_count,
            "sequence_length_summary": {
                "exposure_condition": prepared_dataset.summary.exposure_condition,
                "source_path": str(prepared_dataset.summary.source_path),
                "example_count": prepared_dataset.summary.example_count,
                "min_sequence_length": prepared_dataset.summary.min_sequence_length,
                "max_sequence_length": prepared_dataset.summary.max_sequence_length,
                "mean_sequence_length": prepared_dataset.summary.mean_sequence_length,
            },
            "base_model_name": run_config.model.name,
            "single_gpu_only": True,
            "started_at_utc": start_time,
            "completed_at_utc": end_time,
            "command": sys.argv,
            "trainable_parameters": trainable_summary,
        },
    )

    environment_path = run_config.run_dir / "environment.json"
    _write_json(environment_path, _build_environment_summary(run_config))

    return Stage1TrainingResult(
        run_dir=run_config.run_dir,
        adapter_dir=adapter_dir,
        tokenizer_dir=tokenizer_dir,
        metrics_path=metrics_path,
        metadata_path=metadata_path,
        environment_path=environment_path,
        resolved_config_path=resolved_config_path,
        train_metrics=train_metrics,
        train_examples=prepared_dataset.summary.example_count,
    )


def _build_training_arguments(run_config: Stage1RunConfig) -> TrainingArguments:
    kwargs = {
        "output_dir": str(_trainer_output_dir(run_config)),
        "do_train": True,
        "per_device_train_batch_size": run_config.training.per_device_train_batch_size,
        "gradient_accumulation_steps": run_config.training.gradient_accumulation_steps,
        "learning_rate": run_config.training.learning_rate,
        "weight_decay": run_config.training.weight_decay,
        "warmup_ratio": run_config.training.warmup_ratio,
        "max_steps": run_config.training.max_steps,
        "logging_strategy": "steps",
        "logging_steps": run_config.training.logging_steps,
        "logging_first_step": True,
        "save_strategy": "steps",
        "save_steps": run_config.training.save_steps,
        "save_total_limit": run_config.training.save_total_limit,
        "lr_scheduler_type": run_config.training.lr_scheduler_type,
        "optim": run_config.training.optim,
        "dataloader_num_workers": run_config.training.dataloader_num_workers,
        "dataloader_pin_memory": run_config.training.dataloader_pin_memory,
        "remove_unused_columns": False,
        "bf16": run_config.training.bf16,
        "tf32": run_config.training.tf32,
        "gradient_checkpointing": run_config.training.gradient_checkpointing,
        "report_to": [],
        "seed": run_config.seed,
        "data_seed": run_config.seed,
        "save_safetensors": True,
        "run_name": run_config.run_name,
    }
    return TrainingArguments(**_filter_supported_kwargs(TrainingArguments.__init__, kwargs))


def _build_trainer_kwargs(
    *,
    model: torch.nn.Module,
    run_config: Stage1RunConfig,
    train_dataset: Any,
    tokenizer: Any,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "args": _build_training_arguments(run_config),
        "train_dataset": train_dataset,
        "data_collator": FullSequenceDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8),
    }

    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        kwargs["tokenizer"] = tokenizer
    return kwargs


def _load_tokenizer(run_config: Stage1RunConfig) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        run_config.model.name,
        use_fast=run_config.tokenizer.use_fast,
        trust_remote_code=run_config.model.trust_remote_code,
    )
    tokenizer.padding_side = run_config.tokenizer.padding_side
    tokenizer.truncation_side = run_config.tokenizer.truncation_side
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise RuntimeError("Tokenizer does not define a pad token or eos token.")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(run_config: Stage1RunConfig, tokenizer: Any) -> torch.nn.Module:
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16 if run_config.training.bf16 else torch.float32,
        "trust_remote_code": run_config.model.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if run_config.model.attn_implementation:
        model_kwargs["attn_implementation"] = run_config.model.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(run_config.model.name, **model_kwargs)
    model.config.use_cache = False
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=run_config.lora.r,
        lora_alpha=run_config.lora.lora_alpha,
        lora_dropout=run_config.lora.lora_dropout,
        bias=run_config.lora.bias,
        target_modules=list(run_config.lora.target_modules),
    )
    return get_peft_model(model, lora_config)


def _prepare_run_directory(run_dir: Path) -> None:
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)


def _trainer_output_dir(run_config: Stage1RunConfig) -> Path:
    return run_config.run_dir / "trainer_output"


def _build_environment_summary(run_config: Stage1RunConfig) -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": _package_version("torch"),
        "transformers_version": _package_version("transformers"),
        "peft_version": _package_version("peft"),
        "accelerate_version": _package_version("accelerate"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "bf16_enabled": run_config.training.bf16,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "tf32_enabled": run_config.training.tf32,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _summarize_parameters(model: torch.nn.Module) -> dict[str, Any]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
    trainable_ratio = 0.0 if total == 0 else trainable / total
    return {
        "total": total,
        "trainable": trainable,
        "trainable_ratio": trainable_ratio,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _package_version(distribution_name: str) -> str | None:
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return None


def _filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    supported = inspect.signature(callable_obj).parameters
    return {key: value for key, value in kwargs.items() if key in supported}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
