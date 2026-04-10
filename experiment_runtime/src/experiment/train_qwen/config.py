from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
import json
import re
import tomllib

from qwen_langgraph_demo.runtime.protocol import load_protocol_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STAGE1_TRAIN_CONFIG_PATH = PROJECT_ROOT / "configs" / "train" / "stage1_lora.toml"
EXPOSURE_CONDITIONS = ("1x", "10x", "50x")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str
    trust_remote_code: bool
    attn_implementation: str


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    use_fast: bool
    add_generation_prompt: bool
    padding_side: str
    truncation_side: str
    max_sequence_length: int


@dataclass(frozen=True, slots=True)
class LoraHyperparameters:
    r: int
    lora_alpha: int
    lora_dropout: float
    bias: str
    target_modules: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrainingHyperparameters:
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_steps: int
    logging_steps: int
    save_steps: int
    save_total_limit: int
    lr_scheduler_type: str
    optim: str
    dataloader_num_workers: int
    dataloader_pin_memory: bool
    bf16: bool
    tf32: bool
    gradient_checkpointing: bool
    max_train_examples: int | None


@dataclass(frozen=True, slots=True)
class SmokeRunConfig:
    max_train_examples: int
    max_steps: int
    logging_steps: int
    save_steps: int


@dataclass(frozen=True, slots=True)
class Stage1TrainConfig:
    protocol_config_dir: Path
    corpus_paths: dict[str, Path]
    output_root: Path
    model: ModelConfig
    tokenizer: TokenizerConfig
    lora: LoraHyperparameters
    training: TrainingHyperparameters
    smoke: SmokeRunConfig
    seed: int

    @classmethod
    def from_toml(cls, config_path: Path | str | None = None) -> "Stage1TrainConfig":
        path = _resolve_path(config_path or DEFAULT_STAGE1_TRAIN_CONFIG_PATH)
        with path.open("rb") as handle:
            document = tomllib.load(handle)

        protocol_dir = _resolve_path(document["protocol"]["config_dir"])
        protocol = load_protocol_bundle(protocol_dir)

        corpus_paths = {
            str(exposure): _resolve_path(corpus_path)
            for exposure, corpus_path in document["inputs"]["training_corpora"].items()
        }
        if tuple(corpus_paths) != tuple(protocol.core.exposure_conditions):
            raise ValueError(
                "Stage 1 training config must define corpus paths for exposure conditions "
                f"{protocol.core.exposure_conditions}, found {tuple(corpus_paths)}."
            )

        model = ModelConfig(
            name=str(document["model"]["name"]),
            trust_remote_code=bool(document["model"]["trust_remote_code"]),
            attn_implementation=str(document["model"]["attn_implementation"]),
        )
        if model.name != protocol.core.primary_model:
            raise ValueError(
                "Stage 1 training must target the frozen primary model "
                f"{protocol.core.primary_model}, found {model.name}."
            )

        tokenizer = TokenizerConfig(
            use_fast=bool(document["tokenizer"]["use_fast"]),
            add_generation_prompt=bool(document["tokenizer"]["add_generation_prompt"]),
            padding_side=str(document["tokenizer"]["padding_side"]),
            truncation_side=str(document["tokenizer"]["truncation_side"]),
            max_sequence_length=int(document["tokenizer"]["max_sequence_length"]),
        )
        if tokenizer.add_generation_prompt:
            raise ValueError("Stage 1 training must set tokenizer.add_generation_prompt = false.")
        if tokenizer.max_sequence_length <= 0:
            raise ValueError("tokenizer.max_sequence_length must be positive.")

        lora = LoraHyperparameters(
            r=int(document["lora"]["r"]),
            lora_alpha=int(document["lora"]["lora_alpha"]),
            lora_dropout=float(document["lora"]["lora_dropout"]),
            bias=str(document["lora"]["bias"]),
            target_modules=tuple(str(module) for module in document["lora"]["target_modules"]),
        )
        if not lora.target_modules:
            raise ValueError("lora.target_modules must not be empty.")

        training_doc = document["training"]
        training = TrainingHyperparameters(
            per_device_train_batch_size=int(training_doc["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(training_doc["gradient_accumulation_steps"]),
            learning_rate=float(training_doc["learning_rate"]),
            weight_decay=float(training_doc["weight_decay"]),
            warmup_ratio=float(training_doc["warmup_ratio"]),
            max_steps=int(training_doc["max_steps"]),
            logging_steps=int(training_doc["logging_steps"]),
            save_steps=int(training_doc["save_steps"]),
            save_total_limit=int(training_doc["save_total_limit"]),
            lr_scheduler_type=str(training_doc["lr_scheduler_type"]),
            optim=str(training_doc["optim"]),
            dataloader_num_workers=int(training_doc["dataloader_num_workers"]),
            dataloader_pin_memory=bool(training_doc["dataloader_pin_memory"]),
            bf16=bool(training_doc["bf16"]),
            tf32=bool(training_doc["tf32"]),
            gradient_checkpointing=bool(training_doc["gradient_checkpointing"]),
            max_train_examples=_normalize_optional_int(training_doc.get("max_train_examples")),
        )
        _validate_training_hyperparameters(training)

        smoke = SmokeRunConfig(
            max_train_examples=int(document["smoke"]["max_train_examples"]),
            max_steps=int(document["smoke"]["max_steps"]),
            logging_steps=int(document["smoke"]["logging_steps"]),
            save_steps=int(document["smoke"]["save_steps"]),
        )
        if smoke.max_train_examples <= 0:
            raise ValueError("smoke.max_train_examples must be positive.")
        if min(smoke.max_steps, smoke.logging_steps, smoke.save_steps) <= 0:
            raise ValueError("Smoke-run max_steps, logging_steps, and save_steps must be positive.")

        return cls(
            protocol_config_dir=protocol_dir,
            corpus_paths=corpus_paths,
            output_root=_resolve_path(document["outputs"]["root_dir"]),
            model=model,
            tokenizer=tokenizer,
            lora=lora,
            training=training,
            smoke=smoke,
            seed=int(document["seed"]["value"]),
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "protocol": {
                "config_dir": str(self.protocol_config_dir),
            },
            "inputs": {
                "training_corpora": {exposure: str(path) for exposure, path in self.corpus_paths.items()},
            },
            "outputs": {
                "root_dir": str(self.output_root),
            },
            "model": asdict(self.model),
            "tokenizer": asdict(self.tokenizer),
            "lora": {
                **asdict(self.lora),
                "target_modules": list(self.lora.target_modules),
            },
            "training": _training_to_document(self.training),
            "smoke": asdict(self.smoke),
            "seed": {
                "value": self.seed,
            },
        }


@dataclass(frozen=True, slots=True)
class Stage1RunConfig:
    config_path: Path
    protocol_config_dir: Path
    corpus_path: Path
    output_root: Path
    run_name: str
    run_dir: Path
    exposure_condition: str
    smoke_enabled: bool
    model: ModelConfig
    tokenizer: TokenizerConfig
    lora: LoraHyperparameters
    training: TrainingHyperparameters
    seed: int

    def to_document(self) -> dict[str, Any]:
        return {
            "config": {
                "path": str(self.config_path),
                "smoke_enabled": self.smoke_enabled,
            },
            "protocol": {
                "config_dir": str(self.protocol_config_dir),
            },
            "run": {
                "name": self.run_name,
                "exposure_condition": self.exposure_condition,
                "run_dir": str(self.run_dir),
            },
            "inputs": {
                "corpus_path": str(self.corpus_path),
            },
            "outputs": {
                "root_dir": str(self.output_root),
            },
            "model": asdict(self.model),
            "tokenizer": asdict(self.tokenizer),
            "lora": {
                **asdict(self.lora),
                "target_modules": list(self.lora.target_modules),
            },
            "training": _training_to_document(self.training),
            "seed": {
                "value": self.seed,
            },
        }


def resolve_run_config(
    config: Stage1TrainConfig,
    *,
    config_path: Path | str | None,
    exposure_condition: str,
    run_name: str | None,
    smoke: bool,
) -> Stage1RunConfig:
    if exposure_condition not in EXPOSURE_CONDITIONS:
        raise ValueError(
            f"Unsupported exposure condition {exposure_condition!r}; "
            f"expected one of {EXPOSURE_CONDITIONS}."
        )
    corpus_path = config.corpus_paths[exposure_condition]
    resolved_run_name = normalize_run_name(run_name or f"stage1-{exposure_condition}")
    effective_training = config.training
    if smoke:
        effective_training = replace(
            effective_training,
            max_train_examples=config.smoke.max_train_examples,
            max_steps=config.smoke.max_steps,
            logging_steps=config.smoke.logging_steps,
            save_steps=config.smoke.save_steps,
        )

    return Stage1RunConfig(
        config_path=_resolve_path(config_path or DEFAULT_STAGE1_TRAIN_CONFIG_PATH),
        protocol_config_dir=config.protocol_config_dir,
        corpus_path=corpus_path,
        output_root=config.output_root,
        run_name=resolved_run_name,
        run_dir=config.output_root / resolved_run_name,
        exposure_condition=exposure_condition,
        smoke_enabled=smoke,
        model=config.model,
        tokenizer=config.tokenizer,
        lora=config.lora,
        training=effective_training,
        seed=config.seed,
    )


def normalize_run_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    normalized = normalized.strip(".-")
    if not normalized:
        raise ValueError("Run name must contain at least one alphanumeric character.")
    return normalized


def render_toml_document(document: dict[str, Any]) -> str:
    lines: list[str] = []
    _render_table(lines, document, prefix=())
    return "\n".join(lines).rstrip() + "\n"


def _render_table(lines: list[str], table: dict[str, Any], *, prefix: tuple[str, ...]) -> None:
    scalar_items: list[tuple[str, Any]] = []
    table_items: list[tuple[str, dict[str, Any]]] = []
    for key, value in table.items():
        if isinstance(value, dict):
            table_items.append((key, value))
        else:
            scalar_items.append((key, value))

    if prefix and scalar_items:
        if lines:
            lines.append("")
        lines.append(f"[{'.'.join(_format_key_part(part) for part in prefix)}]")

    for key, value in scalar_items:
        lines.append(f"{_format_key_part(key)} = {_format_value(value)}")

    for key, nested in table_items:
        _render_table(lines, nested, prefix=prefix + (key,))


def _format_key_part(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]+", value):
        return value
    return json.dumps(value, ensure_ascii=True)


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Path):
        return json.dumps(str(value), ensure_ascii=True)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, tuple):
        return _format_value(list(value))
    if isinstance(value, list):
        rendered = ", ".join(_format_value(item) for item in value)
        return f"[{rendered}]"
    raise TypeError(f"Unsupported TOML value type: {type(value)!r}")


def _training_to_document(training: TrainingHyperparameters) -> dict[str, Any]:
    document = asdict(training)
    if document["max_train_examples"] is None:
        document["max_train_examples"] = 0
    return document


def _validate_training_hyperparameters(training: TrainingHyperparameters) -> None:
    positive_fields = {
        "per_device_train_batch_size": training.per_device_train_batch_size,
        "gradient_accumulation_steps": training.gradient_accumulation_steps,
        "learning_rate": training.learning_rate,
        "max_steps": training.max_steps,
        "logging_steps": training.logging_steps,
        "save_steps": training.save_steps,
        "save_total_limit": training.save_total_limit,
    }
    for field_name, value in positive_fields.items():
        if value <= 0:
            raise ValueError(f"training.{field_name} must be positive.")

    if training.weight_decay < 0:
        raise ValueError("training.weight_decay must be non-negative.")
    if not 0 <= training.warmup_ratio <= 1:
        raise ValueError("training.warmup_ratio must be between 0 and 1.")
    if training.max_train_examples is not None and training.max_train_examples <= 0:
        raise ValueError("training.max_train_examples must be positive when provided.")


def _normalize_optional_int(value: Any) -> int | None:
    if value in (None, 0, "0"):
        return None
    parsed = int(value)
    return parsed


def _resolve_path(path_value: Path | str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
