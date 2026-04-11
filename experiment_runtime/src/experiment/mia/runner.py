from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import json

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM

from experiment.mia.config import (
    DEFAULT_STAGE1_MIA_CONFIG_PATH,
    OfficialRunReference,
    Stage1MiaConfig,
    resolve_exposure_conditions,
)
from experiment.mia.data import TokenizedMiaExample, load_mia_eval_examples, tokenize_mia_examples
from experiment.mia.metrics import compute_bootstrap_intervals, compute_membership_score, compute_roc_metrics
from experiment.train_qwen.data import FullSequenceDataCollator, load_stage1_tokenizer


BASE_LOSS_COLUMNS = ("eval_id", "record_id", "split", "is_member", "is_canary", "loss_base")
STAGE1_LOSS_COLUMNS = (
    "eval_id",
    "record_id",
    "split",
    "is_member",
    "is_canary",
    "exposure_condition",
    "run_name",
    "loss_base",
    "loss_ft",
    "membership_score",
)
ROC_COLUMNS = ("threshold", "fpr", "tpr")


@dataclass(frozen=True, slots=True)
class BaseLossRow:
    eval_id: str
    record_id: str
    split: str
    is_member: int
    is_canary: int
    loss_base: float


@dataclass(frozen=True, slots=True)
class ExposureLossRow:
    eval_id: str
    record_id: str
    split: str
    is_member: int
    is_canary: int
    exposure_condition: str
    run_name: str
    loss_base: float
    loss_ft: float
    membership_score: float


@dataclass(frozen=True, slots=True)
class ExposureArtifacts:
    exposure_condition: str
    stage1_losses_path: Path
    stage1_metrics_path: Path
    roc_curve_path: Path
    canary_metrics_path: Path
    bootstrap_metrics_path: Path


@dataclass(frozen=True, slots=True)
class Stage1MiaResult:
    base_losses_path: Path
    summary_path: Path
    exposure_artifacts: dict[str, ExposureArtifacts]


def run_stage1_mia_evaluation(
    *,
    config_path: Path | str | None = None,
    exposure: str,
) -> Stage1MiaResult:
    config = Stage1MiaConfig.from_toml(config_path or DEFAULT_STAGE1_MIA_CONFIG_PATH)
    selected_exposures = resolve_exposure_conditions(exposure)
    _prepare_output_root(config.output_root)

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 1 MIA evaluation requires a CUDA-capable NVIDIA GPU.")
    if config.inference.bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The configured bf16 inference mode is not supported on this GPU.")
    if config.inference.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = load_stage1_tokenizer(
        config.tokenizer.source,
        use_fast=config.tokenizer.use_fast,
        trust_remote_code=config.model.trust_remote_code,
        padding_side=config.tokenizer.padding_side,
        truncation_side=config.tokenizer.truncation_side,
    )
    source_examples = load_mia_eval_examples(config.mia_eval_path)
    tokenized_examples = tokenize_mia_examples(
        source_examples,
        tokenizer=tokenizer,
        max_sequence_length=config.tokenizer.max_sequence_length,
        add_generation_prompt=config.tokenizer.add_generation_prompt,
    )

    base_losses_path = config.output_root / "base_losses.csv"
    base_loss_rows = _load_base_loss_cache(base_losses_path, tokenized_examples)
    if base_loss_rows is None:
        base_model = _load_base_model(config)
        try:
            base_losses = _compute_model_losses(
                model=base_model,
                tokenizer=tokenizer,
                tokenized_examples=tokenized_examples,
                batch_size=config.inference.batch_size,
            )
        finally:
            _release_model(base_model)
        base_loss_rows = tuple(
            BaseLossRow(
                eval_id=example.eval_id,
                record_id=example.record_id,
                split=example.split,
                is_member=example.is_member,
                is_canary=example.is_canary,
                loss_base=loss_base,
            )
            for example, loss_base in zip(tokenized_examples, base_losses, strict=True)
        )
        _write_base_losses_csv(base_losses_path, base_loss_rows)

    exposure_artifacts: dict[str, ExposureArtifacts] = {}
    for exposure_condition in selected_exposures:
        official_run = config.official_runs[exposure_condition]
        exposure_artifacts[exposure_condition] = _evaluate_single_exposure(
            config=config,
            official_run=official_run,
            tokenizer=tokenizer,
            tokenized_examples=tokenized_examples,
            base_loss_rows=base_loss_rows,
        )

    summary_path = config.output_root / "mia_summary.json"
    _write_summary_json(
        summary_path=summary_path,
        base_losses_path=base_losses_path,
        output_root=config.output_root,
        exposures=selected_exposures,
    )
    return Stage1MiaResult(
        base_losses_path=base_losses_path,
        summary_path=summary_path,
        exposure_artifacts=exposure_artifacts,
    )


def _evaluate_single_exposure(
    *,
    config: Stage1MiaConfig,
    official_run: OfficialRunReference,
    tokenizer: Any,
    tokenized_examples: tuple[TokenizedMiaExample, ...],
    base_loss_rows: tuple[BaseLossRow, ...],
) -> ExposureArtifacts:
    model = _load_adapter_model(config, official_run)
    try:
        fine_tuned_losses = _compute_model_losses(
            model=model,
            tokenizer=tokenizer,
            tokenized_examples=tokenized_examples,
            batch_size=config.inference.batch_size,
        )
    finally:
        _release_model(model)

    exposure_output_dir = config.output_root / official_run.exposure_condition
    exposure_output_dir.mkdir(parents=True, exist_ok=True)

    stage1_loss_rows = tuple(
        ExposureLossRow(
            eval_id=base_row.eval_id,
            record_id=base_row.record_id,
            split=base_row.split,
            is_member=base_row.is_member,
            is_canary=base_row.is_canary,
            exposure_condition=official_run.exposure_condition,
            run_name=official_run.run_name,
            loss_base=base_row.loss_base,
            loss_ft=loss_ft,
            membership_score=compute_membership_score(base_row.loss_base, loss_ft),
        )
        for base_row, loss_ft in zip(base_loss_rows, fine_tuned_losses, strict=True)
    )

    stage1_losses_path = exposure_output_dir / "stage1_losses.csv"
    _write_stage1_losses_csv(stage1_losses_path, stage1_loss_rows)

    full_metrics = _build_metrics_payload(
        rows=stage1_loss_rows,
        exposure_condition=official_run.exposure_condition,
        run_name=official_run.run_name,
        base_model_name=config.model.name,
        adapter_run_dir=official_run.run_dir,
        mia_eval_path=config.mia_eval_path,
    )
    stage1_metrics_path = exposure_output_dir / "stage1_metrics.json"
    _write_json(stage1_metrics_path, full_metrics["metrics"])

    roc_curve_path = exposure_output_dir / "roc_curve.csv"
    _write_roc_curve_csv(roc_curve_path, full_metrics["roc"])

    canary_rows = tuple(row for row in stage1_loss_rows if row.is_member == 0 or row.is_canary == 1)
    canary_metrics = _build_metrics_payload(
        rows=canary_rows,
        exposure_condition=official_run.exposure_condition,
        run_name=official_run.run_name,
        base_model_name=config.model.name,
        adapter_run_dir=official_run.run_dir,
        mia_eval_path=config.mia_eval_path,
    )
    canary_metrics_path = exposure_output_dir / "canary_metrics.json"
    _write_json(canary_metrics_path, canary_metrics["metrics"])

    bootstrap_payload = {
        "exposure_condition": official_run.exposure_condition,
        "run_name": official_run.run_name,
        "base_model_name": config.model.name,
        "adapter_run_dir": str(official_run.run_dir),
        "mia_eval_path": str(config.mia_eval_path),
        **compute_bootstrap_intervals(
            [row.is_member for row in stage1_loss_rows],
            [row.membership_score for row in stage1_loss_rows],
            replicates=config.bootstrap.replicates,
            confidence_level=config.bootstrap.confidence_level,
            seed=config.seed,
        ),
        "point_estimates": {
            "auc_roc": full_metrics["metrics"]["auc_roc"],
            "tpr_at_1_fpr": full_metrics["metrics"]["tpr_at_1_fpr"],
            "tpr_at_10_fpr": full_metrics["metrics"]["tpr_at_10_fpr"],
        },
    }
    bootstrap_metrics_path = exposure_output_dir / "bootstrap_metrics.json"
    _write_json(bootstrap_metrics_path, bootstrap_payload)

    return ExposureArtifacts(
        exposure_condition=official_run.exposure_condition,
        stage1_losses_path=stage1_losses_path,
        stage1_metrics_path=stage1_metrics_path,
        roc_curve_path=roc_curve_path,
        canary_metrics_path=canary_metrics_path,
        bootstrap_metrics_path=bootstrap_metrics_path,
    )


def _build_metrics_payload(
    *,
    rows: tuple[ExposureLossRow, ...],
    exposure_condition: str,
    run_name: str,
    base_model_name: str,
    adapter_run_dir: Path,
    mia_eval_path: Path,
) -> dict[str, object]:
    labels = [row.is_member for row in rows]
    scores = [row.membership_score for row in rows]
    roc_metrics = compute_roc_metrics(labels, scores)
    point_1 = roc_metrics.operating_point(0.01)
    point_10 = roc_metrics.operating_point(0.10)
    metrics = {
        "exposure_condition": exposure_condition,
        "run_name": run_name,
        "base_model_name": base_model_name,
        "adapter_run_dir": str(adapter_run_dir),
        "mia_eval_path": str(mia_eval_path),
        "example_count": len(rows),
        "member_count": sum(row.is_member for row in rows),
        "non_member_count": sum(1 for row in rows if row.is_member == 0),
        "canary_count": sum(row.is_canary for row in rows),
        "membership_score_definition": "loss_base / loss_ft",
        "auc_roc": roc_metrics.auc_roc,
        "tpr_at_1_fpr": point_1.tpr,
        "threshold_at_1_fpr": point_1.threshold,
        "tpr_at_10_fpr": point_10.tpr,
        "threshold_at_10_fpr": point_10.threshold,
    }
    roc_rows = [
        {
            "threshold": threshold,
            "fpr": fpr,
            "tpr": tpr,
        }
        for threshold, fpr, tpr in zip(
            roc_metrics.thresholds,
            roc_metrics.fpr,
            roc_metrics.tpr,
            strict=True,
        )
    ]
    return {
        "metrics": metrics,
        "roc": roc_rows,
    }


def _compute_model_losses(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    tokenized_examples: tuple[TokenizedMiaExample, ...],
    batch_size: int,
) -> tuple[float, ...]:
    device = torch.device("cuda")
    collator = FullSequenceDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
    losses: list[float] = []

    model.eval()
    with torch.inference_mode():
        for start_index in range(0, len(tokenized_examples), batch_size):
            batch_examples = tokenized_examples[start_index : start_index + batch_size]
            batch = collator([example.to_feature() for example in batch_examples])
            batch = {
                key: value.to(device=device, non_blocking=True)
                for key, value in batch.items()
            }
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            losses.extend(_per_example_losses(outputs.logits, batch["labels"]))
    return tuple(losses)


def _per_example_losses(logits: torch.Tensor, labels: torch.Tensor) -> list[float]:
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    flat_losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.shape[0], shift_labels.shape[1])
    valid_positions = shift_labels.ne(-100)
    token_counts = valid_positions.sum(dim=1)
    if torch.any(token_counts <= 0):
        raise ValueError("Per-example loss requires at least one unmasked label position per example.")
    per_example_losses = flat_losses.sum(dim=1) / token_counts
    return [float(value.item()) for value in per_example_losses]


def _load_base_model(config: Stage1MiaConfig) -> torch.nn.Module:
    dtype = torch.bfloat16 if config.inference.bf16 else torch.float32
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": config.model.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if config.model.attn_implementation:
        model_kwargs["attn_implementation"] = config.model.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(config.model.name, **model_kwargs)
    model.config.use_cache = False
    model.to("cuda")
    return model


def _load_adapter_model(config: Stage1MiaConfig, official_run: OfficialRunReference) -> torch.nn.Module:
    base_model = _load_base_model(config)
    model = PeftModel.from_pretrained(base_model, str(official_run.adapter_dir), is_trainable=False)
    model.config.use_cache = False
    model.to("cuda")
    return model


def _load_base_loss_cache(
    path: Path,
    tokenized_examples: tuple[TokenizedMiaExample, ...],
) -> tuple[BaseLossRow, ...] | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != BASE_LOSS_COLUMNS:
            return None
        rows = tuple(
            BaseLossRow(
                eval_id=str(row["eval_id"]),
                record_id=str(row["record_id"]),
                split=str(row["split"]),
                is_member=int(row["is_member"]),
                is_canary=int(row["is_canary"]),
                loss_base=float(row["loss_base"]),
            )
            for row in reader
        )

    if len(rows) != len(tokenized_examples):
        return None
    for row, example in zip(rows, tokenized_examples, strict=True):
        if (
            row.eval_id != example.eval_id
            or row.record_id != example.record_id
            or row.split != example.split
            or row.is_member != example.is_member
            or row.is_canary != example.is_canary
        ):
            return None
    return rows


def _write_base_losses_csv(path: Path, rows: tuple[BaseLossRow, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(BASE_LOSS_COLUMNS)
        for row in rows:
            writer.writerow(
                (
                    row.eval_id,
                    row.record_id,
                    row.split,
                    row.is_member,
                    row.is_canary,
                    _format_float(row.loss_base),
                )
            )


def _write_stage1_losses_csv(path: Path, rows: tuple[ExposureLossRow, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(STAGE1_LOSS_COLUMNS)
        for row in rows:
            writer.writerow(
                (
                    row.eval_id,
                    row.record_id,
                    row.split,
                    row.is_member,
                    row.is_canary,
                    row.exposure_condition,
                    row.run_name,
                    _format_float(row.loss_base),
                    _format_float(row.loss_ft),
                    _format_float(row.membership_score),
                )
            )


def _write_roc_curve_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(ROC_COLUMNS)
        for row in rows:
            writer.writerow(
                (
                    _format_float(row["threshold"]),
                    _format_float(row["fpr"]),
                    _format_float(row["tpr"]),
                )
            )


def _write_summary_json(
    *,
    summary_path: Path,
    base_losses_path: Path,
    output_root: Path,
    exposures: tuple[str, ...],
) -> None:
    runs = []
    for exposure in exposures:
        exposure_dir = output_root / exposure
        metrics = _read_json(exposure_dir / "stage1_metrics.json")
        canary_metrics = _read_json(exposure_dir / "canary_metrics.json")
        bootstrap_metrics = _read_json(exposure_dir / "bootstrap_metrics.json")
        runs.append(
            {
                "exposure_condition": exposure,
                "run_name": metrics["run_name"],
                "metrics_path": str(exposure_dir / "stage1_metrics.json"),
                "canary_metrics_path": str(exposure_dir / "canary_metrics.json"),
                "bootstrap_metrics_path": str(exposure_dir / "bootstrap_metrics.json"),
                "auc_roc": metrics["auc_roc"],
                "tpr_at_1_fpr": metrics["tpr_at_1_fpr"],
                "tpr_at_10_fpr": metrics["tpr_at_10_fpr"],
                "canary_auc_roc": canary_metrics["auc_roc"],
                "bootstrap_auc_roc_ci": bootstrap_metrics["percentile_intervals"]["auc_roc"],
                "bootstrap_tpr_at_1_fpr_ci": bootstrap_metrics["percentile_intervals"]["tpr_at_1_fpr"],
                "bootstrap_tpr_at_10_fpr_ci": bootstrap_metrics["percentile_intervals"]["tpr_at_10_fpr"],
            }
        )

    _write_json(
        summary_path,
        {
            "base_losses_path": str(base_losses_path),
            "runs": runs,
        },
    )


def _prepare_output_root(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _release_model(model: torch.nn.Module) -> None:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _format_float(value: float) -> str:
    return format(float(value), ".17g")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
