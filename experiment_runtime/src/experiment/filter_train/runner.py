from __future__ import annotations

from csv import DictWriter
from dataclasses import dataclass
from importlib import metadata
import json
from pathlib import Path
import random
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch

from .config import (
    DEFAULT_STAGE3_PLAINTEXT_FILTER_CONFIG_PATH,
    Stage3PlaintextFilterConfig,
)
from .data import INT_TO_LABEL, LABEL_TO_INT, Stage3DatasetSplit, load_stage3_datasets
from .embeddings import (
    EmbeddedStage3Split,
    embed_dataset_split,
    load_sentence_encoder,
    write_embedding_artifact,
)
from .metrics import (
    CLASSIFICATION_METRIC_COLUMNS,
    ClassificationMetrics,
    ThresholdSweepResult,
    compare_selection_results,
    compute_classification_metrics,
    select_best_threshold,
    sweep_thresholds,
)


MODEL_SELECTION_COLUMNS = (
    "candidate_c",
    "selected_threshold",
    *CLASSIFICATION_METRIC_COLUMNS,
    "selected_model",
)
THRESHOLD_SWEEP_COLUMNS = (
    "candidate_c",
    "threshold",
    *CLASSIFICATION_METRIC_COLUMNS,
    "is_best_for_c",
    "is_selected_model",
    "is_selected_threshold",
)
TEST_PREDICTION_COLUMNS = (
    "message_id",
    "template_family",
    "source_type",
    "true_label",
    "block_probability",
    "predicted_label",
    "threshold",
)


@dataclass(frozen=True, slots=True)
class Stage3PlaintextArtifacts:
    metrics_path: Path
    model_selection_path: Path
    validation_threshold_sweep_path: Path
    test_predictions_path: Path
    encoder_metadata_path: Path
    train_embeddings_path: Path
    val_embeddings_path: Path
    test_embeddings_path: Path
    model_parameters_path: Path
    logistic_regression_path: Path


@dataclass(frozen=True, slots=True)
class Stage3PlaintextRunResult:
    config: Stage3PlaintextFilterConfig
    output_root: Path
    artifacts: Stage3PlaintextArtifacts
    encoder_model_name: str
    embedding_dimension: int
    selected_c: float
    selected_threshold: float
    validation_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics


@dataclass(frozen=True, slots=True)
class CandidateModelResult:
    c_value: float
    classifier: LogisticRegression
    threshold_sweep: tuple[ThresholdSweepResult, ...]
    best_validation_result: ThresholdSweepResult


def run_stage3_plaintext_training(
    config_path: Path | str | None = None,
) -> Stage3PlaintextRunResult:
    config = Stage3PlaintextFilterConfig.from_toml(
        config_path or DEFAULT_STAGE3_PLAINTEXT_FILTER_CONFIG_PATH
    )
    _set_global_seed(config.seed)

    datasets = load_stage3_datasets(
        train_path=config.datasets.train,
        val_path=config.datasets.val,
        test_path=config.datasets.test,
    )
    output_root = config.output_root
    embeddings_dir = output_root / "embeddings"
    model_dir = output_root / "model"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    encoder, encoder_device, embedding_dimension = load_sentence_encoder(config.encoder)
    embedded_splits = {
        split_name: embed_dataset_split(
            encoder,
            dataset,
            settings=config.encoder,
        )
        for split_name, dataset in datasets.items()
    }

    train_embeddings_path = embeddings_dir / "train_embeddings.npz"
    val_embeddings_path = embeddings_dir / "val_embeddings.npz"
    test_embeddings_path = embeddings_dir / "test_embeddings.npz"
    write_embedding_artifact(
        train_embeddings_path,
        embedded_splits["train"],
        encoder_model_name=config.encoder.model_name,
        normalize_embeddings=config.encoder.normalize_embeddings,
    )
    write_embedding_artifact(
        val_embeddings_path,
        embedded_splits["val"],
        encoder_model_name=config.encoder.model_name,
        normalize_embeddings=config.encoder.normalize_embeddings,
    )
    write_embedding_artifact(
        test_embeddings_path,
        embedded_splits["test"],
        encoder_model_name=config.encoder.model_name,
        normalize_embeddings=config.encoder.normalize_embeddings,
    )

    candidate_results: list[CandidateModelResult] = []
    for c_value in config.logistic_regression.candidate_c_values:
        classifier = LogisticRegression(
            C=c_value,
            solver=config.logistic_regression.solver,
            max_iter=config.logistic_regression.max_iter,
            random_state=config.seed,
        )
        classifier.fit(
            embedded_splits["train"].embeddings,
            embedded_splits["train"].dataset.labels,
        )
        validation_probabilities = _block_probabilities(
            classifier,
            embedded_splits["val"].embeddings,
        )
        threshold_sweep = sweep_thresholds(
            embedded_splits["val"].dataset.labels,
            validation_probabilities,
        )
        candidate_results.append(
            CandidateModelResult(
                c_value=float(c_value),
                classifier=classifier,
                threshold_sweep=threshold_sweep,
                best_validation_result=select_best_threshold(threshold_sweep),
            )
        )

    selected_candidate = _select_best_candidate(candidate_results)
    selected_threshold = selected_candidate.best_validation_result.threshold
    test_probabilities = _block_probabilities(
        selected_candidate.classifier,
        embedded_splits["test"].embeddings,
    )
    test_predictions = np.asarray(test_probabilities >= selected_threshold, dtype=np.int8)
    test_metrics = compute_classification_metrics(
        embedded_splits["test"].dataset.labels,
        test_predictions,
    )

    artifacts = Stage3PlaintextArtifacts(
        metrics_path=output_root / "stage3_plaintext_metrics.json",
        model_selection_path=output_root / "model_selection.csv",
        validation_threshold_sweep_path=output_root / "validation_threshold_sweep.csv",
        test_predictions_path=output_root / "test_predictions.csv",
        encoder_metadata_path=output_root / "encoder_metadata.json",
        train_embeddings_path=train_embeddings_path,
        val_embeddings_path=val_embeddings_path,
        test_embeddings_path=test_embeddings_path,
        model_parameters_path=model_dir / "model_parameters.json",
        logistic_regression_path=model_dir / "logistic_regression.joblib",
    )

    encoder_metadata = _build_encoder_metadata(
        config=config,
        output_root=output_root,
        embedding_dimension=embedding_dimension,
        encoder_device=encoder_device,
        embedded_splits=embedded_splits,
    )
    _write_json(artifacts.encoder_metadata_path, encoder_metadata)

    _write_model_selection_csv(artifacts.model_selection_path, candidate_results, selected_candidate)
    _write_validation_threshold_sweep_csv(
        artifacts.validation_threshold_sweep_path,
        candidate_results,
        selected_candidate,
    )
    _write_test_predictions_csv(
        artifacts.test_predictions_path,
        dataset=embedded_splits["test"].dataset,
        block_probabilities=test_probabilities,
        threshold=selected_threshold,
    )
    _write_json(
        artifacts.model_parameters_path,
        _build_model_parameters_document(
            config=config,
            classifier=selected_candidate.classifier,
            embedding_dimension=embedding_dimension,
            threshold=selected_threshold,
        ),
    )
    joblib.dump(selected_candidate.classifier, artifacts.logistic_regression_path)
    _write_json(
        artifacts.metrics_path,
        {
            "config_path": str(config.config_path),
            "output_root": str(output_root),
            "encoder_model_name": config.encoder.model_name,
            "embedding_dimension": embedding_dimension,
            "normalize_embeddings": config.encoder.normalize_embeddings,
            "selected_c": selected_candidate.c_value,
            "selected_threshold": selected_threshold,
            "validation_metrics": selected_candidate.best_validation_result.metrics.to_document(),
            "test_metrics": test_metrics.to_document(),
        },
    )

    return Stage3PlaintextRunResult(
        config=config,
        output_root=output_root,
        artifacts=artifacts,
        encoder_model_name=config.encoder.model_name,
        embedding_dimension=embedding_dimension,
        selected_c=selected_candidate.c_value,
        selected_threshold=selected_threshold,
        validation_metrics=selected_candidate.best_validation_result.metrics,
        test_metrics=test_metrics,
    )


def _select_best_candidate(candidate_results: list[CandidateModelResult]) -> CandidateModelResult:
    if not candidate_results:
        raise ValueError("Stage 3 plaintext training requires at least one candidate model.")

    best = candidate_results[0]
    for candidate in candidate_results[1:]:
        comparison = compare_selection_results(
            candidate.best_validation_result,
            best.best_validation_result,
        )
        if comparison > 0 or (
            comparison == 0 and candidate.c_value < best.c_value
        ):
            best = candidate
    return best


def _block_probabilities(classifier: LogisticRegression, embeddings: np.ndarray) -> np.ndarray:
    class_index = list(classifier.classes_).index(LABEL_TO_INT["BLOCK"])
    return np.asarray(classifier.predict_proba(embeddings)[:, class_index], dtype=np.float64)


def _write_model_selection_csv(
    path: Path,
    candidate_results: list[CandidateModelResult],
    selected_candidate: CandidateModelResult,
) -> None:
    rows = []
    for candidate in candidate_results:
        rows.append(
            {
                "candidate_c": candidate.c_value,
                "selected_threshold": candidate.best_validation_result.threshold,
                **candidate.best_validation_result.metrics.to_row(),
                "selected_model": int(candidate.c_value == selected_candidate.c_value),
            }
        )
    _write_csv(path, MODEL_SELECTION_COLUMNS, rows)


def _write_validation_threshold_sweep_csv(
    path: Path,
    candidate_results: list[CandidateModelResult],
    selected_candidate: CandidateModelResult,
) -> None:
    rows: list[dict[str, Any]] = []
    for candidate in candidate_results:
        for threshold_result in candidate.threshold_sweep:
            rows.append(
                {
                    "candidate_c": candidate.c_value,
                    "threshold": threshold_result.threshold,
                    **threshold_result.metrics.to_row(),
                    "is_best_for_c": int(
                        threshold_result.threshold == candidate.best_validation_result.threshold
                    ),
                    "is_selected_model": int(candidate.c_value == selected_candidate.c_value),
                    "is_selected_threshold": int(
                        candidate.c_value == selected_candidate.c_value
                        and threshold_result.threshold == selected_candidate.best_validation_result.threshold
                    ),
                }
            )
    _write_csv(path, THRESHOLD_SWEEP_COLUMNS, rows)


def _write_test_predictions_csv(
    path: Path,
    *,
    dataset: Stage3DatasetSplit,
    block_probabilities: np.ndarray,
    threshold: float,
) -> None:
    rows = []
    for row, block_probability in zip(dataset.rows, block_probabilities, strict=True):
        predicted_label = INT_TO_LABEL[int(block_probability >= threshold)]
        rows.append(
            {
                "message_id": row.message_id,
                "template_family": row.template_family,
                "source_type": row.source_type,
                "true_label": row.label,
                "block_probability": float(block_probability),
                "predicted_label": predicted_label,
                "threshold": threshold,
            }
        )
    _write_csv(path, TEST_PREDICTION_COLUMNS, rows)


def _build_model_parameters_document(
    *,
    config: Stage3PlaintextFilterConfig,
    classifier: LogisticRegression,
    embedding_dimension: int,
    threshold: float,
) -> dict[str, Any]:
    if classifier.coef_.shape != (1, embedding_dimension):
        raise ValueError(
            "Stage 3 logistic regression parameters must be rank-2 with one binary row, "
            f"found {classifier.coef_.shape}."
        )

    return {
        "encoder_model_name": config.encoder.model_name,
        "embedding_dimension": embedding_dimension,
        "normalize_embeddings": config.encoder.normalize_embeddings,
        "classifier_type": "sklearn.linear_model.LogisticRegression",
        "solver": config.logistic_regression.solver,
        "c_value": float(classifier.C),
        "classes": [INT_TO_LABEL[int(value)] for value in classifier.classes_.tolist()],
        "class_mapping": {
            "ALLOW": LABEL_TO_INT["ALLOW"],
            "BLOCK": LABEL_TO_INT["BLOCK"],
        },
        "weights": [float(value) for value in classifier.coef_[0].tolist()],
        "intercept": float(classifier.intercept_[0]),
        "threshold": float(threshold),
        "score_definition": "probability of BLOCK",
        "decision_rule": config.threshold_selection.decision_rule,
        "threshold_selection_split": config.threshold_selection.selection_split,
    }


def _build_encoder_metadata(
    *,
    config: Stage3PlaintextFilterConfig,
    output_root: Path,
    embedding_dimension: int,
    encoder_device: str,
    embedded_splits: dict[str, EmbeddedStage3Split],
) -> dict[str, Any]:
    return {
        "model_name": config.encoder.model_name,
        "embedding_dimension": embedding_dimension,
        "normalize_embeddings": config.encoder.normalize_embeddings,
        "batch_size": config.encoder.batch_size,
        "device": encoder_device,
        "package_versions": {
            "sentence_transformers": metadata.version("sentence-transformers"),
            "scikit_learn": metadata.version("scikit-learn"),
            "torch": metadata.version("torch"),
            "transformers": metadata.version("transformers"),
        },
        "dataset_paths": {
            "train": str(config.datasets.train),
            "val": str(config.datasets.val),
            "test": str(config.datasets.test),
        },
        "embedding_artifacts": {
            split_name: str(output_root / "embeddings" / f"{split_name}_embeddings.npz")
            for split_name in ("train", "val", "test")
        },
        "row_counts": {
            split_name: len(embedded_split.dataset.rows)
            for split_name, embedded_split in embedded_splits.items()
        },
    }


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
