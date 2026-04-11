from __future__ import annotations

from csv import DictWriter
from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np

from experiment.filter_train.data import INT_TO_LABEL
from experiment.filter_train.metrics import compute_classification_metrics

from .config import DEFAULT_STAGE3_FHE_CONFIG_PATH, Stage3FheConfig
from .data import (
    PlaintextPredictionArtifact,
    PlaintextRunMetrics,
    compute_plaintext_logits,
    load_and_validate_embedding_artifact,
    load_plaintext_model_parameters,
    load_plaintext_prediction_artifact,
    load_plaintext_run_metrics,
    predict_labels,
    sigmoid,
    validate_prediction_artifact,
)
from .metrics import (
    LATENCY_SAMPLE_COLUMNS,
    LatencySample,
    PredictionComparisonMetrics,
    classification_metrics_from_document,
    compute_prediction_comparison_metrics,
    summarize_latency_samples,
)
from .openfhe_backend import OpenFheCkksScorer
from .openfhe_backend import OpenFheBundlePaths


COMPARISON_COLUMNS = (
    "message_id",
    "template_family",
    "source_type",
    "true_label",
    "plaintext_block_probability",
    "plaintext_predicted_label",
    "fhe_decrypted_logit",
    "fhe_block_probability",
    "fhe_predicted_label",
    "probability_abs_delta",
    "prediction_match",
)


@dataclass(frozen=True, slots=True)
class Stage3FheArtifacts:
    metrics_path: Path
    comparison_path: Path
    latency_summary_path: Path
    latency_samples_path: Path
    context_metadata_path: Path
    compiled_bundle_manifest_path: Path


@dataclass(frozen=True, slots=True)
class Stage3FheRunResult:
    config: Stage3FheConfig
    artifacts: Stage3FheArtifacts
    plaintext_metrics: dict[str, Any]
    fhe_metrics: dict[str, Any]
    comparison_metrics: PredictionComparisonMetrics
    latency_summary: dict[str, dict[str, float]]
    compiled_bundle_reused: bool


def run_stage3_fhe_evaluation(
    config_path: Path | str | None = None,
) -> Stage3FheRunResult:
    config = Stage3FheConfig.from_toml(config_path or DEFAULT_STAGE3_FHE_CONFIG_PATH)
    _set_global_seed(config.seed)

    plaintext_metrics = load_plaintext_run_metrics(
        config.plaintext_artifacts.plaintext_metrics_path
    )
    model_parameters = load_plaintext_model_parameters(
        config.plaintext_artifacts.model_parameters_path
    )
    test_embeddings = load_and_validate_embedding_artifact(
        config.plaintext_artifacts.test_embeddings_path,
        model_parameters=model_parameters,
    )
    val_embeddings = load_and_validate_embedding_artifact(
        config.plaintext_artifacts.val_embeddings_path,
        model_parameters=model_parameters,
    )
    _validate_split_names(test_embeddings_split=test_embeddings.split_name, val_embeddings_split=val_embeddings.split_name)

    plaintext_predictions = load_plaintext_prediction_artifact(
        config.plaintext_artifacts.test_predictions_path
    )
    validate_prediction_artifact(
        plaintext_predictions,
        embeddings=test_embeddings,
        model_parameters=model_parameters,
    )

    plaintext_test_metrics = classification_metrics_from_document(
        plaintext_metrics.test_metrics_document
    )
    _validate_plaintext_baseline(
        plaintext_metrics=plaintext_metrics,
        model_parameters=model_parameters,
        test_embeddings=test_embeddings,
        plaintext_predictions=plaintext_predictions,
        plaintext_test_metrics=plaintext_test_metrics,
    )

    example_count = min(config.benchmark.example_count, len(test_embeddings.message_ids))
    bundle_paths = OpenFheBundlePaths.for_root(config.output_root / "compiled")
    scorer = OpenFheCkksScorer.load_or_create(
        settings=config.fhe,
        model_parameters=model_parameters,
        bundle_paths=bundle_paths,
    )

    fhe_probabilities = np.empty(example_count, dtype=np.float64)
    fhe_predictions = np.empty(example_count, dtype=np.int8)
    latency_samples: list[LatencySample] = []
    comparison_rows: list[dict[str, str | float | int]] = []

    for index in range(example_count):
        end_to_end_start = time.perf_counter()
        decrypted_logit, latency_breakdown = scorer.score_embedding(test_embeddings.embeddings[index])
        block_probability = float(sigmoid(np.asarray([decrypted_logit], dtype=np.float64))[0])
        predicted_label = int(block_probability >= model_parameters.threshold)
        end_to_end_end = time.perf_counter()

        fhe_probabilities[index] = block_probability
        fhe_predictions[index] = predicted_label

        comparison_rows.append(
            {
                "message_id": test_embeddings.message_ids[index],
                "template_family": test_embeddings.template_families[index],
                "source_type": test_embeddings.source_types[index],
                "true_label": test_embeddings.label_names[index],
                "plaintext_block_probability": float(
                    plaintext_predictions.block_probabilities[index]
                ),
                "plaintext_predicted_label": plaintext_predictions.predicted_labels[index],
                "fhe_decrypted_logit": decrypted_logit,
                "fhe_block_probability": block_probability,
                "fhe_predicted_label": INT_TO_LABEL[predicted_label],
                "probability_abs_delta": float(
                    abs(plaintext_predictions.block_probabilities[index] - block_probability)
                ),
                "prediction_match": int(
                    plaintext_predictions.predicted_labels[index] == INT_TO_LABEL[predicted_label]
                ),
            }
        )
        latency_samples.append(
            LatencySample(
                message_id=test_embeddings.message_ids[index],
                encryption_ms=latency_breakdown.encryption_ms,
                scoring_ms=latency_breakdown.scoring_ms,
                decryption_ms=latency_breakdown.decryption_ms,
                end_to_end_ms=(end_to_end_end - end_to_end_start) * 1000.0,
            )
        )

    fhe_test_metrics = compute_classification_metrics(
        test_embeddings.labels[:example_count],
        fhe_predictions,
    )
    comparison_metrics = compute_prediction_comparison_metrics(
        plaintext_probabilities=plaintext_predictions.block_probabilities[:example_count],
        plaintext_predictions=np.asarray(
            [
                1 if label == "BLOCK" else 0
                for label in plaintext_predictions.predicted_labels[:example_count]
            ],
            dtype=np.int8,
        ),
        plaintext_metrics=plaintext_test_metrics,
        fhe_probabilities=fhe_probabilities,
        fhe_predictions=fhe_predictions,
        fhe_metrics=fhe_test_metrics,
    )
    latency_summary = summarize_latency_samples(latency_samples)

    artifacts = Stage3FheArtifacts(
        metrics_path=config.output_root / "stage3_fhe_metrics.json",
        comparison_path=config.output_root / "plaintext_vs_fhe_comparison.csv",
        latency_summary_path=config.output_root / "latency_summary.json",
        latency_samples_path=config.output_root / "latency_samples.csv",
        context_metadata_path=config.output_root / "context_metadata.json",
        compiled_bundle_manifest_path=config.output_root / "compiled_bundle_manifest.json",
    )
    config.output_root.mkdir(parents=True, exist_ok=True)

    _write_csv(artifacts.comparison_path, COMPARISON_COLUMNS, comparison_rows)
    _write_csv(
        artifacts.latency_samples_path,
        LATENCY_SAMPLE_COLUMNS,
        [sample.to_row() for sample in latency_samples],
    )
    _write_json(artifacts.latency_summary_path, latency_summary)
    _write_json(
        artifacts.compiled_bundle_manifest_path,
        _build_compiled_bundle_manifest(
            bundle_paths=bundle_paths,
            model_parameters=model_parameters,
            model_parameters_path=config.plaintext_artifacts.model_parameters_path,
            scorer=scorer,
        ),
    )
    _write_json(
        artifacts.context_metadata_path,
        _build_context_metadata(
            config=config,
            model_parameters=model_parameters,
            example_count=example_count,
            scorer=scorer,
            bundle_paths=bundle_paths,
            compiled_bundle_manifest_path=artifacts.compiled_bundle_manifest_path,
        ),
    )
    _write_json(
        artifacts.metrics_path,
        {
            "config_path": str(config.config_path),
            "plaintext_artifact_paths": {
                "plaintext_metrics_path": str(config.plaintext_artifacts.plaintext_metrics_path),
                "model_parameters_path": str(config.plaintext_artifacts.model_parameters_path),
                "test_embeddings_path": str(config.plaintext_artifacts.test_embeddings_path),
                "val_embeddings_path": str(config.plaintext_artifacts.val_embeddings_path),
                "test_predictions_path": str(config.plaintext_artifacts.test_predictions_path),
            },
            "fhe_library_used": scorer.resolved_parameters.backend,
            "resolved_ckks_parameters": scorer.resolved_parameters.to_document(),
            "compiled_bundle_reused": scorer.reused_existing_bundle,
            "compiled_bundle_manifest_path": str(artifacts.compiled_bundle_manifest_path),
            "plaintext_test_metrics": plaintext_test_metrics.to_document(),
            "fhe_test_metrics": fhe_test_metrics.to_document(),
            **comparison_metrics.to_document(),
            "benchmark_example_count": example_count,
            "comparison_artifact_path": str(artifacts.comparison_path),
            "latency_summary_path": str(artifacts.latency_summary_path),
            "latency_samples_path": str(artifacts.latency_samples_path),
            "context_metadata_path": str(artifacts.context_metadata_path),
        },
    )

    return Stage3FheRunResult(
        config=config,
        artifacts=artifacts,
        plaintext_metrics=plaintext_test_metrics.to_document(),
        fhe_metrics=fhe_test_metrics.to_document(),
        comparison_metrics=comparison_metrics,
        latency_summary=latency_summary,
        compiled_bundle_reused=scorer.reused_existing_bundle,
    )


def _validate_split_names(*, test_embeddings_split: str, val_embeddings_split: str) -> None:
    if test_embeddings_split != "test":
        raise ValueError(
            f"Stage 3 FHE evaluation requires test embeddings, found split {test_embeddings_split!r}."
        )
    if val_embeddings_split != "val":
        raise ValueError(
            f"Stage 3 FHE evaluation requires validation embeddings, found split {val_embeddings_split!r}."
        )


def _validate_plaintext_baseline(
    *,
    plaintext_metrics: PlaintextRunMetrics,
    model_parameters,
    test_embeddings,
    plaintext_predictions: PlaintextPredictionArtifact,
    plaintext_test_metrics,
) -> None:
    if plaintext_metrics.encoder_model_name != model_parameters.encoder_model_name:
        raise ValueError("Saved plaintext metrics and model parameters disagree on encoder_model_name.")
    if plaintext_metrics.embedding_dimension != model_parameters.embedding_dimension:
        raise ValueError("Saved plaintext metrics and model parameters disagree on embedding_dimension.")
    if plaintext_metrics.selected_threshold != model_parameters.threshold:
        raise ValueError("Saved plaintext metrics and model parameters disagree on selected threshold.")
    if plaintext_metrics.selected_threshold != plaintext_predictions.threshold:
        raise ValueError("Saved plaintext metrics and test predictions disagree on selected threshold.")

    plaintext_logits = compute_plaintext_logits(model_parameters, test_embeddings.embeddings)
    recomputed_probabilities = sigmoid(plaintext_logits)
    recomputed_predictions = predict_labels(model_parameters, recomputed_probabilities)

    probability_delta = np.abs(recomputed_probabilities - plaintext_predictions.block_probabilities)
    if float(np.max(probability_delta)) > 1e-12:
        raise ValueError("Saved plaintext predictions do not match the saved linear model parameters.")

    expected_prediction_labels = tuple(
        INT_TO_LABEL[int(value)] for value in recomputed_predictions.tolist()
    )
    if expected_prediction_labels != plaintext_predictions.predicted_labels:
        raise ValueError("Saved plaintext predicted labels do not match the saved linear model parameters.")

    recomputed_metrics = compute_classification_metrics(
        test_embeddings.labels,
        recomputed_predictions,
    )
    if not _classification_metrics_match(recomputed_metrics, plaintext_test_metrics):
        raise ValueError("Saved plaintext test metrics do not match the saved linear model parameters.")


def _classification_metrics_match(left, right) -> bool:
    left_doc = left.to_document()
    right_doc = right.to_document()
    for key in (
        "allow_precision",
        "allow_recall",
        "allow_f1",
        "block_precision",
        "block_recall",
        "block_f1",
        "macro_f1",
        "accuracy",
        "adversarial_block_rate",
        "benign_false_positive_rate",
    ):
        if not np.isclose(float(left_doc[key]), float(right_doc[key]), atol=1e-12, rtol=0.0):
            return False
    return left.confusion_matrix == right.confusion_matrix


def _build_context_metadata(
    *,
    config: Stage3FheConfig,
    model_parameters,
    example_count: int,
    scorer: OpenFheCkksScorer,
    bundle_paths: OpenFheBundlePaths,
    compiled_bundle_manifest_path: Path,
) -> dict[str, Any]:
    return {
        "python_version": sys.version.split()[0],
        "package_versions": {
            package_name: _safe_package_version(package_name)
            for package_name in (
                "fhe-experiment-runtime",
                "numpy",
                "openfhe",
                "scikit-learn",
                "sentence-transformers",
                "torch",
                "transformers",
            )
        },
        "chosen_fhe_backend": scorer.resolved_parameters.backend,
        "resolved_ckks_parameters": scorer.resolved_parameters.to_document(),
        "compiled_bundle_reused": scorer.reused_existing_bundle,
        "compiled_bundle_paths": bundle_paths.to_document(),
        "compiled_bundle_manifest_path": str(compiled_bundle_manifest_path),
        "vector_dimension": model_parameters.embedding_dimension,
        "benchmark_split": config.benchmark.split_name,
        "benchmark_example_count": example_count,
        "threshold": model_parameters.threshold,
        "thresholding_policy": (
            "Decrypt the CKKS linear logit, compute sigmoid(z) in plaintext, "
            "then predict BLOCK iff block_probability >= the saved plaintext threshold."
        ),
    }


def _build_compiled_bundle_manifest(
    *,
    bundle_paths: OpenFheBundlePaths,
    model_parameters,
    model_parameters_path: Path,
    scorer: OpenFheCkksScorer,
) -> dict[str, Any]:
    return {
        "bundle_format": "openfhe_ckks_stage3_v1",
        "bundle_root": str(bundle_paths.root_dir),
        "resolved_ckks_parameters": scorer.resolved_parameters.to_document(),
        "embedding_dimension": model_parameters.embedding_dimension,
        "threshold": model_parameters.threshold,
        "plaintext_model_parameters_path": str(model_parameters_path),
        "plaintext_model_parameters_sha256": _sha256_file(model_parameters_path),
        "thresholding_policy": "local plaintext thresholding after decrypting the CKKS logit",
        "artifacts": {
            "crypto_context": _bundle_file_document(
                bundle_paths.crypto_context_path,
                scope="shared OpenFHE crypto context",
            ),
            "public_key": _bundle_file_document(
                bundle_paths.public_key_path,
                scope="client encryption / server evaluation public key",
            ),
            "secret_key": _bundle_file_document(
                bundle_paths.secret_key_path,
                scope="client-only decryption key",
            ),
            "eval_mult_key": _bundle_file_document(
                bundle_paths.eval_mult_key_path,
                scope="server-side CKKS multiplication evaluation key",
            ),
            "eval_automorphism_key": _bundle_file_document(
                bundle_paths.eval_automorphism_key_path,
                scope="server-side CKKS rotation/sum evaluation key",
            ),
        },
    }


def _safe_package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _bundle_file_document(path: Path, *, scope: str) -> dict[str, str | int]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "scope": scope,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
