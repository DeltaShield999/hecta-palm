from __future__ import annotations

from csv import DictReader
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiment.filter_train.config import PINNED_STAGE3_ENCODER_MODEL_NAME
from experiment.filter_train.data import INT_TO_LABEL
from experiment.filter_train.embeddings import (
    EXPECTED_STAGE3_EMBEDDING_DIMENSION,
    StoredEmbeddingArtifact,
    load_embedding_artifact,
)


PREDICTED_BLOCK_LABEL = "BLOCK"
PREDICTED_ALLOW_LABEL = "ALLOW"


@dataclass(frozen=True, slots=True)
class PlaintextModelParameters:
    encoder_model_name: str
    embedding_dimension: int
    normalize_embeddings: bool
    classes: tuple[str, ...]
    class_mapping: dict[str, int]
    weights: np.ndarray
    intercept: float
    threshold: float
    score_definition: str
    decision_rule: str
    threshold_selection_split: str


@dataclass(frozen=True, slots=True)
class PlaintextRunMetrics:
    config_path: str
    output_root: str
    encoder_model_name: str
    embedding_dimension: int
    normalize_embeddings: bool
    selected_c: float
    selected_threshold: float
    validation_metrics_document: dict[str, Any]
    test_metrics_document: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PlaintextPredictionArtifact:
    message_ids: tuple[str, ...]
    template_families: tuple[str, ...]
    source_types: tuple[str, ...]
    true_labels: tuple[str, ...]
    block_probabilities: np.ndarray
    predicted_labels: tuple[str, ...]
    threshold: float


def load_plaintext_model_parameters(path: Path) -> PlaintextModelParameters:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)

    weights = np.asarray(document["weights"], dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError("Stage 3 plaintext model weights must be rank-1.")
    if weights.shape[0] != EXPECTED_STAGE3_EMBEDDING_DIMENSION:
        raise ValueError(
            "Stage 3 plaintext model weights must match the pinned 384-dimensional encoder, "
            f"found {weights.shape[0]}."
        )
    if document["encoder_model_name"] != PINNED_STAGE3_ENCODER_MODEL_NAME:
        raise ValueError(
            "Stage 3 plaintext model parameters must use the pinned encoder "
            f"{PINNED_STAGE3_ENCODER_MODEL_NAME}, found "
            f"{document['encoder_model_name']}."
        )
    if int(document["embedding_dimension"]) != EXPECTED_STAGE3_EMBEDDING_DIMENSION:
        raise ValueError(
            "Stage 3 plaintext model parameters must record embedding_dimension = 384."
        )
    if not bool(document["normalize_embeddings"]):
        raise ValueError("Stage 3 plaintext model parameters must use normalized embeddings.")
    if tuple(document["classes"]) != (PREDICTED_ALLOW_LABEL, PREDICTED_BLOCK_LABEL):
        raise ValueError(
            "Stage 3 plaintext model parameters must use classes ['ALLOW', 'BLOCK']."
        )
    if dict(document["class_mapping"]) != {PREDICTED_ALLOW_LABEL: 0, PREDICTED_BLOCK_LABEL: 1}:
        raise ValueError(
            "Stage 3 plaintext model parameters must use class_mapping {'ALLOW': 0, 'BLOCK': 1}."
        )
    threshold = float(document["threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"Stage 3 plaintext threshold must be in (0, 1), found {threshold}.")
    if document["threshold_selection_split"] != "val":
        raise ValueError("Stage 3 plaintext threshold must have been selected on validation only.")

    return PlaintextModelParameters(
        encoder_model_name=str(document["encoder_model_name"]),
        embedding_dimension=int(document["embedding_dimension"]),
        normalize_embeddings=bool(document["normalize_embeddings"]),
        classes=tuple(str(value) for value in document["classes"]),
        class_mapping={str(key): int(value) for key, value in document["class_mapping"].items()},
        weights=weights,
        intercept=float(document["intercept"]),
        threshold=threshold,
        score_definition=str(document["score_definition"]),
        decision_rule=str(document["decision_rule"]),
        threshold_selection_split=str(document["threshold_selection_split"]),
    )


def load_plaintext_run_metrics(path: Path) -> PlaintextRunMetrics:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)

    return PlaintextRunMetrics(
        config_path=str(document["config_path"]),
        output_root=str(document["output_root"]),
        encoder_model_name=str(document["encoder_model_name"]),
        embedding_dimension=int(document["embedding_dimension"]),
        normalize_embeddings=bool(document["normalize_embeddings"]),
        selected_c=float(document["selected_c"]),
        selected_threshold=float(document["selected_threshold"]),
        validation_metrics_document=dict(document["validation_metrics"]),
        test_metrics_document=dict(document["test_metrics"]),
    )


def load_plaintext_prediction_artifact(path: Path) -> PlaintextPredictionArtifact:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Stage 3 plaintext prediction artifact is empty: {path}")

    thresholds = {float(row["threshold"]) for row in rows}
    if len(thresholds) != 1:
        raise ValueError("Stage 3 plaintext prediction artifact must use a single threshold.")

    return PlaintextPredictionArtifact(
        message_ids=tuple(row["message_id"] for row in rows),
        template_families=tuple(row["template_family"] for row in rows),
        source_types=tuple(row["source_type"] for row in rows),
        true_labels=tuple(row["true_label"] for row in rows),
        block_probabilities=np.asarray(
            [float(row["block_probability"]) for row in rows],
            dtype=np.float64,
        ),
        predicted_labels=tuple(row["predicted_label"] for row in rows),
        threshold=float(rows[0]["threshold"]),
    )


def load_and_validate_embedding_artifact(
    path: Path,
    *,
    model_parameters: PlaintextModelParameters,
) -> StoredEmbeddingArtifact:
    artifact = load_embedding_artifact(path)
    if artifact.encoder_model_name != model_parameters.encoder_model_name:
        raise ValueError(
            f"Embedding artifact {path} uses encoder {artifact.encoder_model_name!r}; expected "
            f"{model_parameters.encoder_model_name!r}."
        )
    if artifact.embeddings.ndim != 2:
        raise ValueError(f"Embedding artifact {path} must be rank-2.")
    if artifact.embeddings.shape[1] != model_parameters.embedding_dimension:
        raise ValueError(
            f"Embedding artifact {path} dimension {artifact.embeddings.shape[1]} does not match "
            f"plaintext model dimension {model_parameters.embedding_dimension}."
        )
    if not artifact.normalize_embeddings or not model_parameters.normalize_embeddings:
        raise ValueError("Stage 3 FHE evaluation requires normalized plaintext embeddings.")
    if len(artifact.message_ids) != artifact.embeddings.shape[0]:
        raise ValueError(f"Embedding artifact {path} message_ids length does not match embeddings.")
    if len(artifact.label_names) != artifact.embeddings.shape[0]:
        raise ValueError(f"Embedding artifact {path} label_names length does not match embeddings.")
    if len(artifact.template_families) != artifact.embeddings.shape[0]:
        raise ValueError(
            f"Embedding artifact {path} template_families length does not match embeddings."
        )
    if len(artifact.source_types) != artifact.embeddings.shape[0]:
        raise ValueError(f"Embedding artifact {path} source_types length does not match embeddings.")

    expected_label_names = tuple(INT_TO_LABEL[int(value)] for value in artifact.labels.tolist())
    if artifact.label_names != expected_label_names:
        raise ValueError(f"Embedding artifact {path} label_names do not match stored integer labels.")
    return artifact


def validate_prediction_artifact(
    artifact: PlaintextPredictionArtifact,
    *,
    embeddings: StoredEmbeddingArtifact,
    model_parameters: PlaintextModelParameters,
) -> None:
    if artifact.message_ids != embeddings.message_ids:
        raise ValueError("Plaintext prediction artifact message_id order must match test embeddings.")
    if artifact.template_families != embeddings.template_families:
        raise ValueError(
            "Plaintext prediction artifact template_family order must match test embeddings."
        )
    if artifact.source_types != embeddings.source_types:
        raise ValueError("Plaintext prediction artifact source_type order must match test embeddings.")
    if artifact.true_labels != embeddings.label_names:
        raise ValueError("Plaintext prediction artifact true_label order must match test embeddings.")
    if artifact.block_probabilities.shape[0] != embeddings.embeddings.shape[0]:
        raise ValueError("Plaintext prediction artifact probability length must match test embeddings.")
    if artifact.threshold != model_parameters.threshold:
        raise ValueError(
            "Plaintext prediction artifact threshold must match saved plaintext model parameters."
        )


def compute_plaintext_logits(
    model_parameters: PlaintextModelParameters,
    embeddings: np.ndarray,
) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2:
        raise ValueError("Plaintext logits require a rank-2 embedding matrix.")
    if embeddings.shape[1] != model_parameters.weights.shape[0]:
        raise ValueError(
            f"Embedding dimension {embeddings.shape[1]} does not match plaintext model "
            f"dimension {model_parameters.weights.shape[0]}."
        )
    return np.asarray(embeddings @ model_parameters.weights + model_parameters.intercept, dtype=np.float64)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    positive = logits >= 0.0
    output = np.empty_like(logits, dtype=np.float64)
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    output[~positive] = exp_logits / (1.0 + exp_logits)
    return output


def predict_labels(
    model_parameters: PlaintextModelParameters,
    block_probabilities: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        np.asarray(block_probabilities, dtype=np.float64) >= model_parameters.threshold,
        dtype=np.int8,
    )
