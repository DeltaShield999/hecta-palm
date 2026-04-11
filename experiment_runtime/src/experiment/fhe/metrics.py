from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from experiment.filter_train.metrics import ClassificationMetrics


LATENCY_SAMPLE_COLUMNS = (
    "message_id",
    "encryption_ms",
    "scoring_ms",
    "decryption_ms",
    "end_to_end_ms",
)


@dataclass(frozen=True, slots=True)
class PredictionComparisonMetrics:
    prediction_match_rate: float
    plaintext_vs_fhe_accuracy_delta: float
    mean_abs_probability_delta: float
    max_abs_probability_delta: float

    def to_document(self) -> dict[str, float]:
        return {
            "prediction_match_rate": self.prediction_match_rate,
            "plaintext_vs_fhe_accuracy_delta": self.plaintext_vs_fhe_accuracy_delta,
            "mean_abs_probability_delta": self.mean_abs_probability_delta,
            "max_abs_probability_delta": self.max_abs_probability_delta,
        }


@dataclass(frozen=True, slots=True)
class LatencyBreakdown:
    encryption_ms: float
    scoring_ms: float
    decryption_ms: float


@dataclass(frozen=True, slots=True)
class LatencySample:
    message_id: str
    encryption_ms: float
    scoring_ms: float
    decryption_ms: float
    end_to_end_ms: float

    def to_row(self) -> dict[str, str | float]:
        return {
            "message_id": self.message_id,
            "encryption_ms": self.encryption_ms,
            "scoring_ms": self.scoring_ms,
            "decryption_ms": self.decryption_ms,
            "end_to_end_ms": self.end_to_end_ms,
        }


def classification_metrics_from_document(document: dict[str, object]) -> ClassificationMetrics:
    return ClassificationMetrics(
        allow_precision=float(document["allow_precision"]),
        allow_recall=float(document["allow_recall"]),
        allow_f1=float(document["allow_f1"]),
        block_precision=float(document["block_precision"]),
        block_recall=float(document["block_recall"]),
        block_f1=float(document["block_f1"]),
        macro_f1=float(document["macro_f1"]),
        accuracy=float(document["accuracy"]),
        adversarial_block_rate=float(document["adversarial_block_rate"]),
        benign_false_positive_rate=float(document["benign_false_positive_rate"]),
        true_allow_pred_allow=int(document["true_allow_pred_allow"]),
        true_allow_pred_block=int(document["true_allow_pred_block"]),
        true_block_pred_allow=int(document["true_block_pred_allow"]),
        true_block_pred_block=int(document["true_block_pred_block"]),
    )


def compute_prediction_comparison_metrics(
    *,
    plaintext_probabilities: np.ndarray,
    plaintext_predictions: np.ndarray,
    plaintext_metrics: ClassificationMetrics,
    fhe_probabilities: np.ndarray,
    fhe_predictions: np.ndarray,
    fhe_metrics: ClassificationMetrics,
) -> PredictionComparisonMetrics:
    plaintext_probabilities = np.asarray(plaintext_probabilities, dtype=np.float64)
    plaintext_predictions = np.asarray(plaintext_predictions, dtype=np.int8)
    fhe_probabilities = np.asarray(fhe_probabilities, dtype=np.float64)
    fhe_predictions = np.asarray(fhe_predictions, dtype=np.int8)
    if plaintext_probabilities.shape != fhe_probabilities.shape:
        raise ValueError("Plaintext and FHE probability arrays must have the same shape.")
    if plaintext_predictions.shape != fhe_predictions.shape:
        raise ValueError("Plaintext and FHE prediction arrays must have the same shape.")

    probability_abs_delta = np.abs(plaintext_probabilities - fhe_probabilities)
    prediction_match = np.asarray(plaintext_predictions == fhe_predictions, dtype=np.float64)
    return PredictionComparisonMetrics(
        prediction_match_rate=float(np.mean(prediction_match)),
        plaintext_vs_fhe_accuracy_delta=float(
            abs(plaintext_metrics.accuracy - fhe_metrics.accuracy)
        ),
        mean_abs_probability_delta=float(np.mean(probability_abs_delta)),
        max_abs_probability_delta=float(np.max(probability_abs_delta)),
    )


def summarize_latency_samples(samples: Iterable[LatencySample]) -> dict[str, dict[str, float]]:
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("Latency summary requires at least one sample.")

    return {
        "encryption_ms": _summarize_metric(sample.encryption_ms for sample in sample_list),
        "scoring_ms": _summarize_metric(sample.scoring_ms for sample in sample_list),
        "decryption_ms": _summarize_metric(sample.decryption_ms for sample in sample_list),
        "end_to_end_ms": _summarize_metric(sample.end_to_end_ms for sample in sample_list),
    }


def _summarize_metric(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }
