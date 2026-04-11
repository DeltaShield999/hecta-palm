from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


CLASSIFICATION_METRIC_COLUMNS = (
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
    "true_allow_pred_allow",
    "true_allow_pred_block",
    "true_block_pred_allow",
    "true_block_pred_block",
)

SELECTION_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    allow_precision: float
    allow_recall: float
    allow_f1: float
    block_precision: float
    block_recall: float
    block_f1: float
    macro_f1: float
    accuracy: float
    adversarial_block_rate: float
    benign_false_positive_rate: float
    true_allow_pred_allow: int
    true_allow_pred_block: int
    true_block_pred_allow: int
    true_block_pred_block: int

    @property
    def confusion_matrix(self) -> dict[str, int]:
        return {
            "true_ALLOW_pred_ALLOW": self.true_allow_pred_allow,
            "true_ALLOW_pred_BLOCK": self.true_allow_pred_block,
            "true_BLOCK_pred_ALLOW": self.true_block_pred_allow,
            "true_BLOCK_pred_BLOCK": self.true_block_pred_block,
        }

    def to_document(self) -> dict[str, float | int | dict[str, int]]:
        return {
            **self.to_row(),
            "confusion_matrix": self.confusion_matrix,
        }

    def to_row(self) -> dict[str, float | int]:
        return {
            "allow_precision": self.allow_precision,
            "allow_recall": self.allow_recall,
            "allow_f1": self.allow_f1,
            "block_precision": self.block_precision,
            "block_recall": self.block_recall,
            "block_f1": self.block_f1,
            "macro_f1": self.macro_f1,
            "accuracy": self.accuracy,
            "adversarial_block_rate": self.adversarial_block_rate,
            "benign_false_positive_rate": self.benign_false_positive_rate,
            "true_allow_pred_allow": self.true_allow_pred_allow,
            "true_allow_pred_block": self.true_allow_pred_block,
            "true_block_pred_allow": self.true_block_pred_allow,
            "true_block_pred_block": self.true_block_pred_block,
        }


@dataclass(frozen=True, slots=True)
class ThresholdSweepResult:
    threshold: float
    metrics: ClassificationMetrics

    def to_row(self) -> dict[str, float | int]:
        return {
            "threshold": self.threshold,
            **self.metrics.to_row(),
        }


def compute_classification_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> ClassificationMetrics:
    true_labels = np.asarray(true_labels, dtype=np.int8)
    predicted_labels = np.asarray(predicted_labels, dtype=np.int8)

    true_allow_pred_allow = int(np.sum((true_labels == 0) & (predicted_labels == 0)))
    true_allow_pred_block = int(np.sum((true_labels == 0) & (predicted_labels == 1)))
    true_block_pred_allow = int(np.sum((true_labels == 1) & (predicted_labels == 0)))
    true_block_pred_block = int(np.sum((true_labels == 1) & (predicted_labels == 1)))

    allow_precision = _safe_divide(true_allow_pred_allow, true_allow_pred_allow + true_block_pred_allow)
    allow_recall = _safe_divide(true_allow_pred_allow, true_allow_pred_allow + true_allow_pred_block)
    allow_f1 = _f1(allow_precision, allow_recall)

    block_precision = _safe_divide(true_block_pred_block, true_block_pred_block + true_allow_pred_block)
    block_recall = _safe_divide(true_block_pred_block, true_block_pred_block + true_block_pred_allow)
    block_f1 = _f1(block_precision, block_recall)

    total = len(true_labels)
    accuracy = _safe_divide(true_allow_pred_allow + true_block_pred_block, total)
    benign_false_positive_rate = _safe_divide(
        true_allow_pred_block,
        true_allow_pred_allow + true_allow_pred_block,
    )

    return ClassificationMetrics(
        allow_precision=allow_precision,
        allow_recall=allow_recall,
        allow_f1=allow_f1,
        block_precision=block_precision,
        block_recall=block_recall,
        block_f1=block_f1,
        macro_f1=(allow_f1 + block_f1) / 2.0,
        accuracy=accuracy,
        adversarial_block_rate=block_recall,
        benign_false_positive_rate=benign_false_positive_rate,
        true_allow_pred_allow=true_allow_pred_allow,
        true_allow_pred_block=true_allow_pred_block,
        true_block_pred_allow=true_block_pred_allow,
        true_block_pred_block=true_block_pred_block,
    )


def sweep_thresholds(
    true_labels: np.ndarray,
    block_probabilities: np.ndarray,
) -> tuple[ThresholdSweepResult, ...]:
    thresholds = np.unique(np.asarray(block_probabilities, dtype=np.float64))
    results: list[ThresholdSweepResult] = []
    for threshold in thresholds.tolist():
        predicted_labels = np.asarray(block_probabilities >= threshold, dtype=np.int8)
        results.append(
            ThresholdSweepResult(
                threshold=float(threshold),
                metrics=compute_classification_metrics(true_labels, predicted_labels),
            )
        )
    return tuple(results)


def select_best_threshold(results: tuple[ThresholdSweepResult, ...]) -> ThresholdSweepResult:
    if not results:
        raise ValueError("Threshold selection requires at least one validation result.")

    best = results[0]
    for candidate in results[1:]:
        if compare_selection_results(candidate, best) > 0:
            best = candidate
    return best


def compare_selection_results(left: ThresholdSweepResult, right: ThresholdSweepResult) -> int:
    for left_value, right_value, prefer_smaller in (
        (left.metrics.macro_f1, right.metrics.macro_f1, False),
        (left.metrics.block_recall, right.metrics.block_recall, False),
        (left.metrics.benign_false_positive_rate, right.metrics.benign_false_positive_rate, True),
        (left.threshold, right.threshold, True),
    ):
        comparison = _compare_scalar(left_value, right_value, prefer_smaller=prefer_smaller)
        if comparison != 0:
            return comparison
    return 0


def _compare_scalar(left: float, right: float, *, prefer_smaller: bool) -> int:
    if math.isclose(left, right, rel_tol=0.0, abs_tol=SELECTION_EPSILON):
        return 0
    if prefer_smaller:
        return 1 if left < right else -1
    return 1 if left > right else -1


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _f1(precision: float, recall: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))
