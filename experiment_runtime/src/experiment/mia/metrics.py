from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence


LOW_FPR_TARGETS = (0.01, 0.10)


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    max_fpr: float
    threshold: float
    tpr: float


@dataclass(frozen=True, slots=True)
class RocMetrics:
    thresholds: tuple[float, ...]
    fpr: tuple[float, ...]
    tpr: tuple[float, ...]
    auc_roc: float
    operating_points: tuple[OperatingPoint, ...]

    def operating_point(self, max_fpr: float) -> OperatingPoint:
        for point in self.operating_points:
            if point.max_fpr == max_fpr:
                return point
        raise KeyError(max_fpr)


def compute_membership_score(loss_base: float, loss_ft: float) -> float:
    if loss_base <= 0.0 or loss_ft <= 0.0:
        raise ValueError("Per-example losses must be positive to compute membership_score.")
    return loss_base / loss_ft


def compute_roc_metrics(labels: Sequence[int], scores: Sequence[float]) -> RocMetrics:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length.")
    if not labels:
        raise ValueError("labels and scores must be non-empty.")

    positive_count = sum(1 for label in labels if label == 1)
    negative_count = len(labels) - positive_count
    if positive_count <= 0 or negative_count <= 0:
        raise ValueError("ROC metrics require at least one positive and one negative example.")

    sorted_pairs = sorted(zip(scores, labels, strict=True), key=lambda item: item[0], reverse=True)

    max_score = max(scores)
    start_threshold = math.nextafter(max_score, math.inf)
    thresholds = [start_threshold]
    tpr_values = [0.0]
    fpr_values = [0.0]

    true_positives = 0
    false_positives = 0
    index = 0
    while index < len(sorted_pairs):
        threshold = float(sorted_pairs[index][0])
        while index < len(sorted_pairs) and sorted_pairs[index][0] == threshold:
            label = sorted_pairs[index][1]
            if label == 1:
                true_positives += 1
            else:
                false_positives += 1
            index += 1

        thresholds.append(threshold)
        tpr_values.append(true_positives / positive_count)
        fpr_values.append(false_positives / negative_count)

    auc_roc = _trapezoid_area(fpr_values, tpr_values)
    operating_points = tuple(
        _select_operating_point(
            thresholds=thresholds,
            fpr_values=fpr_values,
            tpr_values=tpr_values,
            max_fpr=max_fpr,
        )
        for max_fpr in LOW_FPR_TARGETS
    )
    return RocMetrics(
        thresholds=tuple(thresholds),
        fpr=tuple(fpr_values),
        tpr=tuple(tpr_values),
        auc_roc=auc_roc,
        operating_points=operating_points,
    )


def compute_bootstrap_intervals(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, object]:
    if replicates <= 0:
        raise ValueError("replicates must be positive.")

    positives = [float(score) for label, score in zip(labels, scores, strict=True) if label == 1]
    negatives = [float(score) for label, score in zip(labels, scores, strict=True) if label == 0]
    if not positives or not negatives:
        raise ValueError("Bootstrap metrics require at least one positive and one negative example.")

    rng = random.Random(seed)
    auc_samples: list[float] = []
    tpr_at_1_samples: list[float] = []
    tpr_at_10_samples: list[float] = []
    for _ in range(replicates):
        sampled_scores = [positives[rng.randrange(len(positives))] for _ in range(len(positives))]
        sampled_scores.extend(negatives[rng.randrange(len(negatives))] for _ in range(len(negatives)))
        sampled_labels = [1] * len(positives) + [0] * len(negatives)
        metrics = compute_roc_metrics(sampled_labels, sampled_scores)
        auc_samples.append(metrics.auc_roc)
        tpr_at_1_samples.append(metrics.operating_point(0.01).tpr)
        tpr_at_10_samples.append(metrics.operating_point(0.10).tpr)

    lower_percentile = (1.0 - confidence_level) / 2.0
    upper_percentile = 1.0 - lower_percentile
    return {
        "bootstrap_replicates": replicates,
        "confidence_level": confidence_level,
        "interval_method": "percentile",
        "seed": seed,
        "stratified_by_label": True,
        "percentile_intervals": {
            "auc_roc": {
                "lower": _percentile(auc_samples, lower_percentile),
                "upper": _percentile(auc_samples, upper_percentile),
            },
            "tpr_at_1_fpr": {
                "lower": _percentile(tpr_at_1_samples, lower_percentile),
                "upper": _percentile(tpr_at_1_samples, upper_percentile),
            },
            "tpr_at_10_fpr": {
                "lower": _percentile(tpr_at_10_samples, lower_percentile),
                "upper": _percentile(tpr_at_10_samples, upper_percentile),
            },
        },
    }


def _select_operating_point(
    *,
    thresholds: Sequence[float],
    fpr_values: Sequence[float],
    tpr_values: Sequence[float],
    max_fpr: float,
) -> OperatingPoint:
    selected_index = 0
    for index, fpr in enumerate(fpr_values):
        if fpr <= max_fpr:
            selected_index = index
        else:
            break
    return OperatingPoint(
        max_fpr=max_fpr,
        threshold=float(thresholds[selected_index]),
        tpr=float(tpr_values[selected_index]),
    )


def _trapezoid_area(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    area = 0.0
    for index in range(1, len(x_values)):
        area += (x_values[index] - x_values[index - 1]) * (
            y_values[index] + y_values[index - 1]
        ) / 2.0
    return area


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("values must be non-empty.")
    if len(values) == 1:
        return float(values[0])

    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * percentile
    lower_index = math.floor(index)
    upper_index = math.ceil(index)
    if lower_index == upper_index:
        return ordered[lower_index]

    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    weight = index - lower_index
    return lower_value + (upper_value - lower_value) * weight
