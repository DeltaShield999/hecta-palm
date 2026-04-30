from __future__ import annotations

from math import sqrt
from statistics import NormalDist
from typing import Any


DEFAULT_CONFIDENCE_LEVEL = 0.95
WILSON_METHOD = "wilson"


def proportion_estimate(numerator: int, denominator: int) -> float | None:
    numerator, denominator = _validate_counts(numerator, denominator)
    if denominator == 0:
        return None
    return numerator / denominator


def wilson_interval(
    numerator: int,
    denominator: int,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    numerator, denominator = _validate_counts(numerator, denominator)
    _validate_confidence_level(confidence_level)

    if denominator == 0:
        lower: float | None = None
        upper: float | None = None
    else:
        z_score = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
        z_squared = z_score * z_score
        point_estimate = numerator / denominator
        denominator_adjustment = 1.0 + z_squared / denominator
        center = (point_estimate + z_squared / (2.0 * denominator)) / denominator_adjustment
        half_width = (
            z_score
            * sqrt(
                (point_estimate * (1.0 - point_estimate) / denominator)
                + (z_squared / (4.0 * denominator * denominator))
            )
            / denominator_adjustment
        )
        lower = max(0.0, center - half_width)
        upper = min(1.0, center + half_width)

    return {
        "method": WILSON_METHOD,
        "confidence_level": confidence_level,
        "lower": lower,
        "upper": upper,
        "numerator": numerator,
        "denominator": denominator,
    }


def wilson_confidence_interval(
    numerator: int,
    denominator: int,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    return wilson_interval(numerator, denominator, confidence_level)


def _validate_counts(numerator: int, denominator: int) -> tuple[int, int]:
    numerator = int(numerator)
    denominator = int(denominator)
    if denominator < 0:
        raise ValueError("Wilson interval denominator must be non-negative.")
    if numerator < 0:
        raise ValueError("Wilson interval numerator must be non-negative.")
    if numerator > denominator:
        raise ValueError("Wilson interval numerator cannot exceed denominator.")
    return numerator, denominator


def _validate_confidence_level(confidence_level: float) -> None:
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")
