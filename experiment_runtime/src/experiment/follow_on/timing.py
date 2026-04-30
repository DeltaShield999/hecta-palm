from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from math import sqrt
from time import perf_counter
from typing import Any


TIMING_SUMMARY_FIELDS = (
    "count",
    "mean",
    "p50",
    "p90",
    "p95",
    "p99",
    "min",
    "max",
    "std",
)

SETUP_TIMING_COLUMNS = (
    "component",
    "duration_ms",
    "detail",
)

FILTER_TIMING_COLUMNS = (
    "row_id",
    "eval_dataset",
    "filter_mode",
    "embedding_ms",
    "encryption_ms",
    "fhe_scoring_ms",
    "decryption_ms",
    "threshold_ms",
    "io_ms",
    "total_filter_ms",
)

PIPELINE_TIMING_COLUMNS = (
    "row_id",
    "exposure_condition",
    "eval_dataset",
    "condition",
    "filter_mode",
    "source_type",
    "filter_decision",
    "response_generated",
    "filter_total_ms",
    "llm_generation_ms",
    "routing_ms",
    "total_pipeline_ms",
)

DEFAULT_FILTER_TIMING_NUMERIC_COLUMNS = (
    "embedding_ms",
    "encryption_ms",
    "fhe_scoring_ms",
    "decryption_ms",
    "threshold_ms",
    "io_ms",
    "total_filter_ms",
)

DEFAULT_PIPELINE_TIMING_NUMERIC_COLUMNS = (
    "filter_total_ms",
    "llm_generation_ms",
    "routing_ms",
    "total_pipeline_ms",
)


@dataclass(frozen=True, slots=True)
class SetupTimingEntry:
    component: str
    duration_ms: float | None
    detail: str | None = None

    def to_row(self) -> dict[str, str | float | None]:
        return {
            "component": self.component,
            "duration_ms": self.duration_ms,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class FilterTimingSample:
    row_id: str
    eval_dataset: str
    filter_mode: str
    embedding_ms: float | None = None
    encryption_ms: float | None = None
    fhe_scoring_ms: float | None = None
    decryption_ms: float | None = None
    threshold_ms: float | None = None
    io_ms: float | None = None
    total_filter_ms: float | None = None

    def to_row(self) -> dict[str, str | float | None]:
        return {
            "row_id": self.row_id,
            "eval_dataset": self.eval_dataset,
            "filter_mode": self.filter_mode,
            "embedding_ms": self.embedding_ms,
            "encryption_ms": self.encryption_ms,
            "fhe_scoring_ms": self.fhe_scoring_ms,
            "decryption_ms": self.decryption_ms,
            "threshold_ms": self.threshold_ms,
            "io_ms": self.io_ms,
            "total_filter_ms": self.total_filter_ms,
        }


@dataclass(frozen=True, slots=True)
class PipelineTimingSample:
    row_id: str
    exposure_condition: str
    eval_dataset: str
    condition: str
    filter_mode: str | None
    source_type: str | None
    filter_decision: str | None
    response_generated: int
    filter_total_ms: float | None = None
    llm_generation_ms: float | None = None
    routing_ms: float | None = None
    total_pipeline_ms: float | None = None

    def to_row(self) -> dict[str, str | int | float | None]:
        return {
            "row_id": self.row_id,
            "exposure_condition": self.exposure_condition,
            "eval_dataset": self.eval_dataset,
            "condition": self.condition,
            "filter_mode": self.filter_mode,
            "source_type": self.source_type,
            "filter_decision": self.filter_decision,
            "response_generated": self.response_generated,
            "filter_total_ms": self.filter_total_ms,
            "llm_generation_ms": self.llm_generation_ms,
            "routing_ms": self.routing_ms,
            "total_pipeline_ms": self.total_pipeline_ms,
        }


def elapsed_ms(start_time: float, end_time: float | None = None) -> float:
    end_time = perf_counter() if end_time is None else end_time
    return (end_time - start_time) * 1000.0


def summarize_numeric_values(values: Iterable[Any]) -> dict[str, float | int | None]:
    numeric_values = sorted(_numeric_values(values))
    count = len(numeric_values)
    if count == 0:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
            "std": None,
        }

    mean = sum(numeric_values) / count
    variance = sum((value - mean) ** 2 for value in numeric_values) / count
    return {
        "count": count,
        "mean": mean,
        "p50": _percentile(numeric_values, 50.0),
        "p90": _percentile(numeric_values, 90.0),
        "p95": _percentile(numeric_values, 95.0),
        "p99": _percentile(numeric_values, 99.0),
        "min": numeric_values[0],
        "max": numeric_values[-1],
        "std": sqrt(variance),
    }


def summarize_timing_rows(
    rows: Iterable[Any],
    *,
    numeric_columns: Sequence[str] | None = None,
) -> dict[str, dict[str, float | int | None]]:
    row_list = list(rows)
    columns = tuple(numeric_columns) if numeric_columns is not None else _infer_numeric_columns(row_list)
    return {
        column: summarize_numeric_values(_row_value(row, column) for row in row_list)
        for column in columns
    }


def summarize_timing_samples(
    rows: Iterable[Any],
    *,
    numeric_columns: Sequence[str] | None = None,
) -> dict[str, dict[str, float | int | None]]:
    return summarize_timing_rows(rows, numeric_columns=numeric_columns)


def summarize_filter_timing_samples(
    rows: Iterable[Any],
) -> dict[str, dict[str, float | int | None]]:
    return summarize_timing_rows(rows, numeric_columns=DEFAULT_FILTER_TIMING_NUMERIC_COLUMNS)


def summarize_pipeline_timing_samples(
    rows: Iterable[Any],
) -> dict[str, dict[str, float | int | None]]:
    return summarize_timing_rows(rows, numeric_columns=DEFAULT_PIPELINE_TIMING_NUMERIC_COLUMNS)


def _numeric_values(values: Iterable[Any]) -> list[float]:
    numeric_values: list[float] = []
    for value in values:
        if value in (None, ""):
            continue
        numeric_values.append(float(value))
    return numeric_values


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute a percentile without values.")
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = (len(sorted_values) - 1) * percentile / 100.0
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return float(lower_value + (upper_value - lower_value) * fraction)


def _infer_numeric_columns(rows: Sequence[Any]) -> tuple[str, ...]:
    ordered_columns: dict[str, None] = {}
    for row in rows:
        row_mapping = _row_mapping(row)
        for column in row_mapping:
            if column.endswith("_ms") or column == "duration_ms":
                ordered_columns.setdefault(column, None)
    return tuple(ordered_columns.keys())


def _row_value(row: Any, column: str) -> Any:
    if isinstance(row, Mapping):
        return row.get(column)
    if hasattr(row, "to_row"):
        return row.to_row().get(column)
    return getattr(row, column, None)


def _row_mapping(row: Any) -> Mapping[str, Any]:
    if isinstance(row, Mapping):
        return row
    if hasattr(row, "to_row"):
        return row.to_row()
    if is_dataclass(row):
        return asdict(row)
    raise TypeError("Timing rows must be mappings, dataclasses, or expose to_row().")
