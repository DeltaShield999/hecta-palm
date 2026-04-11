from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Sequence

from .config import DEFAULT_STAGE3_FHE_CONFIG_PATH
from .runner import run_stage3_fhe_evaluation


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser(
        description="Run the Stage 3 CKKS wrapper against the saved plaintext filter artifacts."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_STAGE3_FHE_CONFIG_PATH,
        help="Path to the Stage 3 FHE config TOML.",
    )
    args = parser.parse_args(argv)

    result = run_stage3_fhe_evaluation(args.config)
    print(f"metrics: {result.artifacts.metrics_path}")
    print(f"comparison: {result.artifacts.comparison_path}")
    print(f"latency_summary: {result.artifacts.latency_summary_path}")
    print(f"latency_samples: {result.artifacts.latency_samples_path}")
    print(f"context_metadata: {result.artifacts.context_metadata_path}")
    print(f"prediction_match_rate: {result.comparison_metrics.prediction_match_rate}")
    print(
        "mean_abs_probability_delta: "
        f"{result.comparison_metrics.mean_abs_probability_delta}"
    )
    print(f"end_to_end_mean_ms: {result.latency_summary['end_to_end_ms']['mean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
