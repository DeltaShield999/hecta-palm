from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Sequence

from .runner import (
    DEFAULT_FOLLOW_ON_MIXED_REPLAY_CONFIG_PATH,
    run_follow_on_mixed_evaluation,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser(description="Run the follow-on mixed-traffic replay harness.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_FOLLOW_ON_MIXED_REPLAY_CONFIG_PATH,
        help="Path to the follow-on mixed-traffic replay config TOML.",
    )
    parser.add_argument(
        "--exposure",
        choices=("1x", "10x", "50x", "all"),
        required=True,
        help="Exposure condition to evaluate, or 'all' to run all official adapters.",
    )
    parser.add_argument(
        "--filter-mode",
        choices=("plaintext_filter", "fhe_filter", "all"),
        required=True,
        help="Filter mode to evaluate, or 'all' to run both filter modes.",
    )
    args = parser.parse_args(argv)

    result = run_follow_on_mixed_evaluation(
        config_path=args.config,
        exposure=args.exposure,
        filter_mode=args.filter_mode,
    )

    for key in sorted(result.artifacts):
        artifact = result.artifacts[key]
        print(f"{artifact.exposure_condition}_{artifact.condition}_responses: {artifact.responses_path}")
        print(f"{artifact.exposure_condition}_{artifact.condition}_metrics: {artifact.metrics_path}")
        print(
            f"{artifact.exposure_condition}_{artifact.condition}_family_metrics: "
            f"{artifact.family_metrics_path}"
        )
        print(
            f"{artifact.exposure_condition}_{artifact.condition}_pipeline_timing: "
            f"{artifact.pipeline_timing_summary_path}"
        )
    print(f"summary: {result.summary_path}")
    print(f"ci_summary: {result.ci_summary_path}")
    print(f"parity_summary: {result.parity_summary_path}")
    print(f"setup_timing: {result.setup_timing_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
