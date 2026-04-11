from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Sequence

from .config import DEFAULT_STAGE2_FILTERED_REPLAY_CONFIG_PATH
from .runner import run_stage2_filtered_evaluation


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser(description="Run the Stage 2 filtered integrated reruns.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_STAGE2_FILTERED_REPLAY_CONFIG_PATH,
        help="Path to the Stage 2 filtered replay config TOML.",
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
        help="Filter mode to evaluate, or 'all' to run both filtered conditions.",
    )
    args = parser.parse_args(argv)

    result = run_stage2_filtered_evaluation(
        config_path=args.config,
        exposure=args.exposure,
        filter_mode=args.filter_mode,
    )

    for key in sorted(result.artifacts):
        artifact = result.artifacts[key]
        print(f"{artifact.exposure_condition}_{artifact.filter_mode}_responses: {artifact.responses_path}")
        print(f"{artifact.exposure_condition}_{artifact.filter_mode}_metrics: {artifact.metrics_path}")
        print(
            f"{artifact.exposure_condition}_{artifact.filter_mode}_family_metrics: "
            f"{artifact.family_metrics_path}"
        )
    print(f"summary: {result.summary_path}")
    print(f"parity_summary: {result.parity_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
