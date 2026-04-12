from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Sequence

from .config import DEFAULT_STAGE1_MIA_CONFIG_PATH
from .runner import run_stage1_mia_evaluation


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser(description="Run the Stage 1 membership-inference evaluator.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_STAGE1_MIA_CONFIG_PATH,
        help="Path to the Stage 1 MIA config TOML.",
    )
    parser.add_argument(
        "--exposure",
        choices=("1x", "10x", "50x", "all"),
        required=True,
        help="Exposure condition to evaluate, or 'all' to run all official adapters.",
    )
    args = parser.parse_args(argv)

    result = run_stage1_mia_evaluation(
        config_path=args.config,
        exposure=args.exposure,
    )

    print(f"base_losses: {result.base_losses_path}")
    if result.base_loss_batches_path is not None:
        print(f"base_loss_batches: {result.base_loss_batches_path}")
    for exposure_condition, artifacts in result.exposure_artifacts.items():
        print(f"{exposure_condition}_losses: {artifacts.stage1_losses_path}")
        print(f"{exposure_condition}_metrics: {artifacts.stage1_metrics_path}")
        if artifacts.forward_batches_path is not None:
            print(f"{exposure_condition}_forward_batches: {artifacts.forward_batches_path}")
        if artifacts.timing_path is not None:
            print(f"{exposure_condition}_timing: {artifacts.timing_path}")
    print(f"summary: {result.summary_path}")
    if result.timing_summary_path is not None:
        print(f"timing_summary: {result.timing_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
