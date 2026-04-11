from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Sequence

from .config import DEFAULT_STAGE2_REPLAY_CONFIG_PATH
from .runner import run_stage2_evaluation


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser(description="Run the Stage 2 adapter replay and leakage scorer.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_STAGE2_REPLAY_CONFIG_PATH,
        help="Path to the Stage 2 replay config TOML.",
    )
    parser.add_argument(
        "--exposure",
        choices=("1x", "10x", "50x", "all"),
        required=True,
        help="Exposure condition to evaluate, or 'all' to run all official adapters.",
    )
    parser.add_argument(
        "--condition",
        choices=("no_system_prompt", "system_prompt_active", "all"),
        required=True,
        help="Prompt condition to evaluate, or 'all' to run both frozen conditions.",
    )
    args = parser.parse_args(argv)

    result = run_stage2_evaluation(
        config_path=args.config,
        exposure=args.exposure,
        condition=args.condition,
    )

    for key in sorted(result.artifacts):
        artifact = result.artifacts[key]
        print(f"{artifact.exposure_condition}_{artifact.prompt_condition}_responses: {artifact.responses_path}")
        print(f"{artifact.exposure_condition}_{artifact.prompt_condition}_metrics: {artifact.metrics_path}")
        print(f"{artifact.exposure_condition}_{artifact.prompt_condition}_family_metrics: {artifact.family_metrics_path}")
    print(f"summary: {result.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
