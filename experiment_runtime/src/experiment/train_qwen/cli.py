from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Sequence

from .config import DEFAULT_STAGE1_TRAIN_CONFIG_PATH, EXPOSURE_CONDITIONS
from .runner import run_stage1_training


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser(description="Train the Stage 1 Qwen2-1.5B-Instruct LoRA adapter.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_STAGE1_TRAIN_CONFIG_PATH,
        help="Path to the Stage 1 training config TOML.",
    )
    parser.add_argument(
        "--exposure",
        choices=EXPOSURE_CONDITIONS,
        required=True,
        help="Frozen exposure condition to train on.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for the isolated run folder under runs/stage1/.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Apply the cheap smoke-run overrides from the config.",
    )
    args = parser.parse_args(argv)

    result = run_stage1_training(
        config_path=args.config,
        exposure_condition=args.exposure,
        run_name=args.run_name,
        smoke=args.smoke,
    )

    print(f"run_dir: {result.run_dir}")
    print(f"adapter_dir: {result.adapter_dir}")
    print(f"tokenizer_dir: {result.tokenizer_dir}")
    print(f"train_metrics: {result.metrics_path}")
    print(f"train_examples: {result.train_examples}")
    print(f"global_step: {result.train_metrics.get('global_step')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
