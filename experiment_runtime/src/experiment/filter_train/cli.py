from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Sequence

from .config import DEFAULT_STAGE3_PLAINTEXT_FILTER_CONFIG_PATH
from .runner import run_stage3_plaintext_training


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser(description="Train and evaluate the Stage 3 plaintext filter.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_STAGE3_PLAINTEXT_FILTER_CONFIG_PATH,
        help="Path to the Stage 3 plaintext filter config TOML.",
    )
    args = parser.parse_args(argv)

    result = run_stage3_plaintext_training(args.config)
    print(f"metrics: {result.artifacts.metrics_path}")
    print(f"model_selection: {result.artifacts.model_selection_path}")
    print(f"validation_threshold_sweep: {result.artifacts.validation_threshold_sweep_path}")
    print(f"test_predictions: {result.artifacts.test_predictions_path}")
    print(f"encoder_metadata: {result.artifacts.encoder_metadata_path}")
    print(f"selected_c: {result.selected_c}")
    print(f"selected_threshold: {result.selected_threshold}")
    print(f"macro_f1: {result.test_metrics.macro_f1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
