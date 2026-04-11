from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

from qwen_langgraph_demo.runtime.protocol import load_protocol_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STAGE3_PLAINTEXT_FILTER_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "eval" / "stage3_plaintext_filter.toml"
)
PINNED_STAGE3_ENCODER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PINNED_LOGISTIC_REGRESSION_C_VALUES = (0.01, 0.1, 1.0, 10.0)
PINNED_THRESHOLD_TIE_BREAKERS = (
    "block_recall",
    "benign_false_positive_rate",
    "smallest_threshold",
)


@dataclass(frozen=True, slots=True)
class DatasetPaths:
    train: Path
    val: Path
    test: Path

    def by_split(self) -> dict[str, Path]:
        return {
            "train": self.train,
            "val": self.val,
            "test": self.test,
        }


@dataclass(frozen=True, slots=True)
class EncoderSettings:
    model_name: str
    normalize_embeddings: bool
    batch_size: int
    device: str


@dataclass(frozen=True, slots=True)
class LogisticRegressionSettings:
    candidate_c_values: tuple[float, ...]
    solver: str
    max_iter: int


@dataclass(frozen=True, slots=True)
class ThresholdSelectionPolicy:
    selection_split: str
    score_label: str
    decision_rule: str
    objective: str
    tie_breakers: tuple[str, ...]
    c_tie_breaker: str


@dataclass(frozen=True, slots=True)
class Stage3PlaintextFilterConfig:
    config_path: Path
    protocol_config_dir: Path
    datasets: DatasetPaths
    output_root: Path
    encoder: EncoderSettings
    logistic_regression: LogisticRegressionSettings
    threshold_selection: ThresholdSelectionPolicy
    seed: int

    @classmethod
    def from_toml(cls, config_path: Path | str | None = None) -> "Stage3PlaintextFilterConfig":
        path = _resolve_path(config_path or DEFAULT_STAGE3_PLAINTEXT_FILTER_CONFIG_PATH)
        with path.open("rb") as handle:
            document = tomllib.load(handle)

        protocol_config_dir = _resolve_path(document["protocol"]["config_dir"])
        load_protocol_bundle(protocol_config_dir)

        datasets = DatasetPaths(
            train=_resolve_existing_path(document["inputs"]["train_dataset_path"]),
            val=_resolve_existing_path(document["inputs"]["val_dataset_path"]),
            test=_resolve_existing_path(document["inputs"]["test_dataset_path"]),
        )

        encoder = EncoderSettings(
            model_name=str(document["encoder"]["model_name"]),
            normalize_embeddings=bool(document["encoder"]["normalize_embeddings"]),
            batch_size=int(document["encoder"]["batch_size"]),
            device=str(document["encoder"].get("device", "cpu")),
        )
        if encoder.model_name != PINNED_STAGE3_ENCODER_MODEL_NAME:
            raise ValueError(
                "Stage 3 plaintext training must use the pinned encoder "
                f"{PINNED_STAGE3_ENCODER_MODEL_NAME}, found {encoder.model_name}."
            )
        if not encoder.normalize_embeddings:
            raise ValueError("Stage 3 plaintext training must set normalize_embeddings = true.")
        if encoder.batch_size <= 0:
            raise ValueError("encoder.batch_size must be positive.")

        logistic_regression = LogisticRegressionSettings(
            candidate_c_values=tuple(
                float(value) for value in document["logistic_regression"]["candidate_c_values"]
            ),
            solver=str(document["logistic_regression"]["solver"]),
            max_iter=int(document["logistic_regression"]["max_iter"]),
        )
        if logistic_regression.candidate_c_values != PINNED_LOGISTIC_REGRESSION_C_VALUES:
            raise ValueError(
                "Stage 3 plaintext training must sweep the frozen C values "
                f"{PINNED_LOGISTIC_REGRESSION_C_VALUES}, found "
                f"{logistic_regression.candidate_c_values}."
            )
        if logistic_regression.solver != "liblinear":
            raise ValueError("Stage 3 plaintext training must use solver = 'liblinear'.")
        if logistic_regression.max_iter <= 0:
            raise ValueError("logistic_regression.max_iter must be positive.")

        threshold_selection = ThresholdSelectionPolicy(
            selection_split=str(document["threshold_selection"]["selection_split"]),
            score_label=str(document["threshold_selection"]["score_label"]),
            decision_rule=str(document["threshold_selection"]["decision_rule"]),
            objective=str(document["threshold_selection"]["objective"]),
            tie_breakers=tuple(str(value) for value in document["threshold_selection"]["tie_breakers"]),
            c_tie_breaker=str(document["threshold_selection"].get("c_tie_breaker", "smallest_c")),
        )
        if threshold_selection.selection_split != "val":
            raise ValueError("Stage 3 threshold selection must operate on the validation split only.")
        if threshold_selection.score_label != "BLOCK":
            raise ValueError("Stage 3 threshold scores must use the BLOCK probability.")
        if threshold_selection.objective != "macro_f1":
            raise ValueError("Stage 3 threshold selection must optimize macro_f1.")
        if threshold_selection.tie_breakers != PINNED_THRESHOLD_TIE_BREAKERS:
            raise ValueError(
                "Stage 3 threshold selection must use tie breakers "
                f"{PINNED_THRESHOLD_TIE_BREAKERS}, found {threshold_selection.tie_breakers}."
            )
        if threshold_selection.c_tie_breaker != "smallest_c":
            raise ValueError("Stage 3 C-selection ties must resolve with the smallest C.")

        return cls(
            config_path=path,
            protocol_config_dir=protocol_config_dir,
            datasets=datasets,
            output_root=_resolve_path(document["outputs"]["root_dir"]),
            encoder=encoder,
            logistic_regression=logistic_regression,
            threshold_selection=threshold_selection,
            seed=int(document["seed"]["value"]),
        )


def _resolve_existing_path(value: Path | str) -> Path:
    path = _resolve_path(value)
    if not path.exists():
        raise FileNotFoundError(f"Stage 3 plaintext config references a missing path: {path}")
    return path


def _resolve_path(value: Path | str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()
