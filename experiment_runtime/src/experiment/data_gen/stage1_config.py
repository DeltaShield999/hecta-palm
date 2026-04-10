from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

from qwen_langgraph_demo.runtime.protocol import load_protocol_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STAGE1_CORPORA_CONFIG_PATH = PROJECT_ROOT / "configs" / "data" / "stage1_corpora.toml"


@dataclass(frozen=True, slots=True)
class Stage1CorpusConfig:
    protocol_config_dir: Path
    tier1_records_path: Path
    canary_registry_path: Path
    output_dir: Path
    training_filenames: dict[str, str]
    mia_eval_filename: str

    @property
    def training_paths(self) -> dict[str, Path]:
        return {
            exposure_condition: self.output_dir / filename
            for exposure_condition, filename in self.training_filenames.items()
        }

    @property
    def mia_eval_path(self) -> Path:
        return self.output_dir / self.mia_eval_filename

    @classmethod
    def from_toml(cls, config_path: Path | str | None = None) -> "Stage1CorpusConfig":
        path = _resolve_path(config_path or DEFAULT_STAGE1_CORPORA_CONFIG_PATH)
        with path.open("rb") as handle:
            document = tomllib.load(handle)

        protocol_dir = _resolve_path(document["protocol"]["config_dir"])
        protocol = load_protocol_bundle(protocol_dir)
        outputs = document["outputs"]
        training_filenames = {str(key): str(value) for key, value in outputs["training"].items()}

        expected_exposures = tuple(protocol.core.exposure_conditions)
        if tuple(training_filenames) != expected_exposures:
            raise ValueError(
                "Stage 1 corpus config must define training filenames for exposure conditions "
                f"{expected_exposures}, found {tuple(training_filenames)}."
            )

        return cls(
            protocol_config_dir=protocol_dir,
            tier1_records_path=_resolve_path(document["inputs"]["tier1_records_path"]),
            canary_registry_path=_resolve_path(document["inputs"]["canary_registry_path"]),
            output_dir=_resolve_path(outputs["output_dir"]),
            training_filenames=training_filenames,
            mia_eval_filename=str(outputs["mia_eval_filename"]),
        )


def _resolve_path(path_value: Path | str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
