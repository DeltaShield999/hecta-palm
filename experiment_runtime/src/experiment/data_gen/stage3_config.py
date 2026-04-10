from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STAGE3_FILTER_MESSAGES_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "data" / "stage3_filter_messages.toml"
)


@dataclass(frozen=True, slots=True)
class Stage3FilterMessageConfig:
    protocol_config_dir: Path
    tier1_records_path: Path
    stage2_attack_prompts_path: Path
    output_dir: Path
    train_filename: str
    val_filename: str
    test_filename: str

    @property
    def train_output_path(self) -> Path:
        return self.output_dir / self.train_filename

    @property
    def val_output_path(self) -> Path:
        return self.output_dir / self.val_filename

    @property
    def test_output_path(self) -> Path:
        return self.output_dir / self.test_filename

    @classmethod
    def from_toml(cls, config_path: Path | str | None = None) -> "Stage3FilterMessageConfig":
        path = _resolve_path(config_path or DEFAULT_STAGE3_FILTER_MESSAGES_CONFIG_PATH)
        with path.open("rb") as handle:
            document = tomllib.load(handle)

        return cls(
            protocol_config_dir=_resolve_path(document["protocol"]["config_dir"]),
            tier1_records_path=_resolve_path(document["inputs"]["tier1_records_path"]),
            stage2_attack_prompts_path=_resolve_path(document["inputs"]["stage2_attack_prompts_path"]),
            output_dir=_resolve_path(document["outputs"]["output_dir"]),
            train_filename=str(document["outputs"]["train_filename"]),
            val_filename=str(document["outputs"]["val_filename"]),
            test_filename=str(document["outputs"]["test_filename"]),
        )


def _resolve_path(path_value: Path | str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
