from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STAGE2_ATTACK_PROMPTS_CONFIG_PATH = PROJECT_ROOT / "configs" / "data" / "stage2_attack_prompts.toml"


@dataclass(frozen=True, slots=True)
class Stage2AttackPromptConfig:
    protocol_config_dir: Path
    tier1_records_path: Path
    canary_registry_path: Path
    output_dir: Path
    output_filename: str

    @property
    def output_path(self) -> Path:
        return self.output_dir / self.output_filename

    @classmethod
    def from_toml(cls, config_path: Path | str | None = None) -> "Stage2AttackPromptConfig":
        path = _resolve_path(config_path or DEFAULT_STAGE2_ATTACK_PROMPTS_CONFIG_PATH)
        with path.open("rb") as handle:
            document = tomllib.load(handle)

        return cls(
            protocol_config_dir=_resolve_path(document["protocol"]["config_dir"]),
            tier1_records_path=_resolve_path(document["inputs"]["tier1_records_path"]),
            canary_registry_path=_resolve_path(document["inputs"]["canary_registry_path"]),
            output_dir=_resolve_path(document["outputs"]["output_dir"]),
            output_filename=str(document["outputs"]["output_filename"]),
        )


def _resolve_path(path_value: Path | str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
