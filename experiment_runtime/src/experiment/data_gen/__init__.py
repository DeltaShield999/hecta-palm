"""Deterministic Tier 1 dataset generation and validation."""

from .config import DEFAULT_CONFIG_PATH, Tier1DataConfig
from .materialize_tier1 import MaterializationResult, materialize_tier1_artifacts
from .stage1_config import DEFAULT_STAGE1_CORPORA_CONFIG_PATH, Stage1CorpusConfig
from .stage1_corpora import (
    Stage1CorpusMaterializationResult,
    build_mia_eval_corpus,
    build_training_corpus,
    materialize_stage1_corpora,
)
from .stage2_attack_prompts import (
    Stage2AttackPromptMaterializationResult,
    build_stage2_attack_prompts,
    materialize_stage2_attack_prompts,
)
from .stage2_config import (
    DEFAULT_STAGE2_ATTACK_PROMPTS_CONFIG_PATH,
    Stage2AttackPromptConfig,
)
from .stage1_validators import (
    MiaEvalValidationSummary,
    Stage1CorpusValidationError,
    TrainingCorpusValidationSummary,
    validate_mia_eval_corpus,
    validate_stage1_source_inputs,
    validate_training_corpus,
)
from .stage2_validators import (
    Stage2AttackPromptValidationError,
    Stage2AttackPromptValidationSummary,
    validate_stage2_attack_prompts,
)
from .tier1 import GeneratedTier1Dataset, generate_tier1_dataset
from .validators import Tier1ValidationError, ValidationSummary, validate_tier1_dataset

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_STAGE1_CORPORA_CONFIG_PATH",
    "DEFAULT_STAGE2_ATTACK_PROMPTS_CONFIG_PATH",
    "GeneratedTier1Dataset",
    "MiaEvalValidationSummary",
    "MaterializationResult",
    "Stage1CorpusConfig",
    "Stage1CorpusMaterializationResult",
    "Stage1CorpusValidationError",
    "Stage2AttackPromptConfig",
    "Stage2AttackPromptMaterializationResult",
    "Stage2AttackPromptValidationError",
    "Stage2AttackPromptValidationSummary",
    "TrainingCorpusValidationSummary",
    "Tier1DataConfig",
    "Tier1ValidationError",
    "ValidationSummary",
    "build_mia_eval_corpus",
    "build_stage2_attack_prompts",
    "build_training_corpus",
    "generate_tier1_dataset",
    "materialize_stage1_corpora",
    "materialize_stage2_attack_prompts",
    "materialize_tier1_artifacts",
    "validate_mia_eval_corpus",
    "validate_stage1_source_inputs",
    "validate_stage2_attack_prompts",
    "validate_tier1_dataset",
    "validate_training_corpus",
]
