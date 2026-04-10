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
from .stage1_validators import (
    MiaEvalValidationSummary,
    Stage1CorpusValidationError,
    TrainingCorpusValidationSummary,
    validate_mia_eval_corpus,
    validate_stage1_source_inputs,
    validate_training_corpus,
)
from .tier1 import GeneratedTier1Dataset, generate_tier1_dataset
from .validators import Tier1ValidationError, ValidationSummary, validate_tier1_dataset

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_STAGE1_CORPORA_CONFIG_PATH",
    "GeneratedTier1Dataset",
    "MiaEvalValidationSummary",
    "MaterializationResult",
    "Stage1CorpusConfig",
    "Stage1CorpusMaterializationResult",
    "Stage1CorpusValidationError",
    "TrainingCorpusValidationSummary",
    "Tier1DataConfig",
    "Tier1ValidationError",
    "ValidationSummary",
    "build_mia_eval_corpus",
    "build_training_corpus",
    "generate_tier1_dataset",
    "materialize_stage1_corpora",
    "materialize_tier1_artifacts",
    "validate_mia_eval_corpus",
    "validate_stage1_source_inputs",
    "validate_tier1_dataset",
    "validate_training_corpus",
]
