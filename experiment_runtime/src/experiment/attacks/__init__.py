"""Stage 2 attack template catalogs and helpers."""

from .stage2_catalog import (
    ATTACK_FAMILY_CATALOGS,
    CANARY_ASSIGNMENT_FAMILY_OFFSET,
    FROZEN_ATTACK_FAMILY_ORDER,
    FROZEN_ATTACK_FAMILY_SLUGS,
    AttackFamilyCatalog,
    AttackPromptSpec,
    build_attack_prompt_specs,
)

__all__ = [
    "ATTACK_FAMILY_CATALOGS",
    "CANARY_ASSIGNMENT_FAMILY_OFFSET",
    "FROZEN_ATTACK_FAMILY_ORDER",
    "FROZEN_ATTACK_FAMILY_SLUGS",
    "AttackFamilyCatalog",
    "AttackPromptSpec",
    "build_attack_prompt_specs",
]
