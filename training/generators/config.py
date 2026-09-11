"""Generator configuration dataclass and defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from generators.capability_registry import (
    DEFAULT_NEGATIVE_RATE,
    HOME_SIZE_WEIGHTS,
    TIER_PROPORTIONS,
)

DEFAULT_TRAIN_COUNT = 40_000
DEFAULT_SEED = 20260905
DEFAULT_TOKEN_BUDGET = 4096
DEFAULT_STT_RATE = 0.15
DEFAULT_MAX_ATTEMPTS_MULTIPLIER = 20
DEFAULT_NEAR_DUPLICATE_LIMIT = 8


@dataclass
class GeneratorConfig:
    """CLI-backed configuration for synthetic dataset generation."""

    count: int = DEFAULT_TRAIN_COUNT
    seed: int = DEFAULT_SEED
    split: str = "train"
    output_path: Path = field(default_factory=lambda: Path("training/datasets/synthetic_v3_train.jsonl"))
    manifest_path: Path | None = None
    tier_proportions: dict[int, float] = field(default_factory=lambda: dict(TIER_PROPORTIONS))
    home_size_weights: dict[int, int] = field(default_factory=lambda: dict(HOME_SIZE_WEIGHTS))
    stt_noise_rate: float = DEFAULT_STT_RATE
    paraphrase_enabled: bool = False
    paraphrase_variants: int = 0
    token_budget: int = DEFAULT_TOKEN_BUDGET
    near_duplicate_limit: int = DEFAULT_NEAR_DUPLICATE_LIMIT
    max_attempts_multiplier: int = DEFAULT_MAX_ATTEMPTS_MULTIPLIER
    ordinary_rate: float = 0.75
    # Share of accepted rows budgeted for refusals, clarifications and absence
    # answers. Positive operation quotas are allocated over the rest, so a refusal
    # can never fill one.
    negative_rate: float = DEFAULT_NEGATIVE_RATE
    # Rows drawn from generators.grounding: same request, different entity graph.
    grounding_rate: float = 0.03
    # Dataset-level audit gates (generators.audit).
    min_positive_per_operation: int = 1
    min_positive_per_tool: int = 1
    max_absence_rate: float = 0.10
    exclude_prompts_path: Path | None = None
    # A real home fetched by scripts/fetch_ha_home.py, mixed in at this rate.
    # Default 0.0: an existing recipe generates the same dataset it always did.
    real_home_path: Path | None = None
    real_home_rate: float = 0.0
    # Max rows in which one real entity may be the target. 0 derives it from
    # count, rate, and the number of real entities (real_home.derive_entity_cap).
    real_home_entity_cap: int = 0
    # Explicit opt-out. A home-specific recipe sets a nonzero real_home_rate; this
    # is how a caller says "synthetic only" on purpose rather than by forgetting.
    synthetic_only: bool = False

    def __post_init__(self) -> None:
        if self.synthetic_only:
            self.real_home_path = None
            self.real_home_rate = 0.0
        if self.real_home_rate and not self.real_home_path:
            raise ValueError("real_home_rate needs real_home_path")

    def max_attempts(self) -> int:
        return self.count * self.max_attempts_multiplier

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "seed": self.seed,
            "split": self.split,
            "output_path": str(self.output_path),
            "tier_proportions": self.tier_proportions,
            "home_size_weights": self.home_size_weights,
            "stt_noise_rate": self.stt_noise_rate,
            "paraphrase_enabled": self.paraphrase_enabled,
            "paraphrase_variants": self.paraphrase_variants,
            "token_budget": self.token_budget,
            "near_duplicate_limit": self.near_duplicate_limit,
            "ordinary_rate": self.ordinary_rate,
            "negative_rate": self.negative_rate,
            "grounding_rate": self.grounding_rate,
            "min_positive_per_operation": self.min_positive_per_operation,
            "min_positive_per_tool": self.min_positive_per_tool,
            "max_absence_rate": self.max_absence_rate,
            "real_home_path": str(self.real_home_path) if self.real_home_path else None,
            "real_home_rate": self.real_home_rate,
            "real_home_entity_cap": self.real_home_entity_cap,
            "synthetic_only": self.synthetic_only,
        }
