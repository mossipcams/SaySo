"""Generator configuration dataclass and defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from generators.capability_registry import HOME_SIZE_WEIGHTS, TIER_PROPORTIONS

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
    exclude_prompts_path: Path | None = None

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
        }
