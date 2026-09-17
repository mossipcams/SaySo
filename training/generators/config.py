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
    # The reachable ceiling is the catalogue size times near_duplicate_limit
    # (pipeline._grounding_capacity), checked before the build. v1 requested 0.03
    # against a 120-row catalogue ceiling and shipped 0.0022 reporting success.
    grounding_rate: float = 0.028
    # Share of accepted rows where the utterance DESCRIBES the target entity
    # without naming it, with same-domain same-area peers present in context.
    # This is the entity-discrimination skill the eval measures. v1 corpora
    # shipped with only 3.7% of rows doing this, so the model learned to echo a
    # name rather than resolve a device.
    discrimination_rate: float = 0.0
    # Fail the build when achieved grounding share falls below this fraction of
    # the reachable ceiling. v1 requested 3% and shipped 0.22% with no error, so
    # the gate must catch order-of-magnitude shortfalls. It must not trip on the
    # normal run-to-run variance of the family rotation, which measured 2.4-2.8%
    # against a 2.8% ceiling (a spread of roughly 15%), hence 0.75 rather than a
    # tight fraction.
    min_rate_achieved_fraction: float = 0.75
    # Share of accepted rows offered the whole catalogue rather than a sampled
    # subset. A subset is built by keeping the expected answer first, so a corpus
    # made only of subsets never shows a tool list that was not pre-filtered to
    # contain the answer; production sends ~23 tools.
    #
    # Opt-in (see --full-catalog-rate): a full catalogue costs ~4k tokens a row
    # and changes what every existing recipe generates, so an old recipe keeps
    # producing the corpus it always did. The issue #52 recipe passes 0.35.
    full_catalog_rate: float = 0.0
    # Versioned area-scenario mix (generators.area_scenarios). Those rows are
    # reserved out of ``count`` and generation fails when one falls short. None
    # keeps the production area-format gate but reserves no scenario rows; the
    # CLI, which builds real corpora, defaults to configs/area_distribution_v1.json.
    area_distribution_path: Path | None = None
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
        for name in ("grounding_rate", "discrimination_rate", "ordinary_rate",
                     "full_catalog_rate"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1], got {value}")
        if self.grounding_rate + self.discrimination_rate > 1.0:
            raise ValueError(
                "grounding_rate + discrimination_rate must be <= 1: "
                f"{self.grounding_rate} + {self.discrimination_rate}"
            )
        if not 0.0 <= self.min_rate_achieved_fraction <= 1.0:
            raise ValueError(
                "min_rate_achieved_fraction must be within [0, 1], got "
                f"{self.min_rate_achieved_fraction}"
            )

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
            "discrimination_rate": self.discrimination_rate,
            "full_catalog_rate": self.full_catalog_rate,
            "area_distribution_path": str(self.area_distribution_path) if self.area_distribution_path else None,
            "min_rate_achieved_fraction": self.min_rate_achieved_fraction,
            "min_positive_per_operation": self.min_positive_per_operation,
            "min_positive_per_tool": self.min_positive_per_tool,
            "max_absence_rate": self.max_absence_rate,
            "real_home_path": str(self.real_home_path) if self.real_home_path else None,
            "real_home_rate": self.real_home_rate,
            "real_home_entity_cap": self.real_home_entity_cap,
            "synthetic_only": self.synthetic_only,
        }
