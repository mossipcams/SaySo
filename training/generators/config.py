"""YAML recipe loader and generator configuration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from generators.capability_registry import HOME_SIZE_WEIGHTS, TIER_PROPORTIONS
from generators.planning import PRIMARY_FAMILIES

DEFAULT_TRAIN_COUNT = 40_000
DEFAULT_SEED = 20260905
DEFAULT_TOKEN_BUDGET = 5120
DEFAULT_STT_RATE = 0.15
DEFAULT_MAX_ATTEMPTS_MULTIPLIER = 20
DEFAULT_NEAR_DUPLICATE_LIMIT = 8
DEFAULT_TOKENIZER = "LiquidAI/LFM2.5-230M-Base"


def default_allocations() -> dict[str, float]:
    return {family: 1.0 / len(PRIMARY_FAMILIES) for family in PRIMARY_FAMILIES}


@dataclass
class GeneratorConfig:
    """Versioned recipe-backed configuration for synthetic dataset generation."""

    count: int = DEFAULT_TRAIN_COUNT
    seed: int = DEFAULT_SEED
    split: str = "train"
    output_path: Path = field(
        default_factory=lambda: Path("training/datasets/synthetic_v3_train.jsonl")
    )
    manifest_path: Path | None = None
    recipe_path: Path | None = None
    recipe_version: int = 1
    allocations: dict[str, float] = field(default_factory=dict)
    tier_proportions: dict[int, float] = field(default_factory=lambda: dict(TIER_PROPORTIONS))
    home_size_weights: dict[int, int] = field(default_factory=lambda: dict(HOME_SIZE_WEIGHTS))
    stt_noise_rate: float = DEFAULT_STT_RATE
    paraphrase_enabled: bool = False
    paraphrase_variants: int = 0
    token_budget: int = DEFAULT_TOKEN_BUDGET
    tokenizer_model: str = DEFAULT_TOKENIZER
    schema_path: Path = field(
        default_factory=lambda: Path("schemas/sayso-tool-schema-v2.json")
    )
    near_duplicate_limit: int = DEFAULT_NEAR_DUPLICATE_LIMIT
    max_attempts_multiplier: int = DEFAULT_MAX_ATTEMPTS_MULTIPLIER
    grounding_rate: float = 0.0
    discrimination_rate: float = 0.0
    area_distribution_path: Path | None = None
    min_positive_per_operation: int = 1
    min_positive_per_tool: int = 1
    max_absence_rate: float = 0.10
    get_datetime_positive_min: int = 1
    exclude_prompts_path: Path | None = None
    real_home_path: Path | None = None
    real_home_rate: float = 0.0
    real_home_entity_cap: int = 0
    synthetic_only: bool = False

    def __post_init__(self) -> None:
        if self.synthetic_only:
            self.real_home_path = None
            self.real_home_rate = 0.0
        if self.real_home_rate and not self.real_home_path:
            raise ValueError("real_home_rate needs real_home_path")
        if self.allocations:
            total = sum(self.allocations.values())
            if abs(total - 1.0) > 0.02:
                raise ValueError(f"allocations must sum to ~1.0, got {total:.4f}")
        for name in ("grounding_rate", "discrimination_rate"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1], got {value}")
        if self.grounding_rate + self.discrimination_rate > 1.0:
            raise ValueError(
                "grounding_rate + discrimination_rate must be <= 1, "
                f"got {self.grounding_rate} + {self.discrimination_rate}"
            )

    def max_attempts(self) -> int:
        return self.count * self.max_attempts_multiplier

    @classmethod
    def from_yaml(cls, path: Path | str, repo_root: Path | None = None) -> GeneratorConfig:
        recipe_path = Path(path).resolve()
        if repo_root is None:
            raise ValueError(
                "repo_root is required when loading a recipe from YAML "
                "(pass the repository root, not training/)"
            )
        raw = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
        root = repo_root
        gen = raw.get("generation", {})
        real = raw.get("real_home", {})
        coverage = raw.get("coverage", {})
        cross = raw.get("cross_cutting", {}).get("area", {})
        schema = raw.get("schema", {})
        tokenizer = raw.get("tokenizer", {})
        output = Path(raw["output"])
        if not output.is_absolute():
            output = root / output
        manifest = raw.get("manifest")
        manifest_path = Path(manifest) if manifest else output.with_suffix(".manifest.json")
        if not manifest_path.is_absolute():
            manifest_path = root / manifest_path
        area_path = cross.get("distribution")
        area_distribution_path = None
        if cross.get("enabled") and area_path:
            area_distribution_path = Path(area_path)
            if not area_distribution_path.is_absolute():
                area_distribution_path = root / area_distribution_path
        schema_path = Path(schema.get("path", "schemas/sayso-tool-schema-v2.json"))
        if not schema_path.is_absolute():
            schema_path = root / schema_path
        real_home = real.get("path")
        real_home_path = Path(real_home) if real_home else None
        if real_home_path and not real_home_path.is_absolute():
            real_home_path = root / real_home_path
        allocations = raw.get("allocations", {})
        return cls(
            count=int(raw["count"]),
            seed=int(raw["seed"]),
            split=str(raw.get("split", "train")),
            output_path=output,
            manifest_path=manifest_path,
            recipe_path=recipe_path,
            recipe_version=int(raw.get("version", 1)),
            allocations={k: float(v) for k, v in allocations.items()},
            stt_noise_rate=float(gen.get("stt_noise_rate", DEFAULT_STT_RATE)),
            paraphrase_enabled=bool(gen.get("paraphrase_enabled", False)),
            token_budget=int(gen.get("token_budget", DEFAULT_TOKEN_BUDGET)),
            tokenizer_model=str(tokenizer.get("model", DEFAULT_TOKENIZER)),
            schema_path=schema_path,
            near_duplicate_limit=int(gen.get("near_duplicate_limit", DEFAULT_NEAR_DUPLICATE_LIMIT)),
            max_attempts_multiplier=int(
                gen.get("max_attempts_multiplier", DEFAULT_MAX_ATTEMPTS_MULTIPLIER)
            ),
            grounding_rate=float(raw.get("grounding", {}).get("rate", 0.0)),
            discrimination_rate=float(raw.get("discrimination", {}).get("rate", 0.0)),
            area_distribution_path=area_distribution_path,
            min_positive_per_operation=int(coverage.get("min_positive_per_operation", 1)),
            min_positive_per_tool=int(coverage.get("min_positive_per_tool", 1)),
            max_absence_rate=float(coverage.get("max_absence_rate", 0.10)),
            get_datetime_positive_min=int(coverage.get("get_datetime_positive_min", 1)),
            real_home_path=real_home_path,
            real_home_rate=float(real.get("rate", 0.0)),
            real_home_entity_cap=int(real.get("entity_cap", 0)),
            synthetic_only=bool(gen.get("synthetic_only", True)),
        )

    def recipe_hash(self) -> str:
        if self.recipe_path and self.recipe_path.is_file():
            return hashlib.sha256(self.recipe_path.read_bytes()).hexdigest()
        return hashlib.sha256(repr(self.to_dict()).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "seed": self.seed,
            "split": self.split,
            "output_path": str(self.output_path),
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "recipe_path": str(self.recipe_path) if self.recipe_path else None,
            "recipe_version": self.recipe_version,
            "allocations": self.allocations,
            "stt_noise_rate": self.stt_noise_rate,
            "paraphrase_enabled": self.paraphrase_enabled,
            "token_budget": self.token_budget,
            "tokenizer_model": self.tokenizer_model,
            "schema_path": str(self.schema_path),
            "near_duplicate_limit": self.near_duplicate_limit,
            "grounding_rate": self.grounding_rate,
            "discrimination_rate": self.discrimination_rate,
            "area_distribution_path": str(self.area_distribution_path)
            if self.area_distribution_path
            else None,
            "min_positive_per_operation": self.min_positive_per_operation,
            "min_positive_per_tool": self.min_positive_per_tool,
            "max_absence_rate": self.max_absence_rate,
            "get_datetime_positive_min": self.get_datetime_positive_min,
            "real_home_path": str(self.real_home_path) if self.real_home_path else None,
            "real_home_rate": self.real_home_rate,
            "real_home_entity_cap": self.real_home_entity_cap,
            "synthetic_only": self.synthetic_only,
        }
