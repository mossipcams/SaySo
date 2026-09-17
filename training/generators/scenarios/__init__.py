"""Structured scenario facts (homes, semantics, area, exclusions)."""

from generators.scenarios.area import (
    DEFAULT_DISTRIBUTION,
    SCENARIOS,
    build_area_spec,
    build_row,
    build_rows,
    load_distribution,
    required_counts,
    validate_distribution,
)
from generators.scenarios.core import build_scenario, pick_robustness, pick_targeting, semantic_id

__all__ = [
    "DEFAULT_DISTRIBUTION",
    "SCENARIOS",
    "build_area_spec",
    "build_row",
    "build_rows",
    "build_scenario",
    "load_distribution",
    "pick_robustness",
    "pick_targeting",
    "required_counts",
    "semantic_id",
    "validate_distribution",
]
