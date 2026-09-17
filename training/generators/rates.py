"""Fail-closed rate gates and reachable-share ceilings for cross-cutting rows."""

from __future__ import annotations

from typing import Any

from generators.config import GeneratorConfig
from generators.grounding import (
    grounding_available_share,
    grounding_capacity,
    grounding_pairs,
    slot_ceiling,
)
from generators.planning import discrimination_available_share

# Below this many accepted rows a percentage rate is not meaningfully assertable.
RATE_GATE_MIN_ROWS = 1000
# Measured delivery band on a 2.8% grounding request (2.4–2.8%).
RATE_ACHIEVED_TOLERANCE = 0.85


def enforce_rate_gate(
    section: dict[str, Any],
    name: str,
    config: GeneratorConfig,
    accepted: int,
    available_share: float = 1.0,
) -> None:
    """Fail the build when a requested row share was not actually achieved."""
    requested = section["requested_rate"]
    if requested <= 0 or accepted < RATE_GATE_MIN_ROWS:
        return
    ceiling = min(requested, available_share)
    floor = ceiling
    if available_share + 1e-9 >= requested:
        floor = ceiling * RATE_ACHIEVED_TOLERANCE
    if section["achieved_rate"] + 1e-9 < floor:
        raise RuntimeError(
            f"{name} rate shortfall: requested {requested:.4f}, "
            f"reachable ceiling {ceiling:.4f}, achieved "
            f"{section['achieved_rate']:.4f} ({section['rows']}/{accepted} rows); "
            f"floor is {floor:.4f} "
            "(fail-closed quota)"
        )


__all__ = [
    "RATE_ACHIEVED_TOLERANCE",
    "RATE_GATE_MIN_ROWS",
    "discrimination_available_share",
    "enforce_rate_gate",
    "grounding_available_share",
    "grounding_capacity",
    "grounding_pairs",
    "slot_ceiling",
]
