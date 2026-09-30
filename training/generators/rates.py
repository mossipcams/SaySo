
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

RATE_GATE_MIN_ROWS = 1000
RATE_ACHIEVED_TOLERANCE = 0.85


def enforce_rate_gate(
    section: dict[str, Any],
    name: str,
    config: GeneratorConfig,
    accepted: int,
    available_share: float = 1.0,
) -> None:
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
