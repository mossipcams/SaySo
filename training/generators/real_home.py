"""Load a fetched Home Assistant home and split its entities train/holdout.

`scripts/fetch_ha_home.py` writes one real home. Generating over it teaches the
model a real topology -- irregular capitalization, brand model numbers, an area
with one light and an area with eleven switches -- that synthetic homes only
approximate.

One home is also 74 names, so every row drawn from it repeats them. Two things
keep that from becoming Run 008's memorization failure:

1. The caller decides the mix (`GeneratorConfig.real_home_rate`); real rows are
   a seasoning on the synthetic corpus, not a phase of their own.
2. A fifth of the entities never enter training at all. Evaluating on those is
   the only way to tell "learned this home" apart from "learned homes".
"""

from __future__ import annotations

import copy
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

# Every fifth entity within a capability is held out. Per capability rather than
# over the whole home so a six-entity domain does not land wholly on one side.
HOLDOUT_STRIDE = 5
SPLITS = ("train", "holdout")


def split_entities(
    entities: list[dict[str, Any]], split: str
) -> list[dict[str, Any]]:
    """Partition entities deterministically by capability position."""
    if split not in SPLITS:
        msg = f"split must be one of {SPLITS}, got {split!r}"
        raise ValueError(msg)
    by_capability: dict[str, list[dict[str, Any]]] = {}
    for entity in sorted(entities, key=lambda e: e["entity_id"]):
        by_capability.setdefault(entity["capability"], []).append(entity)

    selected: list[dict[str, Any]] = []
    for group in by_capability.values():
        # A lone entity of its capability stays in training: holding it out
        # would remove the capability from the corpus to test one name.
        held = [e for i, e in enumerate(group) if i % HOLDOUT_STRIDE == 0] if len(group) > 1 else []
        wanted = held if split == "holdout" else [e for e in group if e not in held]
        selected.extend(wanted)
    return sorted(selected, key=lambda e: e["entity_id"])


# A snapshot may mix in only when Home Assistant's own Assist exposure list was
# applied. "domain_filter" means the exporter guessed from the domain, which can
# put an entity the assistant cannot see into the corpus.
LIVE_EXPOSURE_SOURCES = frozenset({"assist_exposure"})
TESTABLE_EXPOSURE_SOURCES = frozenset({"assist_exposure", "synthetic_fixture"})


def exposure_source(path: str | Path) -> str:
    return json.loads(Path(path).read_text(encoding="utf-8")).get("exposure_source", "unknown")


def require_exposure_source(path: str | Path, *, allowed: frozenset[str] = LIVE_EXPOSURE_SOURCES) -> None:
    """Fail loudly rather than train a home recipe on a stale or unfiltered export."""
    source = exposure_source(path)
    if source not in allowed:
        raise ValueError(
            f"{path} has exposure_source={source!r}; the home-specific recipe needs one of "
            f"{sorted(allowed)}. Refresh it with scripts/fetch_ha_home.py against the live "
            "Home Assistant instance."
        )


@lru_cache(maxsize=4)
def _load(path: str, split: str) -> dict[str, Any]:
    home = json.loads(Path(path).read_text(encoding="utf-8"))
    entities = split_entities(home["entities"], split)
    if not entities:
        msg = f"{path} has no entities in split {split!r}"
        raise ValueError(msg)
    areas = {entity["area"] for entity in entities}
    return {
        **home,
        "home_id": f"{home['home_id']}_{split}",
        "size": len(entities),
        "entities": entities,
        "sayso_entity_area": (
            home["sayso_entity_area"] if home["sayso_entity_area"] in areas else entities[0]["area"]
        ),
    }


def load_real_home(path: str | Path, *, split: str = "train") -> dict[str, Any]:
    """Return a deep copy of the real home restricted to `split`.

    A copy per call because `scenarios.build_scenario` appends injected entities
    to the home it is given; a shared dict would accumulate them across rows.
    """
    return copy.deepcopy(_load(str(path), split))


# A real home's capabilities are not evenly populated: 34 switches share the
# switch quota while a lone thermostat absorbs every climate row. Without a cap
# the singleton capabilities -- climate, fan, scene -- each memorize one name.
DEFAULT_CAP_MULTIPLIER = 4


def derive_entity_cap(
    count: int,
    rate: float,
    entity_count: int,
    *,
    multiplier: int = DEFAULT_CAP_MULTIPLIER,
) -> int:
    """Cap on how often one real entity may be a target label.

    Anchored on the fair share -- real rows divided evenly across real entities
    -- times a multiplier, so popular entities stay over-represented but a
    singleton capability cannot claim its whole quota. Scales with `count`, so
    a smoke run and a 40k run cap at the same proportion.
    """
    if rate <= 0 or entity_count <= 0:
        return 0
    fair_share = math.ceil(count * rate / entity_count)
    return max(1, fair_share * multiplier)


def holdout_entity_names(path: str | Path) -> frozenset[str]:
    """Names reserved for evaluation, for contamination checks."""
    return frozenset(entity["name"] for entity in _load(str(path), "holdout")["entities"])
