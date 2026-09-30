
from __future__ import annotations

import copy
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

HOLDOUT_STRIDE = 5
SPLITS = ("train", "holdout")


def split_entities(
    entities: list[dict[str, Any]], split: str
) -> list[dict[str, Any]]:
    if split not in SPLITS:
        msg = f"split must be one of {SPLITS}, got {split!r}"
        raise ValueError(msg)
    by_capability: dict[str, list[dict[str, Any]]] = {}
    for entity in sorted(entities, key=lambda e: e["entity_id"]):
        by_capability.setdefault(entity["capability"], []).append(entity)

    selected: list[dict[str, Any]] = []
    for group in by_capability.values():
        held = [e for i, e in enumerate(group) if i % HOLDOUT_STRIDE == 0] if len(group) > 1 else []
        wanted = held if split == "holdout" else [e for e in group if e not in held]
        selected.extend(wanted)
    return sorted(selected, key=lambda e: e["entity_id"])


LIVE_EXPOSURE_SOURCES = frozenset({"assist_exposure"})
TESTABLE_EXPOSURE_SOURCES = frozenset({"assist_exposure", "synthetic_fixture"})


def exposure_source(path: str | Path) -> str:
    return json.loads(Path(path).read_text(encoding="utf-8")).get("exposure_source", "unknown")


def require_exposure_source(path: str | Path, *, allowed: frozenset[str] = LIVE_EXPOSURE_SOURCES) -> None:
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
    for entity in home["entities"]:
        entity["name"] = " ".join(entity["name"].split())
        if entity.get("aliases"):
            entity["aliases"] = [" ".join(alias.split()) for alias in entity["aliases"]]
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
    return copy.deepcopy(_load(str(path), split))


DEFAULT_CAP_MULTIPLIER = 4


def derive_entity_cap(
    count: int,
    rate: float,
    entity_count: int,
    *,
    multiplier: int = DEFAULT_CAP_MULTIPLIER,
) -> int:
    if rate <= 0 or entity_count <= 0:
        return 0
    fair_share = math.ceil(count * rate / entity_count)
    return max(1, fair_share * multiplier)


def holdout_entity_names(path: str | Path) -> frozenset[str]:
    return frozenset(entity["name"] for entity in _load(str(path), "holdout")["entities"])
