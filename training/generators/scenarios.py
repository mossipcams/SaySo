"""Structured scenario generation before utterances."""

from __future__ import annotations

import hashlib
import json
import random
import zlib
from collections import Counter
from typing import Any

from generators.capability_registry import (
    CAPABILITIES,
    CapabilitySpec,
    SupportLevel,
    entities_supporting,
    entity_supports,
    required_features,
    trainable_operations,
)
from generators.gold import gold_from_scenario, target_names_from_expected
from generators.homes import (
    _ENTITY_TEMPLATES,
    _random_entity_name,
    device_areas,
    entities_of_capability,
    generate_home,
    make_entity,
    remove_canonical_alias_collisions,
)


def home_areas(home: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    """Areas and their floors, derived from the entities when the home omits them.

    ``generate_home`` records both; a home exported from a live Home Assistant by
    an older ``fetch_ha_home`` does not, and injecting a missing capability into
    one used to raise KeyError mid-run.
    """
    areas = home.get("areas")
    floors = home.get("area_floors")
    if areas and floors:
        return list(areas), dict(floors)
    derived: dict[str, str] = {}
    for entity in home.get("entities", []):
        derived.setdefault(entity["area"], entity.get("floor") or "Main Floor")
    if not derived:
        derived[home.get("sayso_entity_area") or "Living Room"] = "Main Floor"
    return list(derived), derived


def semantic_id(scenario: dict[str, Any]) -> str:
    """Stable ID from meaningful scenario content (not utterance or attempt index)."""
    expected = scenario.get("expected") or {}
    target_ids: list[str] = []
    if scenario.get("target_entity"):
        entity_id = scenario["target_entity"].get("entity_id")
        if entity_id:
            target_ids.append(entity_id)
    for entity in scenario.get("target_entities") or []:
        entity_id = entity.get("entity_id")
        if entity_id:
            target_ids.append(entity_id)
    key = {
        "capability": scenario.get("capability"),
        "operation": scenario.get("operation"),
        "targeting": scenario.get("targeting"),
        "robustness": scenario.get("robustness"),
        "home_id": scenario.get("home", {}).get("home_id"),
        "target_ids": sorted(target_ids),
        "target_names": sorted(target_names_from_expected(expected)),
        "expected_kind": expected.get("kind"),
        "expected_response": expected.get("response"),
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    return f"sem_{digest}"


def pick_target(
    cap_entities: list[dict[str, Any]], index: int, usage: Counter[str] | None
) -> dict[str, Any]:
    """Rotate targets by row index, or take the least-used entity when tracking usage.

    ``index`` is the global accepted-row count, so plain rotation spreads targets
    across a *synthetic* home well -- every row draws a fresh home anyway. A real
    home is one fixed set of names reused all run, and only a handful of its rows
    land on any one operation, so rotation covers a specific device/operation pair
    by luck: the TV would get ``turn_off`` and never ``turn_on``. Counting how often
    each entity has already been the target *for this operation* makes that coverage
    structural. Ties keep the rotation order, so a seed still reproduces exactly.
    """
    rotation = index % len(cap_entities)
    rotated = cap_entities[rotation:] + cap_entities[:rotation]
    if usage is None:
        return rotated[0]
    return min(rotated, key=lambda entity: usage[entity["name"]])


def build_scenario(
    *,
    index: int,
    seed: int,
    capability: str,
    operation: str,
    home_size: int,
    targeting: str = "individual",
    robustness: str = "ordinary",
    split: str = "train",
    attempt: int = 0,
    home: dict[str, Any] | None = None,
    inject_missing: bool = True,
    target_usage: Counter[str] | None = None,
) -> dict[str, Any]:
    """Build one scenario. `home` overrides synthetic generation and is mutated
    (missing capabilities get an injected entity), so callers pass a fresh copy.

    ``inject_missing=False`` leaves the graph exactly as supplied. A grounding pair
    that tests absence needs that: injecting the very device whose absence is the
    point would turn the refusal back into an action.
    """
    # crc32, not builtin hash(): str hashing is randomized per process and would
    # make the same seed generate a different dataset on every run.
    rng = random.Random(
        (seed << 20)
        ^ (index << 8)
        ^ (attempt << 4)
        ^ zlib.crc32(capability.encode())
        ^ zlib.crc32(operation.encode())
    )
    if home is None:
        home = generate_home(index, home_size, rng)
    cap_entities = entities_of_capability(home, capability)
    if capability == "timers":
        cap_entities = []
    else:
        # Only entities that can actually perform the operation are candidate
        # targets; the rest stay in the home as distractors.
        cap_entities = entities_supporting(cap_entities, capability, operation)
        if inject_missing and not cap_entities and operation not in {"cancel_all"}:
            areas, floors = home_areas(home)
            allowed = device_areas(capability, areas) or areas
            area = rng.choice(allowed)
            floor = floors.get(area, "Main Floor")
            owners = tuple(home.get("owners") or ())
            name = _random_entity_name(
                capability, area, 0, index, rng,
                taken={e["entity_id"].split(".", 1)[1] for e in home["entities"]},
                **({"owners": owners} if owners else {}),
            )
            needed = set(_ENTITY_TEMPLATES[capability][2]) | set(required_features(capability, operation))
            injected = make_entity(
                name=name,
                capability=capability,
                area=area,
                floor=floor,
                rng=rng,
                features=tuple(sorted(needed)),
            )
            home["entities"].append(injected)
            cap_entities = [injected]
    remove_canonical_alias_collisions(home["entities"])
    target_entity = pick_target(cap_entities, index, target_usage) if cap_entities else None
    scenario: dict[str, Any] = {
        "scenario_index": index,
        "attempt": attempt,
        "seed": seed,
        "split": split,
        "capability": capability,
        "operation": operation,
        "targeting": targeting,
        "robustness": robustness,
        "home": home,
        "target_entity": target_entity,
        "target_index": (
            next(i for i, e in enumerate(cap_entities) if e is target_entity)
            if target_entity else 0
        ),
        "area": target_entity["area"] if target_entity else home["sayso_entity_area"],
        "floor": target_entity["floor"] if target_entity else None,
        "excluded_names": [],
        "provenance": {
            "generator": "sayso_synthetic_v3",
            "capability": capability,
            "operation": operation,
            "home_size": home_size,
        },
    }
    if robustness == "exclusion" and len(cap_entities) >= 2:
        scenario["target_entities"] = cap_entities[:2]
        scenario["excluded_names"] = [cap_entities[2]["name"]] if len(cap_entities) > 2 else []
    if robustness == "multi_action" and len(cap_entities) >= 2:
        others = [e for e in home["entities"]
                  if capability != "scripts" and e["capability"] != capability
                  and trainable_operations(CAPABILITIES[e["capability"]])]
        scenario["target_entities"] = [target_entity, rng.choice(others)] if others else cap_entities[:2]
        scenario["targeting"] = "multiple"
    scenario["expected"] = gold_from_scenario(scenario, rng)
    if robustness == "alias_distractor" and target_entity:
        # Use only an unambiguous HA alias; retain the canonical name in labels.
        other_names = {
            name.casefold() for entity in home["entities"] if entity is not target_entity
            for name in [entity["name"], *entity.get("aliases", [])]
        }
        aliases = [alias for alias in target_entity.get("aliases", [])
                   if alias.casefold() != target_entity["name"].casefold()
                   and alias.casefold() not in other_names]
        if aliases:
            scenario["spoken_targets"] = {target_entity["name"]: rng.choice(aliases)}
    scenario["semantic_id"] = semantic_id(scenario)
    scenario["tier"] = CAPABILITIES[capability].tier
    return scenario


def pick_operation(cap: CapabilitySpec, rng: random.Random, *, prefer_trainable: bool = True) -> str:
    if prefer_trainable:
        ops = trainable_operations(cap)
        if ops:
            return rng.choice(ops).name
    supported = [op for op in cap.operations if op.support != SupportLevel.UNAVAILABLE]
    if supported:
        return rng.choice(supported).name
    return cap.operations[0].name


def pick_targeting(cap: CapabilitySpec, rng: random.Random, robustness: str) -> str:
    if robustness in {"multi_action", "exclusion"}:
        return "multiple"
    if robustness == "ambiguity":
        return "area"
    modes = list(cap.targeting_modes)
    if "individual" in modes:
        return rng.choices(modes, weights=[3] + [1] * (len(modes) - 1), k=1)[0]
    return rng.choice(modes)


def pick_robustness(rng: random.Random, ordinary_rate: float = 0.75) -> str:
    if rng.random() < ordinary_rate:
        return "ordinary"
    return rng.choice(
        ("alias_distractor", "similar_name", "large_home", "multi_action", "exclusion", "ambiguity", "unsupported")
    )
