"""Structured scenario generation before utterances."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from generators.capability_registry import (
    CAPABILITIES,
    CapabilitySpec,
    SupportLevel,
    trainable_operations,
)
from generators.gold import gold_from_scenario, target_names_from_expected
from generators.homes import entities_of_capability, generate_home, make_entity, _random_entity_name, _FLOORS


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
) -> dict[str, Any]:
    rng = random.Random((seed << 20) ^ (index << 8) ^ (attempt << 4) ^ hash(capability) ^ hash(operation))
    home = generate_home(index, home_size, rng)
    cap_entities = entities_of_capability(home, capability)
    if capability == "timers":
        cap_entities = []
    elif not cap_entities and operation not in {"cancel_all"}:
        area = home["sayso_entity_area"]
        floor = _FLOORS[(index + 1) % len(_FLOORS)]
        name = _random_entity_name(capability, area, 0, index, rng)
        injected = make_entity(
            name=name,
            capability=capability,
            area=area,
            floor=floor,
            rng=rng,
        )
        home["entities"].append(injected)
        cap_entities = [injected]
    target_entity = cap_entities[index % len(cap_entities)] if cap_entities else None
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
        "target_index": index % max(len(cap_entities), 1),
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
        scenario["target_entities"] = cap_entities[:2]
        scenario["targeting"] = "multiple"
    scenario["expected"] = gold_from_scenario(scenario, rng)
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
