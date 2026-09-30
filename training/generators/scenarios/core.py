
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
    operation_spec,
    required_features,
    trainable_operations,
)
from generators.gold import gold_from_scenario, target_names_from_expected
from generators.tools import namespaced_tool_name
from generators.utterances import _plural
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
        "unsupported_names": sorted(expected.get("unsupported_names") or []),
        "unavailable_tools": sorted(expected.get("unavailable_tools") or []),
        "removed_tools": sorted(scenario.get("removed_tools") or []),
        "expected_kind": expected.get("kind"),
        "expected_response": expected.get("response"),
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    return f"sem_{digest}"


def _ensure_supporting_in_area(
    home: dict[str, Any],
    capability: str,
    operation: str,
    area: str,
    minimum: int,
    rng: random.Random,
    index: int,
) -> None:
    areas, floors = home_areas(home)
    floor = floors.get(area, "Main Floor")
    owners = tuple(home.get("owners") or ())
    taken = {e["entity_id"].split(".", 1)[1] for e in home["entities"]}
    while True:
        in_area = [
            entity
            for entity in entities_of_capability(home, capability)
            if entity["area"] == area
        ]
        supporting = entities_supporting(in_area, capability, operation)
        if len(supporting) >= minimum:
            return
        name = _random_entity_name(
            capability,
            area,
            len(in_area),
            index,
            rng,
            taken=taken,
            **({"owners": owners} if owners else {}),
        )
        taken.add(_slug(name))
        needed = set(_ENTITY_TEMPLATES[capability][2]) | set(required_features(capability, operation))
        home["entities"].append(
            make_entity(
                name=name,
                capability=capability,
                area=area,
                floor=floor,
                rng=rng,
                features=tuple(sorted(needed)),
            )
        )


def _usable_aliases(entity: dict[str, Any], home: dict[str, Any]) -> list[str]:
    other_names = {
        name.casefold()
        for other in home["entities"]
        if other is not entity
        for name in [other["name"], *other.get("aliases", [])]
    }
    return [
        alias
        for alias in entity.get("aliases", [])
        if alias.casefold() != entity["name"].casefold()
        and alias.casefold() not in other_names
    ]


def _slug(name: str) -> str:
    return "".join(char.casefold() if char.isalnum() else "_" for char in name).strip("_")


def configure_family_scenario(
    home: dict[str, Any],
    capability: str,
    operation: str,
    family: str,
    rng: random.Random,
    index: int,
) -> tuple[str, str, dict[str, Any] | None, list[str]]:
    area = home["sayso_entity_area"]
    removed_tools: list[str] = []
    request_intent: dict[str, Any] | None = None

    if family in {"clarify", "junk", "follow_up"}:
        _ensure_supporting_in_area(home, capability, operation, area, 2, rng, index)
        return "area", "ambiguity", request_intent, removed_tools

    if family == "correction":
        return "individual", "ordinary", request_intent, removed_tools

    if family == "absence":
        home["entities"] = [
            entity
            for entity in home["entities"]
            if not (entity["capability"] == capability and entity["area"] == area)
        ]
        return "area", "ambiguity", request_intent, removed_tools

    if family == "unavailable":
        op = operation_spec(capability, operation)
        if op and op.support == SupportLevel.UNAVAILABLE:
            return "individual", "unavailable", request_intent, removed_tools
        if op and op.tool_name:
            removed_tools = [op.tool_name]
        withheld = {namespaced_tool_name(tool) for tool in removed_tools}
        for sibling_capability, sibling_tool in (
            ("lights", "light__HassLightSet"),
            ("media_players", "media_player__HassSetVolume"),
        ):
            if sibling_tool in withheld:
                continue
            if any(
                entity["capability"] == sibling_capability for entity in home["entities"]
            ):
                continue
            areas, floors = home_areas(home)
            floor = floors.get(area, "Main Floor")
            owners = tuple(home.get("owners") or ())
            taken = {e["entity_id"].split(".", 1)[1] for e in home["entities"]}
            name = _random_entity_name(
                sibling_capability,
                area,
                0,
                index,
                rng,
                taken=taken,
                **({"owners": owners} if owners else {}),
            )
            home["entities"].append(
                make_entity(
                    name=name,
                    capability=sibling_capability,
                    area=area,
                    floor=floor,
                    rng=rng,
                    features=_ENTITY_TEMPLATES[sibling_capability][2],
                )
            )
        return "individual", "unavailable", request_intent, removed_tools

    if family in {"multi_action", "exclusion"}:
        _ensure_supporting_in_area(home, capability, operation, area, 3 if family == "exclusion" else 2, rng, index)
        return "multiple", family, request_intent, removed_tools

    if family == "unsupported":
        areas, floors = home_areas(home)
        floor = floors.get(area, "Main Floor")
        home["entities"] = [
            entity
            for entity in home["entities"]
            if not (entity["capability"] == capability and entity["area"] == area)
        ]
        _, _, default_features = _ENTITY_TEMPLATES[capability]
        needed = set(required_features(capability, operation))
        incapable_features = tuple(feature for feature in default_features if feature not in needed)
        owners = tuple(home.get("owners") or ())
        taken = {e["entity_id"].split(".", 1)[1] for e in home["entities"]}
        name = _random_entity_name(
            capability,
            area,
            0,
            index,
            rng,
            taken=taken,
            **({"owners": owners} if owners else {}),
        )
        home["entities"].append(
            make_entity(
                name=name,
                capability=capability,
                area=area,
                floor=floor,
                rng=rng,
                features=incapable_features or default_features[:1],
            )
        )
        return "area", "ambiguity", request_intent, removed_tools

    if family == "aliases":
        return "individual", "alias_distractor", request_intent, removed_tools

    return "individual", "ordinary", request_intent, removed_tools


_FAMILY_GRAPH_CONSTRAINTS = frozenset({
    "clarify",
    "follow_up",
    "correction",
    "junk",
    "absence",
    "unavailable",
    "unsupported",
    "multi_action",
    "exclusion",
    "aliases",
})


def pick_target(
    cap_entities: list[dict[str, Any]], index: int, usage: Counter[str] | None
) -> dict[str, Any]:
    rotation = index % len(cap_entities)
    rotated = cap_entities[rotation:] + cap_entities[:rotation]
    if usage is None:
        return rotated[0]
    return min(rotated, key=lambda entity: usage[entity["name"]])


def _near_miss_score(entity: dict[str, Any], in_area: list[dict[str, Any]]) -> int:
    tokens = set(entity["name"].lower().split())
    return max(
        (len(tokens & set(other["name"].lower().split()))
         for other in in_area if other is not entity),
        default=0,
    )


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
    request_intent: dict[str, Any] | None = None,
    family: str | None = None,
    bare_name_rate: float = 0.0,
) -> dict[str, Any]:
    rng = random.Random(
        (seed << 20)
        ^ (index << 8)
        ^ (attempt << 4)
        ^ zlib.crc32(capability.encode())
        ^ zlib.crc32(operation.encode())
    )
    if home is None:
        home = generate_home(index, home_size, rng, bare_name_rate=bare_name_rate)
    if family == "datetime":
        scenario: dict[str, Any] = {
            "scenario_index": index,
            "attempt": attempt,
            "seed": seed,
            "split": split,
            "capability": "datetime",
            "operation": "query_time",
            "targeting": "context",
            "robustness": "datetime",
            "request_intent": None,
            "family": "datetime",
            "removed_tools": [],
            "home": home,
            "target_entity": None,
            "target_index": 0,
            "area": home["sayso_entity_area"],
            "floor": None,
            "excluded_names": [],
            "provenance": {
                "generator": "sayso_synthetic_v3",
                "capability": "datetime",
                "operation": "query_time",
                "home_size": home_size,
            },
        }
        scenario["expected"] = gold_from_scenario(scenario, rng)
        scenario["semantic_id"] = semantic_id(scenario)
        scenario["tier"] = 0
        return scenario
    removed_tools: list[str] = []
    if family in _FAMILY_GRAPH_CONSTRAINTS:
        targeting, robustness, request_intent, removed_tools = configure_family_scenario(
            home, capability, operation, family, rng, index,
        )
        if family in {"absence", "unsupported"}:
            inject_missing = False
    cap_entities = entities_of_capability(home, capability)
    if capability == "timers":
        cap_entities = []
    else:
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
    if robustness == "alias_distractor":
        usable = [entity for entity in cap_entities if _usable_aliases(entity, home)]
        if usable:
            cap_entities = usable
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
        "request_intent": request_intent,
        "family": family,
        "removed_tools": removed_tools,
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
        in_area = [e for e in cap_entities if e["area"] == home["sayso_entity_area"]]
        if len(in_area) >= 3 and rng.random() < 0.6:
            top = max(_near_miss_score(entity, in_area) for entity in in_area)
            kept = rng.choice([e for e in in_area if _near_miss_score(e, in_area) == top])
            scenario["target_entities"] = [e for e in in_area if e is not kept]
            scenario["excluded_names"] = [kept["name"]]
            scenario["exclusion_scope"] = f"the {kept['area'].lower()} {_plural(kept['domain'])}"
            scenario["spoken_targets"] = {e["name"]: scenario["exclusion_scope"] for e in scenario["target_entities"]}
        else:
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
        siblings = [
            entity
            for entity in home["entities"]
            if entity is not target_entity
            and entity["domain"] == target_entity["domain"]
            and entity["area"] == target_entity["area"]
        ]
        if not siblings:
            floors = home_areas(home)[1]
            floor = floors.get(target_entity["area"], "Main Floor")
            owners = tuple(home.get("owners") or ())
            in_area = [
                entity
                for entity in home["entities"]
                if entity["domain"] == target_entity["domain"]
                and entity["area"] == target_entity["area"]
            ]
            taken = {e["entity_id"].split(".", 1)[1] for e in home["entities"]}
            name = _random_entity_name(
                capability,
                target_entity["area"],
                len(in_area),
                index,
                rng,
                taken=taken,
                **({"owners": owners} if owners else {}),
            )
            home["entities"].append(
                make_entity(
                    name=name,
                    capability=capability,
                    area=target_entity["area"],
                    floor=floor,
                    rng=rng,
                    features=_ENTITY_TEMPLATES[capability][2],
                )
            )
            remove_canonical_alias_collisions(home["entities"])
        aliases = _usable_aliases(target_entity, home)
        if aliases:
            scenario["spoken_targets"] = {target_entity["name"]: rng.choice(aliases)}
    scenario["semantic_id"] = semantic_id(scenario)
    scenario["tier"] = CAPABILITIES[capability].tier
    return scenario


def pick_targeting(cap: CapabilitySpec, rng: random.Random, robustness: str) -> str:
    if robustness in {"multi_action", "exclusion"}:
        return "multiple"
    if robustness in {"ambiguity", "unavailable"}:
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
