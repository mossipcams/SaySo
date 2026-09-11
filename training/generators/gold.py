"""Deterministic gold-label generation from structured scenarios."""

from __future__ import annotations

import random
from typing import Any

from generators.capability_registry import (
    CAPABILITIES,
    OperationSpec,
    SupportLevel,
    entities_supporting,
    entity_supports,
    trainable_operations,
)
from generators.homes import entities_in_area, entities_of_capability
from generators.tools import build_call_for_operation


def expected_action(
    entity: dict[str, Any],
    operation: str,
    rng: random.Random,
    *,
    area: str | None = None,
    floor: str | None = None,
) -> dict[str, Any]:
    capability = entity.get("capability", "")
    call = build_call_for_operation(entity, capability, operation, rng, area=area, floor=floor)
    payload: dict[str, Any] = {"kind": "action", "calls": [call]}
    if capability == "scripts":
        # A script tool takes no arguments, so the friendly name has to ride alongside
        # the call for utterance generation and target checks.
        payload["script_targets"] = [entity["name"]]
    return payload


def expected_status(entity: dict[str, Any]) -> dict[str, Any]:
    from generators.tools import build_query

    return {
        "kind": "status",
        "calls": [build_query(entity)],
        "state": entity["state"],
    }


def expected_no_action(response: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"kind": "no_action", "response": response, "calls": []}
    payload.update(extra)
    return payload


def expected_device_unsupported(
    entities: list[dict[str, Any]],
    capability: str,
    operation: str,
    rng: random.Random,
) -> dict[str, Any]:
    """The devices are there, exposed, and cannot do it.

    Distinct from ``area_unavailable`` (nothing of that type in the room) and from
    ``unsupported`` (Home Assistant did not supply the tool at all): here the tool
    is offered and the entity's supported_features simply lack the action.
    """
    target = entities[0]
    requested = {
        "kind": "action",
        "calls": [build_call_for_operation(target, capability, operation, rng)],
    }
    return expected_no_action(
        "device_unsupported",
        requested=requested,
        unsupported_names=[entity["name"] for entity in entities],
    )


def gold_from_scenario(scenario: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Derive authoritative expected behavior from a structured scenario."""
    home = scenario["home"]
    capability = scenario["capability"]
    operation = scenario["operation"]
    targeting = scenario.get("targeting", "individual")
    cap_spec = CAPABILITIES[capability]
    op_spec = next((op for op in cap_spec.operations if op.name == operation), None)

    if scenario.get("robustness") == "unsupported" and op_spec and op_spec.support != SupportLevel.UNAVAILABLE:
        requested = gold_from_scenario(
            {**scenario, "robustness": "ordinary", "targeting": "individual"}, rng
        )
        return expected_no_action(
            "unsupported", requested=requested,
            unavailable_tools=list(dict.fromkeys(call["name"] for call in requested.get("calls", []))),
        )

    if op_spec and op_spec.support == SupportLevel.UNAVAILABLE:
        return expected_no_action(
            "unsupported",
            blocker=op_spec.blocker if op_spec else cap_spec.blocker,
        )

    if capability == "timers" and op_spec is not None and op_spec.support is SupportLevel.SUPPORTED:
        area = scenario.get("area")
        call = build_call_for_operation(None, capability, operation, rng, area=area)
        if operation != "cancel_all":
            call["arguments"]["name"] = rng.choice((
                "tea", "rice", "pasta", "bread", "coffee", "workout",
                "homework", "stretching", "watering", "roast", "cookies", "meditation",
            ))
        return {"kind": "action", "calls": [call]}

    if scenario.get("robustness") == "ambiguity":
        return _ambiguous_gold(home, capability, operation, rng)

    if operation == "query_state":
        entity = _pick_entity(scenario, rng)
        if entity is None:
            return expected_no_action("clarify")
        return expected_status(entity)

    if targeting == "area":
        area = scenario.get("area") or home["sayso_entity_area"]
        present = entities_in_area(home, capability, area)
        matches = entities_supporting(present, capability, operation)
        if not matches:
            if present:
                return expected_device_unsupported(present, capability, operation, rng)
            return expected_no_action(
                "area_unavailable",
                unavailable={"area": area.casefold(), "type": _type_label(capability)},
            )
        if len(matches) == 1:
            return expected_action(matches[0], operation, rng)
        # Area with multiple: single area-targeted call
        call = build_call_for_operation(None, capability, operation, rng, area=area)
        return {"kind": "action", "calls": [call]}

    if targeting == "floor":
        floor = scenario.get("floor") or "Upstairs"
        on_floor = [e for e in entities_of_capability(home, capability) if e["floor"] == floor]
        matches = entities_supporting(on_floor, capability, operation)
        if not matches:
            return expected_no_action("clarify")
        call = build_call_for_operation(None, capability, operation, rng, floor=floor)
        return {"kind": "action", "calls": [call]}

    if targeting == "multiple":
        targets = scenario.get("target_entities") or entities_supporting(
            entities_of_capability(home, capability), capability, operation
        )[:2]
        if not targets:
            return expected_no_action("clarify")
        calls, script_targets = [], []
        for entity in targets:
            entity_cap = entity["capability"]
            chosen_op = operation
            if entity_cap != capability or not entity_supports(entity, entity_cap, operation):
                usable = [
                    op for op in trainable_operations(CAPABILITIES[entity_cap])
                    if entity_supports(entity, entity_cap, op.name)
                ]
                if not usable:
                    return expected_no_action("clarify")
                chosen_op = rng.choice(usable).name
            action = expected_action(entity, chosen_op, rng)
            calls.extend(action["calls"])
            script_targets.extend(action.get("script_targets", []))
        return {"kind": "action", "calls": calls, "script_targets": script_targets}

    entity = _pick_entity(scenario, rng)
    if entity is None:
        return expected_no_action("clarify")
    return expected_action(entity, operation, rng)


def _pick_entity(scenario: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    capability = scenario["capability"]
    operation = scenario["operation"]
    target = scenario.get("target_entity")
    if target and entity_supports(target, capability, operation):
        return target
    home = scenario["home"]
    matches = entities_supporting(entities_of_capability(home, capability), capability, operation)
    if not matches:
        return None
    return matches[scenario.get("target_index", 0) % len(matches)]


def _ambiguous_gold(
    home: dict[str, Any],
    capability: str,
    operation: str,
    rng: random.Random,
) -> dict[str, Any]:
    if capability == "scripts":
        # Script tools expose friendly names, not their hidden HA area assignments.
        return expected_no_action("clarify")
    area = home["sayso_entity_area"]
    present = entities_in_area(home, capability, area)
    matches = entities_supporting(present, capability, operation)
    if len(matches) == 0:
        if present:
            return expected_device_unsupported(present, capability, operation, rng)
        return expected_no_action(
            "area_unavailable",
            unavailable={"area": area.casefold(), "type": _type_label(capability)},
        )
    if len(matches) == 1:
        return expected_status(matches[0]) if operation == "query_state" else expected_action(matches[0], operation, rng)
    return expected_no_action("clarify")


def _type_label(capability: str) -> str:
    labels = {
        "lights": "lights",
        "fans": "fans",
        "switches": "outlets",
        "covers": "blinds",
        "locks": "doors",
        "media_players": "media players",
        "climate": "thermostats",
        "scripts": "scripts",
        "scenes": "scenes",
        "vacuums": "vacuums",
        "timers": "timers",
        "buttons": "buttons",
        "todo_lists": "shopping lists",
        "lawn_mowers": "lawn mowers",
    }
    return labels.get(capability, "devices")


def target_names_from_expected(expected: dict[str, Any]) -> list[str]:
    names: list[str] = []
    scripts = iter(expected.get("script_targets") or [])
    for call in expected.get("calls") or []:
        args = call.get("arguments") or {}
        if isinstance(args.get("name"), str):
            names.append(args["name"])
        elif not call["name"].startswith(("Hass", "Get")):
            names.append(next(scripts))
    return names
