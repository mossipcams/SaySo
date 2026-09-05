"""Deterministic gold-label generation from structured scenarios."""

from __future__ import annotations

import random
from typing import Any

from generators.capability_registry import CAPABILITIES, OperationSpec, SupportLevel
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
    return {"kind": "action", "calls": [call]}


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


def gold_from_scenario(scenario: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Derive authoritative expected behavior from a structured scenario."""
    home = scenario["home"]
    capability = scenario["capability"]
    operation = scenario["operation"]
    targeting = scenario.get("targeting", "individual")
    cap_spec = CAPABILITIES[capability]
    op_spec = next((op for op in cap_spec.operations if op.name == operation), None)

    if scenario.get("robustness") == "unsupported" or (
        op_spec and op_spec.support == SupportLevel.UNAVAILABLE
    ):
        return expected_no_action(
            "unsupported",
            blocker=op_spec.blocker if op_spec else cap_spec.blocker,
        )

    if scenario.get("robustness") == "ambiguity":
        return _ambiguous_gold(home, capability, operation, rng)

    if operation == "query_state":
        entity = _pick_entity(scenario, rng)
        if entity is None:
            return expected_no_action("clarify")
        return expected_status(entity)

    if operation == "cancel_all":
        area = scenario.get("area")
        call = build_call_for_operation(None, capability, operation, rng, area=area)
        return {"kind": "action", "calls": [call]}

    if targeting == "area":
        area = scenario.get("area") or home["sayso_entity_area"]
        matches = entities_in_area(home, capability, area)
        if not matches:
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
        matches = [e for e in entities_of_capability(home, capability) if e["floor"] == floor]
        if not matches:
            return expected_no_action("clarify")
        area = matches[0]["area"]
        call = build_call_for_operation(None, capability, operation, rng, area=area, floor=floor)
        return {"kind": "action", "calls": [call]}

    if targeting == "multiple":
        targets = scenario.get("target_entities") or entities_of_capability(home, capability)[:2]
        calls = [build_call_for_operation(e, capability, operation, rng) for e in targets]
        return {"kind": "action", "calls": calls}

    entity = _pick_entity(scenario, rng)
    if entity is None:
        return expected_no_action("clarify")
    return expected_action(entity, operation, rng)


def _pick_entity(scenario: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    if scenario.get("target_entity"):
        return scenario["target_entity"]
    capability = scenario["capability"]
    home = scenario["home"]
    matches = entities_of_capability(home, capability)
    if not matches:
        return None
    return matches[scenario.get("target_index", 0) % len(matches)]


def _ambiguous_gold(
    home: dict[str, Any],
    capability: str,
    operation: str,
    rng: random.Random,
) -> dict[str, Any]:
    area = home["sayso_entity_area"]
    matches = entities_in_area(home, capability, area)
    if len(matches) == 0:
        return expected_no_action(
            "area_unavailable",
            unavailable={"area": area.casefold(), "type": _type_label(capability)},
        )
    if len(matches) == 1:
        return expected_action(matches[0], operation, rng)
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
    }
    return labels.get(capability, "devices")


def target_names_from_expected(expected: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for call in expected.get("calls") or []:
        args = call.get("arguments") or {}
        if isinstance(args.get("name"), str):
            names.append(args["name"])
    return names
