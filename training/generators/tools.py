"""Runtime-equivalent tool call builders and availability."""

from __future__ import annotations

import random
from typing import Any

from adapters.schema import ALLOWED_HASS_TOOLS, v1_openai_tools
from generators.capability_registry import CAPABILITIES, CapabilitySpec, OperationSpec


def available_tools_for_home(home: dict[str, Any]) -> list[dict[str, Any]]:
    """Return full v1 tool catalog (runtime sends all schema tools regardless of home)."""
    return v1_openai_tools()


def _domain_args(entity: dict[str, Any]) -> dict[str, Any]:
    domain = entity["domain"]
    if domain in {"light", "fan", "switch", "media_player", "climate"}:
        return {"domain": [domain]}
    if entity.get("device_class"):
        return {"device_class": [entity["device_class"]]}
    return {}


def build_turn_on(entity: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {"name": entity["name"], **_domain_args(entity)}
    return {"name": "HassTurnOn", "arguments": args}


def build_turn_off(entity: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {"name": entity["name"], **_domain_args(entity)}
    return {"name": "HassTurnOff", "arguments": args}


def build_light_set(entity: dict[str, Any], rng: random.Random, operation: str) -> dict[str, Any]:
    args: dict[str, Any] = {"name": entity["name"], "domain": ["light"]}
    if operation == "set_brightness":
        args["brightness"] = rng.randrange(10, 101)
    elif operation == "set_color":
        args["color"] = rng.choice(("red", "blue", "warm white", "cool white"))
    elif operation == "set_color_temperature":
        args["temperature"] = rng.choice((2700, 3000, 4000, 5000))
    return {"name": "HassLightSet", "arguments": args}


def build_fan_speed(entity: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    return {
        "name": "HassFanSetSpeed",
        "arguments": {
            "name": entity["name"],
            "domain": ["fan"],
            "percentage": rng.randrange(10, 101),
        },
    }


def build_query(entity: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {}
    if entity.get("name"):
        args["name"] = entity["name"]
    domain = entity.get("domain")
    if domain in {"light", "fan", "switch", "climate", "media_player"}:
        args["domain"] = domain if isinstance(domain, str) else domain
    return {"name": "GetLiveContext", "arguments": args}


def build_cancel_all_timers(area: str | None = None) -> dict[str, Any]:
    args: dict[str, Any] = {}
    if area:
        args["area"] = area
    return {"name": "HassCancelAllTimers", "arguments": args}


def build_area_call(
    capability: str,
    operation: str,
    area: str,
    *,
    floor: str | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    cap = CAPABILITIES[capability]
    domain = cap.domain
    args: dict[str, Any] = {"area": area}
    if floor:
        args["floor"] = floor
    if domain in {"light", "fan", "switch", "media_player", "climate"}:
        args["domain"] = [domain]
    elif cap.device_class:
        args["device_class"] = [cap.device_class]
    tool = _operation_tool(operation, capability)
    if tool == "HassLightSet" and rng:
        args["brightness"] = rng.randrange(20, 90)
    return {"name": tool, "arguments": args}


def _operation_tool(operation: str, capability: str) -> str:
    mapping = {
        "turn_on": "HassTurnOn",
        "open": "HassTurnOn",
        "lock": "HassTurnOn",
        "turn_off": "HassTurnOff",
        "close": "HassTurnOff",
        "unlock": "HassTurnOff",
        "set_brightness": "HassLightSet",
        "set_color": "HassLightSet",
        "set_color_temperature": "HassLightSet",
        "set_speed": "HassFanSetSpeed",
        "cancel_all": "HassCancelAllTimers",
        "query_state": "GetLiveContext",
    }
    return mapping[operation]


def build_call_for_operation(
    entity: dict[str, Any] | None,
    capability: str,
    operation: str,
    rng: random.Random,
    *,
    area: str | None = None,
    floor: str | None = None,
) -> dict[str, Any]:
    if operation == "cancel_all":
        return build_cancel_all_timers(area)
    if operation == "query_state":
        if entity is None:
            cap = CAPABILITIES[capability]
            return {"name": "GetLiveContext", "arguments": {"domain": cap.domain}}
        return build_query(entity)
    if entity is None and area:
        return build_area_call(capability, operation, area, floor=floor, rng=rng)
    if entity is None:
        raise ValueError("entity or area required for operation")
    if operation in {"turn_on", "open", "lock"}:
        return build_turn_on(entity)
    if operation in {"turn_off", "close", "unlock"}:
        return build_turn_off(entity)
    if operation.startswith("set_") and capability == "lights":
        return build_light_set(entity, rng, operation)
    if operation == "set_speed":
        return build_fan_speed(entity, rng)
    raise ValueError(f"unsupported operation {operation!r} for {capability!r}")


def validate_call_tool_name(name: str) -> bool:
    return name in ALLOWED_HASS_TOOLS
