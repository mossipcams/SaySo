"""Runtime-equivalent tool call builders and availability."""

from __future__ import annotations

import random
import zlib
from typing import Any

from adapters.schema import ALLOWED_HASS_TOOLS, v2_openai_tools
from generators.capability_registry import CAPABILITIES

# Home Assistant supplies only the tools exposed for a request, so a row offers a
# candidate set rather than the whole catalog. All 27 tools cost ~4k tokens a row
# (66% of the prompt) and teach nothing about selection. Argument validation still
# runs against the full pinned v2 contract -- only the prompt is compacted.
AMBIENT_TOOLS = ("GetDateTime", "GetLiveContext")
DEFAULT_TOOLS_PER_ROW = 8


def available_tools_for_home(home: dict[str, Any]) -> list[dict[str, Any]]:
    """Return full pinned v2 tool catalog (runtime sends all schema tools regardless of home)."""
    return v2_openai_tools()


def script_tool_name(entity: dict[str, Any]) -> str:
    """Tool name Home Assistant gives a script: its object id, digit-prefixed with _.

    ScriptTool overrides ActionTool's ``domain__action`` name with the bare object id
    (homeassistant/components/script/llm.py, HA 2026.8.3).
    """
    object_id = entity["entity_id"].split(".", 1)[1]
    return f"_{object_id}" if object_id[:1].isdigit() else object_id


def script_tools(home: dict[str, Any]) -> list[dict[str, Any]]:
    """One tool per exposed script, as Home Assistant exposes them.

    Scripts are not controlled through HassTurnOn and never appear in the entity
    overview: each is its own zero-argument tool.
    """
    tools = []
    for entity in sorted(home.get("entities", []), key=lambda item: item["name"]):
        if entity["domain"] != "script":
            continue
        aliases = sorted(set(entity.get("aliases") or []))
        description = entity["name"]
        if aliases:
            description = f"{description}. Aliases: {aliases}"
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": script_tool_name(entity),
                    "description": description,
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        )
    return tools


def offered_tools(
    called_names: list[str],
    seed_key: str,
    *,
    count: int = DEFAULT_TOOLS_PER_ROW,
    extra_tools: list[dict[str, Any]] | None = None,
    excluded_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Candidate tools for one row: every called tool, the ambient pair, then distractors.

    ``extra_tools`` carries the home's per-script tools, which are not in the pinned
    catalog. Returned alphabetically so position never leaks which tool is the
    answer, and seeded by crc32 (not builtin hash) so the same row picks the same
    distractors in every process.
    """
    catalog = {tool["function"]["name"]: tool for tool in v2_openai_tools()}
    for tool in extra_tools or []:
        catalog[tool["function"]["name"]] = tool
    for name in excluded_names or []:
        catalog.pop(name, None)
    keep = {name for name in called_names if name in catalog}
    keep.update(name for name in AMBIENT_TOOLS if name in catalog)
    pool = sorted(set(catalog) - keep)
    rng = random.Random(zlib.crc32(str(seed_key).encode()))
    rng.shuffle(pool)
    keep.update(pool[: max(0, count - len(keep))])
    return [catalog[name] for name in sorted(keep)]


def _domain_args(entity: dict[str, Any]) -> dict[str, Any]:
    domain = entity["domain"]
    if domain in {"light", "fan", "switch", "media_player", "climate", "vacuum", "scene", "script"}:
        return {"domain": [domain]}
    if entity.get("device_class"):
        return {"device_class": [entity["device_class"]]}
    return {}


def _media_player_args(entity: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {"name": entity["name"], "domain": ["media_player"]}
    if entity.get("device_class"):
        args["device_class"] = [entity["device_class"]]
    return args


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


def build_climate_set_temperature(entity: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    return {
        "name": "HassClimateSetTemperature",
        "arguments": {
            "name": entity["name"],
            "temperature": rng.randint(65, 75),
        },
    }


def build_media_pause(entity: dict[str, Any]) -> dict[str, Any]:
    return {"name": "HassMediaPause", "arguments": _media_player_args(entity)}


def build_media_unpause(entity: dict[str, Any]) -> dict[str, Any]:
    return {"name": "HassMediaUnpause", "arguments": _media_player_args(entity)}


def build_set_volume(entity: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    args = _media_player_args(entity)
    args["volume_level"] = rng.randrange(10, 81)
    return {"name": "HassSetVolume", "arguments": args}


def build_volume_relative(entity: dict[str, Any], *, direction: str = "up") -> dict[str, Any]:
    return {
        "name": "HassSetVolumeRelative",
        "arguments": {"name": entity["name"], "volume_step": direction},
    }


def build_media_mute(entity: dict[str, Any]) -> dict[str, Any]:
    return {"name": "HassMediaPlayerMute", "arguments": _media_player_args(entity)}


def build_vacuum_start(entity: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "HassVacuumStart",
        "arguments": {"name": entity["name"], "domain": ["vacuum"]},
    }


def build_vacuum_return_home(entity: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "HassVacuumReturnToBase",
        "arguments": {"name": entity["name"], "domain": ["vacuum"]},
    }


def build_vacuum_clean_area(entity: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "HassVacuumCleanArea",
        "arguments": {"name": entity["name"], "area": entity["area"]},
    }


def build_query(entity: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {}
    if entity.get("name"):
        args["name"] = entity["name"]
    domain = entity.get("domain")
    if domain in {"light", "fan", "switch", "climate", "media_player", "vacuum", "scene", "script"}:
        args["domain"] = domain if isinstance(domain, str) else domain
    return {"name": "GetLiveContext", "arguments": args}


def build_cancel_all_timers(area: str | None = None) -> dict[str, Any]:
    args: dict[str, Any] = {}
    if area:
        args["area"] = area
    return {"name": "HassCancelAllTimers", "arguments": args}


def build_start_timer(rng: random.Random, *, name: str | None = None) -> dict[str, Any]:
    args: dict[str, Any] = {"minutes": rng.randint(5, 30)}
    if name:
        args["name"] = name
    return {"name": "HassStartTimer", "arguments": args}


def build_pause_timer(*, name: str | None = None) -> dict[str, Any]:
    args: dict[str, Any] = {}
    if name:
        args["name"] = name
    return {"name": "HassPauseTimer", "arguments": args}


def build_timer_status(*, name: str | None = None) -> dict[str, Any]:
    args: dict[str, Any] = {}
    if name:
        args["name"] = name
    return {"name": "HassTimerStatus", "arguments": args}


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
    if domain in {"light", "fan", "switch", "media_player", "climate", "vacuum", "scene", "script"}:
        args["domain"] = [domain]
    elif cap.device_class:
        args["device_class"] = [cap.device_class]
    tool = _operation_tool(operation, capability)
    if tool == "HassLightSet" and rng:
        settings = build_light_set({"name": "", "domain": "light"}, rng, operation)["arguments"]
        args.update({key: value for key, value in settings.items() if key not in {"name", "domain"}})
    if tool == "HassFanSetSpeed" and rng:
        args["percentage"] = rng.randrange(10, 101)
    if tool == "HassClimateSetTemperature" and rng:
        args["temperature"] = rng.randint(65, 75)
    if tool == "HassSetVolume" and rng:
        args["volume_level"] = rng.randrange(20, 80)
    if tool == "HassSetVolumeRelative":
        args["volume_step"] = "up"
    return {"name": tool, "arguments": args}


def _operation_tool(operation: str, capability: str) -> str:
    mapping = {
        "turn_on": "HassTurnOn",
        "open": "HassTurnOn",
        "lock": "HassTurnOn",
        "activate": "HassTurnOn",
        "run": "HassTurnOn",
        "turn_off": "HassTurnOff",
        "close": "HassTurnOff",
        "unlock": "HassTurnOff",
        "set_brightness": "HassLightSet",
        "set_color": "HassLightSet",
        "set_color_temperature": "HassLightSet",
        "set_speed": "HassFanSetSpeed",
        "set_temperature": "HassClimateSetTemperature",
        "play": "HassMediaUnpause",
        "pause": "HassMediaPause",
        "volume_set": "HassSetVolume",
        "volume_up": "HassSetVolumeRelative",
        "mute": "HassMediaPlayerMute",
        "start": "HassVacuumStart" if capability == "vacuums" else "HassStartTimer",
        "return_home": "HassVacuumReturnToBase",
        "clean_area": "HassVacuumCleanArea",
        "cancel_all": "HassCancelAllTimers",
        "status": "HassTimerStatus",
        "query_state": "GetLiveContext",
    }
    if operation == "pause" and capability == "timers":
        return "HassPauseTimer"
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
    if capability == "timers":
        if operation == "cancel_all":
            return build_cancel_all_timers(area)
        if operation == "start":
            return build_start_timer(rng)
        if operation == "pause":
            return build_pause_timer()
        if operation == "status":
            return build_timer_status()
    if operation == "query_state":
        if entity is None:
            cap = CAPABILITIES[capability]
            return {"name": "GetLiveContext", "arguments": {"domain": cap.domain}}
        return build_query(entity)
    if entity is None and area:
        return build_area_call(capability, operation, area, floor=floor, rng=rng)
    if entity is None:
        raise ValueError("entity or area required for operation")
    if operation == "run" and capability == "scripts":
        return {"name": script_tool_name(entity), "arguments": {}}
    if operation in {"turn_on", "open", "lock", "activate", "run"}:
        return build_turn_on(entity)
    if operation in {"turn_off", "close", "unlock"}:
        return build_turn_off(entity)
    if operation.startswith("set_") and capability == "lights":
        return build_light_set(entity, rng, operation)
    if operation == "set_speed":
        return build_fan_speed(entity, rng)
    if operation == "set_temperature":
        return build_climate_set_temperature(entity, rng)
    if operation == "play":
        return build_media_unpause(entity)
    if operation == "pause":
        return build_media_pause(entity)
    if operation == "volume_set":
        return build_set_volume(entity, rng)
    if operation == "volume_up":
        return build_volume_relative(entity, direction="up")
    if operation == "mute":
        return build_media_mute(entity)
    if operation == "start" and capability == "vacuums":
        return build_vacuum_start(entity)
    if operation == "return_home":
        return build_vacuum_return_home(entity)
    if operation == "clean_area":
        return build_vacuum_clean_area(entity)
    raise ValueError(f"unsupported operation {operation!r} for {capability!r}")


def validate_call_tool_name(name: str) -> bool:
    return name in ALLOWED_HASS_TOOLS
