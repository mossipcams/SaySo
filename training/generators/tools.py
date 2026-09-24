"""Runtime-equivalent tool call builders and availability."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from adapters.schema import v2_openai_tools
from generators.context import _REPO_ROOT  # noqa: F401 - puts sayso_contract on sys.path
from sayso_contract import tool_schema
from generators.capability_registry import CAPABILITIES

# Home Assistant 2026.9 names every Assist LLM tool ``f"{DOMAIN}__{intent_type}"``,
# where DOMAIN is the component that registers the intent handler, and wraps
# multi-API tools in ``llm.NamespacedTool``. The pinned v2 contract predates that
# and carries the bare names, so a corpus trained only on bare names meets an
# out-of-distribution tool list at runtime. Measured on the deployed stack for
# issue #52: with the production-namespaced list the model picks the wrong tool on
# 2 of 3 TV utterances; with bare names and the same schema it picks the right one.
#
# Verified against homeassistant==2026.9.2: each component's ``llm.py`` exposes
# ``LLM_INTENTS``/``TIMER_INTENTS`` and builds the name with its own DOMAIN.
HA_TOOL_NAMESPACES: dict[str, str] = {
    "GetDateTime": "llm",
    "GetLiveContext": "homeassistant",
    "HassCancelAllTimers": "intent",
    "HassCancelTimer": "intent",
    "HassClimateSetTemperature": "climate",
    "HassDecreaseTimer": "intent",
    "HassFanSetSpeed": "fan",
    "HassIncreaseTimer": "intent",
    "HassLightSet": "light",
    "HassMediaNext": "media_player",
    "HassMediaPause": "media_player",
    "HassMediaPlayerMute": "media_player",
    "HassMediaPlayerUnmute": "media_player",
    "HassMediaPrevious": "media_player",
    "HassMediaSearchAndPlay": "media_player",
    "HassMediaUnpause": "media_player",
    "HassPauseTimer": "intent",
    "HassSetVolume": "media_player",
    "HassSetVolumeRelative": "media_player",
    "HassStartTimer": "intent",
    "HassTimerStatus": "intent",
    "HassTurnOff": "intent",
    "HassTurnOn": "intent",
    "HassUnpauseTimer": "intent",
    "HassVacuumCleanArea": "vacuum",
    "HassVacuumReturnToBase": "vacuum",
    "HassVacuumStart": "vacuum",
}


def namespaced_tool_name(name: str) -> str:
    """Return the Home Assistant 2026.9 name for a pinned tool.

    Per-script tools keep their bare object id: ``ScriptTool`` overrides the
    ``domain__action`` name, so they are already what Home Assistant sends.
    """
    namespace = HA_TOOL_NAMESPACES.get(name)
    return f"{namespace}__{name}" if namespace else name


# Namespaces whose tools Home Assistant offers regardless of which domains are
# exposed: date/time, live context, and the generic intents and timers.
_ALWAYS_OFFERED_NAMESPACES = frozenset({"llm", "homeassistant", "intent"})


def compile_catalog(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Name tools as Home Assistant 2026.9 does and compile them with the
    integration's own compiler, so every row's schema bytes match production."""
    return list(
        tool_schema.compile_source_tools(
            [
                {
                    "name": namespaced_tool_name(tool["function"]["name"]),
                    "description": tool["function"].get("description") or "",
                    "parameters": tool["function"]["parameters"],
                }
                for tool in tools
            ]
        ).tools
    )


def production_catalog(
    home: dict[str, Any], *, removed_tools: tuple[str, ...] | list[str] = ()
) -> list[dict[str, Any]]:
    """The tool list Home Assistant would offer a satellite in ``home``.

    Domain tools (``light__HassLightSet``) appear only when an entity of that
    domain is exposed, exactly as each component's ``llm.py`` decides; the
    generic intents, timers, live context and date/time are always offered;
    every exposed script is its own tool. ``removed_tools`` takes production
    names and withholds only those, for a deliberate missing-capability case.
    """
    domains = {entity["domain"] for entity in home.get("entities", [])}
    offered = [
        tool
        for tool in v2_openai_tools()
        if HA_TOOL_NAMESPACES[tool["function"]["name"]] in _ALWAYS_OFFERED_NAMESPACES | domains
    ]
    catalog = compile_catalog([*offered, *script_tools(home)])
    unknown = set(removed_tools) - {tool["function"]["name"] for tool in catalog}
    if unknown:
        raise ValueError(f"cannot remove tools the catalog does not offer: {sorted(unknown)}")
    return [tool for tool in catalog if tool["function"]["name"] not in set(removed_tools)]


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


def build_media_unmute(entity: dict[str, Any]) -> dict[str, Any]:
    return {"name": "HassMediaPlayerUnmute", "arguments": _media_player_args(entity)}


def build_media_next(entity: dict[str, Any]) -> dict[str, Any]:
    return {"name": "HassMediaNext", "arguments": _media_player_args(entity)}


def build_media_previous(entity: dict[str, Any]) -> dict[str, Any]:
    return {"name": "HassMediaPrevious", "arguments": _media_player_args(entity)}


# Titles a household actually asks for, paired with the media_class Home Assistant
# would search. Names, not adjectives: the point is a realistic search_query slot.
MEDIA_SEARCHES: tuple[tuple[str, str], ...] = (
    ("jazz", "music"),
    ("the news", "channel"),
    ("Fleetwood Mac", "artist"),
    ("the morning playlist", "playlist"),
    ("nature documentaries", "tv_show"),
    ("Bluey", "tv_show"),
    ("relaxing piano", "music"),
    ("the football game", "channel"),
)


def build_media_search_and_play(entity: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    query, media_class = rng.choice(MEDIA_SEARCHES)
    # No domain/device_class in this tool's contract: the pinned schema takes
    # search_query, media_class and a target only.
    return {
        "name": "HassMediaSearchAndPlay",
        "arguments": {"name": entity["name"], "search_query": query, "media_class": media_class},
    }


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


def build_get_datetime() -> dict[str, Any]:
    """GetDateTime has no entity arguments; Home Assistant answers from clock."""
    return {"name": "GetDateTime", "arguments": {}}


def build_query(entity: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {}
    if entity.get("name"):
        args["name"] = entity["name"]
    # GetLiveContext takes any domain string; naming it keeps a cover or lock query
    # as identifiable as a light query.
    if entity.get("domain"):
        args["domain"] = entity["domain"]
    return {"name": "GetLiveContext", "arguments": args}


def build_cancel_all_timers(area: str | None = None) -> dict[str, Any]:
    args: dict[str, Any] = {}
    if area:
        args["area"] = area
    return {"name": "HassCancelAllTimers", "arguments": args}


def build_start_timer(rng: random.Random, *, name: str | None = None) -> dict[str, Any]:
    unit, low, high = rng.choices((("minutes", 1, 90), ("seconds", 10, 90), ("hours", 1, 3)), weights=(6, 2, 2))[0]
    args: dict[str, Any] = {unit: rng.randint(low, high)}
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


def build_unpause_timer(*, name: str | None = None) -> dict[str, Any]:
    args: dict[str, Any] = {}
    if name:
        args["name"] = name
    return {"name": "HassUnpauseTimer", "arguments": args}


def build_cancel_timer(*, name: str | None = None) -> dict[str, Any]:
    args: dict[str, Any] = {}
    if name:
        args["name"] = name
    return {"name": "HassCancelTimer", "arguments": args}


def build_adjust_timer(rng: random.Random, *, direction: str, name: str | None = None) -> dict[str, Any]:
    """HassIncreaseTimer / HassDecreaseTimer: a duration delta on a running timer."""
    args: dict[str, Any] = (
        {"seconds": rng.choice((10, 15, 30, 45))} if rng.random() < 0.25 else {"minutes": rng.choice((1, 2, 5, 10, 15))}
    )
    if name:
        args["name"] = name
    tool = "HassIncreaseTimer" if direction == "increase" else "HassDecreaseTimer"
    return {"name": tool, "arguments": args}


def build_area_call(
    capability: str,
    operation: str,
    area: str | None,
    *,
    floor: str | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    cap = CAPABILITIES[capability]
    domain = cap.domain
    args: dict[str, Any] = {"area": area} if area else {}
    if floor:
        args["floor"] = floor
    if domain in {"light", "fan", "switch", "media_player", "climate", "vacuum", "scene", "script"}:
        args["domain"] = [domain]
    elif cap.device_class:
        args["device_class"] = [cap.device_class]
    tool = _operation_tool(operation, capability)
    # Tools whose pinned contract carries no domain/device_class filter.
    if tool in {"HassMediaSearchAndPlay", "HassSetVolumeRelative", "HassClimateSetTemperature"}:
        args.pop("domain", None)
        args.pop("device_class", None)
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
        args["volume_step"] = "down" if operation == "volume_down" else "up"
    if tool == "HassMediaSearchAndPlay" and rng:
        query, media_class = rng.choice(MEDIA_SEARCHES)
        args["search_query"] = query
        args["media_class"] = media_class
    return {"name": tool, "arguments": args}


# The pinned tool each operation maps to, for area and floor calls where there
# is no entity to inspect. Two operations are overloaded across capabilities and
# are resolved before this table is consulted.
_OPERATION_TOOLS: dict[str, str] = {
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
    "next_track": "HassMediaNext",
    "previous_track": "HassMediaPrevious",
    "volume_set": "HassSetVolume",
    "volume_up": "HassSetVolumeRelative",
    "volume_down": "HassSetVolumeRelative",
    "mute": "HassMediaPlayerMute",
    "unmute": "HassMediaPlayerUnmute",
    "search_and_play": "HassMediaSearchAndPlay",
    "return_home": "HassVacuumReturnToBase",
    "clean_area": "HassVacuumCleanArea",
    "cancel_all": "HassCancelAllTimers",
    "cancel": "HassCancelTimer",
    "unpause": "HassUnpauseTimer",
    "increase": "HassIncreaseTimer",
    "decrease": "HassDecreaseTimer",
    "status": "HassTimerStatus",
    "query_state": "GetLiveContext",
}


def _operation_tool(operation: str, capability: str) -> str:
    """Resolve one operation to its pinned tool name."""
    if operation == "start":
        return "HassVacuumStart" if capability == "vacuums" else "HassStartTimer"
    if operation == "pause" and capability == "timers":
        return "HassPauseTimer"
    if operation not in _OPERATION_TOOLS:
        raise KeyError(f"no tool mapped for {capability!r} operation {operation!r}")
    return _OPERATION_TOOLS[operation]


# What an operation means, once the entity is known. Replaces a forty-branch
# if-chain: the routing is the table, and the exceptions below it are the three
# operations whose meaning genuinely depends on the capability.
_ENTITY_CALLS: dict[str, Callable[[dict[str, Any], random.Random], dict[str, Any]]] = {
    "turn_on": lambda entity, _rng: build_turn_on(entity),
    "open": lambda entity, _rng: build_turn_on(entity),
    "lock": lambda entity, _rng: build_turn_on(entity),
    "activate": lambda entity, _rng: build_turn_on(entity),
    "run": lambda entity, _rng: build_turn_on(entity),
    "turn_off": lambda entity, _rng: build_turn_off(entity),
    "close": lambda entity, _rng: build_turn_off(entity),
    "unlock": lambda entity, _rng: build_turn_off(entity),
    "set_speed": build_fan_speed,
    "set_temperature": build_climate_set_temperature,
    "play": lambda entity, _rng: build_media_unpause(entity),
    "pause": lambda entity, _rng: build_media_pause(entity),
    "volume_set": build_set_volume,
    "volume_up": lambda entity, _rng: build_volume_relative(entity, direction="up"),
    "volume_down": lambda entity, _rng: build_volume_relative(entity, direction="down"),
    "mute": lambda entity, _rng: build_media_mute(entity),
    "unmute": lambda entity, _rng: build_media_unmute(entity),
    "next_track": lambda entity, _rng: build_media_next(entity),
    "previous_track": lambda entity, _rng: build_media_previous(entity),
    "search_and_play": build_media_search_and_play,
    "return_home": lambda entity, _rng: build_vacuum_return_home(entity),
    "clean_area": lambda entity, _rng: build_vacuum_clean_area(entity),
}

# Timers have no entity, and "start" and "pause" mean different tools here than
# they do for a vacuum or a media player.
_TIMER_CALLS: dict[str, Callable[[random.Random, str | None], dict[str, Any]]] = {
    "cancel_all": lambda _rng, area: build_cancel_all_timers(area),
    "start": lambda rng, _area: build_start_timer(rng),
    "pause": lambda _rng, _area: build_pause_timer(),
    "unpause": lambda _rng, _area: build_unpause_timer(),
    "cancel": lambda _rng, _area: build_cancel_timer(),
    "status": lambda _rng, _area: build_timer_status(),
    "increase": lambda rng, _area: build_adjust_timer(rng, direction="increase"),
    "decrease": lambda rng, _area: build_adjust_timer(rng, direction="decrease"),
}


def build_call_for_operation(
    entity: dict[str, Any] | None,
    capability: str,
    operation: str,
    rng: random.Random,
    *,
    area: str | None = None,
    floor: str | None = None,
) -> dict[str, Any]:
    """The one tool call an operation on a capability means."""
    if capability == "timers" and (timer_call := _TIMER_CALLS.get(operation)):
        return timer_call(rng, area)

    if operation == "query_state":
        if entity is not None:
            return build_query(entity)
        return {
            "name": "GetLiveContext",
            "arguments": {"domain": CAPABILITIES[capability].domain},
        }

    if entity is None:
        if area or floor:
            return build_area_call(capability, operation, area, floor=floor, rng=rng)
        raise ValueError("entity or area required for operation")

    # A script is its own tool, so it never routes through HassTurnOn.
    if operation == "run" and capability == "scripts":
        return {"name": script_tool_name(entity), "arguments": {}}
    if operation.startswith("set_") and capability == "lights":
        return build_light_set(entity, rng, operation)
    if operation == "start" and capability == "vacuums":
        return build_vacuum_start(entity)

    build = _ENTITY_CALLS.get(operation)
    if build is None:
        raise ValueError(f"unsupported operation {operation!r} for {capability!r}")
    return build(entity, rng)


