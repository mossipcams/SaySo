
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)
from sayso_contract import area_context as _area_context

AreaContext = _area_context.AreaContext
build_area_context = _area_context.build_area_context
render_system_prompt = _area_context.render_system_prompt

SAYSO_SYSTEM_PROMPT = """You are SaySo, a local Home Assistant voice agent.
Use the available tools for home state queries and actions. Only claim an action succeeded when its tool result confirms success. Use names, areas, and context supplied by Home Assistant. If a request is ambiguous, ask one short question. Keep spoken responses brief. Do not describe tool calls."""

DYNAMIC_CONTEXT_PROMPT = (
    "You ARE equipped to answer questions about the"
    " current state of\n"
    "the home using the `GetLiveContext` tool."
    " This is a primary function."
    " Do not state you lack the\n"
    "functionality if the question requires live data.\n"
    "If the user asks about device existence/type"
    ' (e.g., "Do I have lights in the bedroom?"):'
    " Answer\n"
    "from the static context below.\n"
    "If the user asks about the CURRENT state, value,"
    ' or mode (e.g., "Is the lock locked?",\n'
    '"Is the fan on?",'
    ' "What mode is the thermostat in?",'
    ' "What is the temperature outside?"):\n'
    "    1.  Recognize this requires live data.\n"
    "    2.  You MUST call `GetLiveContext`."
    " This tool will provide the needed real-time"
    " information (like temperature from the local"
    " weather, lock status, etc.).\n"
    "    3.  Use the tool's response** to answer the"
    " user accurately"
    ' (e.g., "The temperature outside is'
    ' [value from tool].").\n'
    "For general knowledge questions not about the"
    " home: Answer truthfully from internal"
    " knowledge.\n"
)

STATIC_CONTEXT_HEADER = "Static Context: An overview of the areas and the devices in this smart home:"

NO_ENTITIES_PROMPT = (
    "Only if the user wants to control a device, tell them to expose entities "
    "to their voice assistant in Home Assistant."
)

DEVICE_CONTROL_TOOL_USAGE_PROMPT = (
    "When controlling Home Assistant always call the intent tools. "
    "Use HassTurnOn to lock and HassTurnOff to unlock a lock. "
    "When controlling a device, prefer passing just name and domain. "
    "When controlling an area, prefer passing just area name and domain."
)

OVERVIEW_EXCLUDED_DOMAINS = frozenset({"calendar", "script"})


def _dump(data: list[dict[str, Any]]) -> str:
    return yaml.dump(
        data,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    ).replace(": null\n", ":\n")


def exposed_entities(home: dict[str, Any]) -> list[dict[str, Any]]:
    entities = []
    for entity in sorted(home.get("entities", []), key=lambda item: item["name"]):
        if entity["domain"] in OVERVIEW_EXCLUDED_DOMAINS:
            continue
        names = dict.fromkeys([entity["name"], *entity.get("aliases", [])])
        info: dict[str, Any] = {"names": ", ".join(names), "domain": entity["domain"]}
        if entity.get("area"):
            info["areas"] = entity["area"]
        entities.append(info)
    return entities


def _namespaced(text: str) -> str:
    from generators.tools import namespaced_tool_name

    for bare in ("GetLiveContext", "HassTurnOn", "HassTurnOff"):
        text = text.replace(bare, namespaced_tool_name(bare))
    return text


def home_areas(home: dict[str, Any]) -> dict[str, list[str]]:
    areas: dict[str, list[str]] = {
        area: [] for area in [*(home.get("areas") or []), *(e.get("area") for e in home.get("entities", []))] if area
    }
    for area, aliases in (home.get("area_aliases") or {}).items():
        areas[area] = list(aliases)
    return areas


def area_context_for(home: dict[str, Any], utterance: str) -> AreaContext:
    return build_area_context(utterance, home_areas(home), home.get("sayso_entity_area"))


def serialize_context(home: dict[str, Any], utterance: str = "") -> str:
    entities = exposed_entities(home)
    if entities:
        api_prompt = "\n".join(
            [_namespaced(DYNAMIC_CONTEXT_PROMPT), STATIC_CONTEXT_HEADER, _dump(entities)]
        )
    else:
        api_prompt = NO_ENTITIES_PROMPT
    api_prompt = "\n".join([api_prompt, _namespaced(DEVICE_CONTROL_TOOL_USAGE_PROMPT)])
    return render_system_prompt(
        "\n".join([SAYSO_SYSTEM_PROMPT, api_prompt]), area_context_for(home, utterance)
    )


def system_prompt(home: dict[str, Any], utterance: str = "") -> str:
    return serialize_context(home, utterance)
