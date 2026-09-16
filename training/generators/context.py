"""Context serialization mirroring the Home Assistant Assist API system prompt.

Copied from Home Assistant 2026.8.3, the version pinned in `hacs.json`:

- `homeassistant/components/homeassistant/llm.py` — GetLiveContext guidance and the
  static entity overview
- `homeassistant/components/intent/llm.py` — device-control and area guidance
- `homeassistant/components/conversation/chat_log.py` — the order the parts are joined

SaySo never builds this prompt itself. `conversation.py` forwards whatever Home
Assistant put in the chat log, so training rows must match Home Assistant's shape
or the model learns to read a context the device never sends.

Two properties matter most and both differ from the pre-2026.8 format:

1. The static overview carries **no state**. Home Assistant builds it with
   `include_state=False`, so live values only ever arrive via `GetLiveContext`.
2. It is YAML with three keys per entity (`names`, `domain`, `areas`), not JSON
   with eight.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import yaml

# Load the integration's dependency-free contract without importing Home
# Assistant's integration package into the training environment.
_AREA_CONTEXT_PATH = Path(__file__).resolve().parents[2] / "custom_components" / "sayso" / "area_context.py"
_AREA_CONTEXT_SPEC = importlib.util.spec_from_file_location("sayso_area_context", _AREA_CONTEXT_PATH)
if _AREA_CONTEXT_SPEC is None or _AREA_CONTEXT_SPEC.loader is None:
    raise ImportError(f"cannot load {_AREA_CONTEXT_PATH}")
_AREA_CONTEXT = importlib.util.module_from_spec(_AREA_CONTEXT_SPEC)
sys.modules[_AREA_CONTEXT_SPEC.name] = _AREA_CONTEXT
_AREA_CONTEXT_SPEC.loader.exec_module(_AREA_CONTEXT)
AreaContext = _AREA_CONTEXT.AreaContext
render_area_context = _AREA_CONTEXT.render_area_context
resolve_area_context = _AREA_CONTEXT.resolve_area_context

# custom_components/sayso/const.py DEFAULT_SYSTEM_PROMPT, sent as user_llm_prompt.
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

# Home Assistant buckets calendars and scripts out of the overview because each has
# its own tools (async_get_exposed_entities). Scripts reach the model as per-script
# tools instead; see generators.tools.script_tools.
OVERVIEW_EXCLUDED_DOMAINS = frozenset({"calendar", "script"})


def _dump(data: list[dict[str, Any]]) -> str:
    """Match annotatedyaml.dumper.dump, which Home Assistant uses for the overview."""
    return yaml.dump(
        data,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    ).replace(": null\n", ":\n")


def exposed_entities(home: dict[str, Any]) -> list[dict[str, Any]]:
    """Entity overview rows in Home Assistant's field order, sorted by name."""
    entities = []
    for entity in sorted(home.get("entities", []), key=lambda item: item["name"]):
        if entity["domain"] in OVERVIEW_EXCLUDED_DOMAINS:
            continue
        # HA's intent.async_get_entity_aliases returns a deduped name set; generators
        # often default an entity's aliases to [name], which would repeat it here and
        # teach the model that names come in pairs.
        names = dict.fromkeys([entity["name"], *entity.get("aliases", [])])
        info: dict[str, Any] = {"names": ", ".join(names), "domain": entity["domain"]}
        if entity.get("area"):
            info["areas"] = entity["area"]
        entities.append(info)
    return entities


def _area_prompt(home: dict[str, Any], utterance: str | None = None) -> str:
    """Render the same request-area contract that production sends."""
    context = resolve_area_context(
        utterance or "",
        satellite_area=home.get("satellite_area", home.get("sayso_entity_area")),
        areas=home.get("areas", ()),
        entities=home.get("entities", ()),
    )
    rendered = render_area_context(context)
    if utterance is None and context.satellite_area:
        # Keep the public no-utterance helper readable for existing fixtures;
        # rendered rows use only the shared three-field contract.
        floor = next(
            (entity.get("floor") for entity in home.get("entities", [])
             if entity.get("area") == context.satellite_area and entity.get("floor")),
            None,
        )
        floor_text = f" (floor {floor})" if floor else ""
        rendered += f"\nYou are in area {context.satellite_area}{floor_text} and all generic commands like 'turn on the lights' should target this area."
    return rendered


def _namespaced(text: str) -> str:
    """Rewrite tool names quoted in the prompt into the 2026.9 contract.

    Home Assistant names the tool in the prompt as well as in the schema, so a row
    that offers ``intent__HassTurnOn`` must also say ``intent__HassTurnOn`` here;
    otherwise the prompt and the tool list disagree about what the tool is called.
    """
    from generators.tools import namespaced_tool_name

    for bare in ("GetLiveContext", "HassTurnOn", "HassTurnOff"):
        text = text.replace(bare, namespaced_tool_name(bare))
    return text


def serialize_context(
    home: dict[str, Any], *, utterance: str | None = None, namespaced: bool = False
) -> str:
    """Serialize exposed entity context as Home Assistant's Assist API sends it."""
    dynamic_prompt = DYNAMIC_CONTEXT_PROMPT
    control_prompt = DEVICE_CONTROL_TOOL_USAGE_PROMPT
    if namespaced:
        dynamic_prompt = _namespaced(dynamic_prompt)
        control_prompt = _namespaced(control_prompt)
    entities = exposed_entities(home)
    if entities:
        api_prompt = "\n".join(
            [dynamic_prompt, STATIC_CONTEXT_HEADER, _dump(entities)]
        )
    else:
        api_prompt = NO_ENTITIES_PROMPT
    # Domain order: the "homeassistant" platform sorts before "intent". The
    # request context precedes HA's appended API context, matching production.
    api_prompt = "\n".join([api_prompt, control_prompt])
    area_prompt = _area_prompt(home, utterance)
    # DATE_TIME_PROMPT is omitted: chat_log only appends it when no GetDateTime tool
    # is offered, and every row offers GetDateTime.
    return "\n".join([SAYSO_SYSTEM_PROMPT, area_prompt, api_prompt])


def system_prompt(
    home: dict[str, Any], *, utterance: str | None = None, namespaced: bool = False
) -> str:
    return serialize_context(home, utterance=utterance, namespaced=namespaced)
