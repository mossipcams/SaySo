"""One behavioral execution pipeline.

For each case: build the production area context from the household and
utterance (never from the expected answer), render the production system
prompt, compile the household's production tool catalog (minus only the
capability an unavailable case withholds), ask the model through the
configured adapter, parse with the production parser, validate with the
production contract check, and score with the shared scorer.

Evaluation never executes Home Assistant actions.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import urllib.error
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import yaml

from evals.cases import (
    Case,
    fingerprint_cases,
    load_gates,
    load_home,
    production_contract_fingerprint,
    select_cases,
)
from evals.outcomes import (
    PRODUCTION_MAX_OUTPUT_TOKENS,
    PRODUCTION_TEMPERATURE,
    CaseResult,
    Expectation,
    ResponseType,
    read_completion,
)
from evals.scorer import check_gates, score, summarize
from sayso_contract import area_context as _area_context
from sayso_contract import tool_schema

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
V2_SCHEMA_PATH = ROOT / "schemas" / "sayso-tool-schema-v2.json"

AreaContext = _area_context.AreaContext
build_area_context = _area_context.build_area_context
render_system_prompt = _area_context.render_system_prompt

# Mirrors training/generators/context.py — kept here so HA compat tests never
# import the training package tree (which pulls jsonschema).
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

# Mirrors training/generators/tools.py HA_TOOL_NAMESPACES.
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

_ALWAYS_OFFERED_NAMESPACES = frozenset({"llm", "homeassistant", "intent"})


class ModelAdapter(Protocol):
    name: str

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any: ...


@contextmanager
def training_path():
    """Expose ``training/generators`` without shadowing the repo ``tests`` package."""
    path = str(ROOT / "training")
    added = path not in sys.path
    if added:
        sys.path.append(path)
    try:
        yield
    finally:
        if added:
            sys.path.remove(path)


@lru_cache(maxsize=1)
def _v2_openai_tools() -> tuple[dict[str, Any], ...]:
    """Pinned v2 catalog from the checked-in schema artifact (no jsonschema)."""
    payload = json.loads(V2_SCHEMA_PATH.read_text(encoding="utf-8"))
    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError("Pinned v2 schema must contain a non-empty tools list")
    return tuple(dict(tool) for tool in tools)


def namespaced_tool_name(name: str) -> str:
    namespace = HA_TOOL_NAMESPACES.get(name)
    return f"{namespace}__{name}" if namespace else name


def _dump_yaml(data: list[dict[str, Any]]) -> str:
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


def home_areas(home: dict[str, Any]) -> dict[str, list[str]]:
    areas: dict[str, list[str]] = {
        area: [] for area in [*(home.get("areas") or []), *(e.get("area") for e in home.get("entities", []))] if area
    }
    for area, aliases in (home.get("area_aliases") or {}).items():
        areas[area] = list(aliases)
    return areas


def area_context_for(home: dict[str, Any], utterance: str) -> AreaContext:
    return build_area_context(utterance, home_areas(home), home.get("sayso_entity_area"))


def _namespaced_prompt(text: str) -> str:
    for bare in ("GetLiveContext", "HassTurnOn", "HassTurnOff"):
        text = text.replace(bare, namespaced_tool_name(bare))
    return text


def serialize_context(home: dict[str, Any], utterance: str = "") -> str:
    entities = exposed_entities(home)
    if entities:
        api_prompt = "\n".join(
            [_namespaced_prompt(DYNAMIC_CONTEXT_PROMPT), STATIC_CONTEXT_HEADER, _dump_yaml(entities)]
        )
    else:
        api_prompt = NO_ENTITIES_PROMPT
    api_prompt = "\n".join([api_prompt, _namespaced_prompt(DEVICE_CONTROL_TOOL_USAGE_PROMPT)])
    return render_system_prompt(
        "\n".join([SAYSO_SYSTEM_PROMPT, api_prompt]), area_context_for(home, utterance)
    )


def _compile_catalog(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _script_tool_name(entity: dict[str, Any]) -> str:
    object_id = entity["entity_id"].split(".", 1)[1]
    return f"_{object_id}" if object_id[:1].isdigit() else object_id


def _script_tools(home: dict[str, Any]) -> list[dict[str, Any]]:
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
                    "name": _script_tool_name(entity),
                    "description": description,
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        )
    return tools


def production_catalog(home: dict[str, Any], *, removed_tools: list[str] | None = None) -> list[dict[str, Any]]:
    removed = removed_tools or []
    domains = {entity["domain"] for entity in home.get("entities", [])}
    offered = [
        tool
        for tool in _v2_openai_tools()
        if HA_TOOL_NAMESPACES[tool["function"]["name"]] in _ALWAYS_OFFERED_NAMESPACES | domains
    ]
    catalog = _compile_catalog([*offered, *_script_tools(home)])
    unknown = set(removed) - {tool["function"]["name"] for tool in catalog}
    if unknown:
        raise ValueError(f"cannot remove tools the catalog does not offer: {sorted(unknown)}")
    return [tool for tool in catalog if tool["function"]["name"] not in set(removed)]


def render_case(case: Case) -> dict[str, Any]:
    """Everything the model and the scorer see for one case."""
    home = load_home(case.household)
    removed = (case.unavailable or {}).get("removed_tools") or []
    expected = case.expected
    area = area_context_for(home, case.utterance)
    return {
        "messages": [
            {"role": "system", "content": serialize_context(home, case.utterance)},
            {"role": "user", "content": case.utterance},
        ],
        "tools": production_catalog(home, removed_tools=removed),
        "exposed_domains": frozenset(entity["domain"] for entity in home["entities"]),
        "area_context": area.as_dict(),
        "expectation": Expectation(
            category=case.category,
            response_type=ResponseType(expected["response_type"]),
            calls=tuple(expected["calls"]),
            forbidden_entities=tuple(expected["forbidden_entities"]),
            target_area=area.target_area,
        ),
    }


def evaluate(cases: list[Case], adapter: ModelAdapter) -> list[CaseResult]:
    """Score every case through the shared parser, validator, and scorer."""
    results: list[CaseResult] = []
    for case in cases:
        rendered = render_case(case)
        try:
            raw = adapter.complete(rendered["messages"], rendered["tools"])
            transport_error = None
        except (urllib.error.URLError, TimeoutError, OSError, ConnectionError) as err:
            raw = {"transport_error": str(err)}
            transport_error = str(err)
        turn = read_completion(raw, rendered["tools"], rendered["exposed_domains"])
        result = score(case.id, rendered["expectation"], turn)
        if transport_error:
            result.flags.append("transport_error")
        results.append(result)
    return results


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_run(
    *,
    cases: list[Case],
    results: list[CaseResult],
    adapter: ModelAdapter,
    suite: str | None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    dest = RESULTS_DIR / run_id
    dest.mkdir(parents=True, exist_ok=True)
    summary = summarize(results)
    gates = load_gates(suite) if suite else None
    failures = check_gates(summary, gates, expected_count=len(cases)) if gates else []
    metadata = {
        "run_id": run_id,
        "suite": suite,
        "adapter": adapter.name,
        "checkpoint": getattr(adapter, "model", None) or (extra_metadata or {}).get("checkpoint"),
        "code_revision": _git_revision(),
        "case_hash": fingerprint_cases(cases),
        "suite_hash": hashlib.sha256("\n".join(case.id for case in cases).encode()).hexdigest(),
        "production_contract_fingerprint": production_contract_fingerprint(),
        "decoding": {
            "temperature": PRODUCTION_TEMPERATURE,
            "max_output_tokens": PRODUCTION_MAX_OUTPUT_TOKENS,
        },
        "created_at": datetime.now(UTC).isoformat(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    if hasattr(adapter, "server"):
        metadata["server"] = adapter.server
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {
        "summary": summary,
        "promotion": {"passed": not failures, "failures": failures} if gates else None,
    }
    (dest / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    outcomes = dest / "outcomes.jsonl"
    with outcomes.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.as_dict(), ensure_ascii=False) + "\n")
    raw_dir = dest / "raw"
    raw_dir.mkdir()
    for result in results:
        (raw_dir / f"{result.case_id}.json").write_text(
            json.dumps(result.turn.raw, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return dest
