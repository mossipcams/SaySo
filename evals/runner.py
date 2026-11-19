from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import time
import urllib.error
from dataclasses import dataclass, field
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
    PRODUCTION_HELD_TTL_SECONDS,
    PRODUCTION_MAX_OUTPUT_TOKENS,
    PRODUCTION_MAX_TOOL_ITERATIONS,
    PRODUCTION_TEMPERATURE,
    QUERY_TOOLS,
    TODO_GET_ITEMS_TOOL,
    CaseResult,
    Expectation,
    ResponseType,
    Turn,
    contract_violations,
    read_completion,
)
from evals.scorer import check_gates, score, summarize
from sayso_contract import area_context as _area_context
from sayso_contract import grammar as _grammar
from sayso_contract import completion, correction, exceptions, grounding, post_model
from sayso_contract import tool_contract, tool_schema

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
V2_SCHEMA_PATH = ROOT / "schemas" / "ha-2026.11.1-tools.json"

AreaContext = _area_context.AreaContext
build_area_context = _area_context.build_area_context
render_system_prompt = _area_context.render_system_prompt

GrammarUnsupported = _grammar.GrammarUnsupported
build_grammar = _grammar.build_grammar

SAYSO_SYSTEM_PROMPT = """You are SaySo, a local Home Assistant voice agent.
Use the available tools for home state queries and actions. Only claim an action succeeded when its tool result confirms success. Use names, areas, and context supplied by Home Assistant. If a request is ambiguous, ask one short question. Keep spoken responses brief. Do not describe tool calls."""

DYNAMIC_CONTEXT_PROMPT = """You ARE equipped to answer questions about the current state of
the home by retrieving live context. This is a primary function. Do not state you lack the
functionality if the question requires live data.
If the user asks about device existence/type (e.g., "Do I have lights in the bedroom?"): Answer
from the static context below.
If the user asks about the CURRENT state, value, or mode (e.g., "Is the lock locked?",
"Is the fan on?", "What mode is the thermostat in?", "What is the temperature outside?"):
    1.  Recognize this requires live data.
    2.  You MUST use the provided tool to retrieve live context. This tool will provide the needed real-time information (like temperature from the local weather, lock status, etc.).
    3.  Use the tool's response** to answer the user accurately (e.g., "The temperature outside is [value from tool].").
For general knowledge questions not about the home: Answer truthfully from internal knowledge.
"""

STATIC_CONTEXT_HEADER = (
    "Static Context: An overview of the areas and the devices in this smart home:"
)

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

HA_TOOL_NAMESPACES: dict[str, str] = {
    "GetLiveContext": "homeassistant",
    "HassTurnOff": "intent",
    "HassTurnOn": "intent",
}


class ModelAdapter(Protocol):
    name: str

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        grammar: str | None = None,
    ) -> Any: ...


_TRANSPORT_ERRORS = (urllib.error.URLError, TimeoutError, OSError, ConnectionError)
_FAKE_RESULT = {"success": []}


@dataclass
class _Trace:
    raws: list[Any] = field(default_factory=list)
    parse_error: str | None = None
    transport_error: str | None = None
    violations: list[dict[str, str]] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)


class _NullSpan:
    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}


class _NullRecorder:
    def open(self, stage: Any, **metadata: Any) -> _NullSpan:
        span = _NullSpan()
        span.metadata.update(metadata)
        return span

    def close(self, span: Any, *, success: bool = True) -> None:
        return None


@lru_cache(maxsize=1)
def _v2_openai_tools() -> tuple[dict[str, Any], ...]:
    payload = json.loads(V2_SCHEMA_PATH.read_text(encoding="utf-8"))
    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError("Pinned HA tool snapshot must contain a non-empty tools list")
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
    entities: list[tuple[str, dict[str, Any]]] = []
    for entity in sorted(home.get("entities", []), key=lambda item: item["name"]):
        if entity["domain"] in OVERVIEW_EXCLUDED_DOMAINS:
            continue
        names = dict.fromkeys([entity["name"], *entity.get("aliases", [])])
        info: dict[str, Any] = {"names": ", ".join(names), "domain": entity["domain"]}
        if entity.get("area"):
            info["areas"] = entity["area"]
        entities.append((entity["name"], info))
    for entity in home.get("context_entities") or []:
        if entity["domain"] in OVERVIEW_EXCLUDED_DOMAINS:
            continue
        info = {"names": entity["name"], "domain": entity["domain"]}
        if entity.get("area"):
            info["areas"] = entity["area"]
        entities.append((entity["name"], info))
    return [info for _, info in sorted(entities, key=lambda item: item[0])]


def home_areas(home: dict[str, Any]) -> dict[str, list[str]]:
    areas: dict[str, list[str]] = {
        area: []
        for area in [
            *(home.get("areas") or []),
            *(e.get("area") for e in home.get("entities", [])),
        ]
        if area
    }
    for area, aliases in (home.get("area_aliases") or {}).items():
        areas[area] = list(aliases)
    return areas


def area_context_for(home: dict[str, Any], utterance: str) -> AreaContext:
    return build_area_context(
        utterance, home_areas(home), home.get("sayso_entity_area")
    )


def _namespaced_prompt(text: str) -> str:
    for bare in ("GetLiveContext", "HassTurnOn", "HassTurnOff"):
        text = text.replace(bare, namespaced_tool_name(bare))
    return text


def serialize_context(home: dict[str, Any], utterance: str = "") -> str:
    entities = exposed_entities(home)
    if entities:
        api_prompt = "\n".join(
            [
                _namespaced_prompt(DYNAMIC_CONTEXT_PROMPT),
                STATIC_CONTEXT_HEADER,
                _dump_yaml(entities),
            ]
        )
    else:
        api_prompt = NO_ENTITIES_PROMPT
    api_prompt = "\n".join(
        [api_prompt, _namespaced_prompt(DEVICE_CONTROL_TOOL_USAGE_PROMPT)]
    )
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
    return f"script__{object_id}"


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


def _todo_list_names(home: dict[str, Any]) -> list[str]:
    names = []
    for entity in home.get("entities", []):
        if entity["domain"] == "todo":
            names.extend(
                dict.fromkeys([entity["name"], *entity.get("aliases", [])])
            )
    return names


def _offer_allowed(offer: str, home: dict[str, Any], domains: set[str]) -> bool:
    if offer == "always":
        return True
    if offer == "todo":
        return "todo" in domains
    if offer == "media_search_play":
        return any(
            entity["domain"] == "media_player"
            and {"search_media", "play_media"}
            <= set(entity.get("capabilities") or [])
            for entity in home.get("entities", [])
        )
    if offer.startswith("domains:"):
        return any(
            name in domains
            for name in offer.removeprefix("domains:").split(",")
        )
    if offer.startswith("domain:"):
        return offer.removeprefix("domain:") in domains
    raise ValueError(f"unknown tool offer condition {offer!r}")


def _snapshot_tool_for_home(
    tool: dict[str, Any], home: dict[str, Any]
) -> dict[str, Any]:
    if tool["function"]["name"] != "todo__get_items":
        return tool
    params = json.loads(json.dumps(tool["function"]["parameters"]))
    params["properties"]["todo_list"]["enum"] = _todo_list_names(home)
    return {**tool, "function": {**tool["function"], "parameters": params}}


def production_catalog(
    home: dict[str, Any], *, removed_tools: list[str] | None = None
) -> list[dict[str, Any]]:
    removed = removed_tools or []
    domains = {entity["domain"] for entity in home.get("entities", [])}
    offered = [
        _snapshot_tool_for_home(tool, home)
        for tool in _v2_openai_tools()
        if _offer_allowed(tool.get("offer", ""), home, domains)
    ]
    catalog = _compile_catalog([*offered, *_script_tools(home)])
    unknown = set(removed) - {tool["function"]["name"] for tool in catalog}
    if unknown:
        raise ValueError(
            f"cannot remove tools the catalog does not offer: {sorted(unknown)}"
        )
    return [tool for tool in catalog if tool["function"]["name"] not in set(removed)]


def render_case(case: Case) -> dict[str, Any]:
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
            target_area_source=area.target_area_source,
            clarification_candidates=tuple(
                (expected.get("clarification") or {}).get("candidates") or ()
            ),
        ),
    }


def fingerprint_inputs(cases: list[Case]) -> str:
    digest = hashlib.sha256()
    for case in cases:
        rendered = render_case(case)
        payload = json.dumps(
            {"messages": rendered["messages"], "tools": rendered["tools"]},
            sort_keys=True,
        )
        digest.update(payload.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _grounding_home(home: dict[str, Any]) -> grounding.GroundingHome:
    entities = tuple(
        grounding.GroundingEntity(
            entity_id=entity.get("entity_id", ""),
            name=entity["name"],
            aliases=tuple(entity.get("aliases") or ()),
            domain=entity["domain"],
            area=entity.get("area"),
            device_id=None,
        )
        for entity in home.get("entities", [])
    )
    return grounding.GroundingHome(
        entities=entities,
        areas={area: tuple(aliases) for area, aliases in home_areas(home).items()},
        satellite_area=home.get("satellite_area"),
        satellite_device_id=None,
    )


def _action_tools(names: set[str]) -> frozenset[str]:
    return frozenset(
        name
        for name in names
        if name not in QUERY_TOOLS and name != TODO_GET_ITEMS_TOOL
    )


def _info_flags(outcome: post_model.TurnOutcome) -> list[str]:
    checks = (
        ("correction_used", outcome.correction_used),
        ("area_dropped", outcome.area_dropped),
        ("grounding_clarify", outcome.clarified),
        ("done_shortcut", outcome.done_shortcut),
    )
    return [name for name, active in checks if active]


def _case_grammar(
    tools: list[dict[str, Any]], markers: bool | None
) -> tuple[str | None, str | None]:
    if markers is None:
        return None, None
    try:
        return build_grammar(tools, markers=markers), None
    except GrammarUnsupported as err:
        return None, err.reason


def _run_case_turn(
    case: Case,
    rendered: dict[str, Any],
    adapter: ModelAdapter,
    schema_fingerprint: str,
    grammar: str | None = None,
) -> tuple[post_model.TurnOutcome, _Trace]:
    tools = rendered["tools"]
    exposed = rendered["exposed_domains"]
    prompt_messages = _area_context.tools_before_area_messages(
        rendered["messages"], tools
    )
    sent_tools = tools if prompt_messages is rendered["messages"] else []
    transcript = list(prompt_messages)
    trace = _Trace()
    names = {tool["function"]["name"] for tool in tools}
    grounding_home = _grounding_home(load_home(case.household))
    request_options = {} if grammar is None else {"grammar": grammar}

    async def complete(messages: list[dict[str, Any]], *, correction: bool):
        started = time.perf_counter()
        try:
            raw = adapter.complete(messages, sent_tools, **request_options)
        except _TRANSPORT_ERRORS as err:
            trace.raws.append({"transport_error": str(err)})
            trace.transport_error = str(err)
            trace.parse_error = str(err)
            if isinstance(err, TimeoutError):
                raise exceptions.SaySoTimeoutError(str(err)) from err
            raise exceptions.SaySoConnectionError(str(err)) from err
        finally:
            trace.latencies_ms.append((time.perf_counter() - started) * 1000.0)
        trace.raws.append(raw)
        try:
            result = completion.parse_completion_result(raw)
        except exceptions.SaySoInvalidResponseError as err:
            trace.parse_error = str(err)
            raise
        trace.violations = contract_violations(
            [(call.name, call.arguments) for call in result.tool_calls],
            tools,
            exposed,
        )
        return result

    def validate(tool_calls):
        return post_model.ValidationResult(
            [(call, call.arguments) for call in tool_calls], []
        )

    async def execute(validated):
        transcript.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    correction._tool_call_message(call.id, call.name, args)
                    for call, args in validated
                ],
            }
        )
        transcript.extend(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(_FAKE_RESULT),
            }
            for call, _ in validated
        )
        return post_model.ExecutionResult(failed=False)

    turn = post_model.TurnInput(
        user_text=case.utterance,
        transcript=lambda: transcript,
        tools=tuple(tools),
        schema_fingerprint=schema_fingerprint,
        available_names=names,
        action_tools=_action_tools(names),
        exposed_domains=exposed,
        home=lambda: grounding_home,
        tool_api=True,
        iteration_limit=lambda: PRODUCTION_MAX_TOOL_ITERATIONS,
        now=time.monotonic(),
        held_ttl=PRODUCTION_HELD_TTL_SECONDS,
        read_only=frozenset(names & tool_contract.READ_ONLY_FALLBACK),
    )
    outcome = asyncio.run(
        post_model.run_turn(
            turn,
            complete=complete,
            validate=validate,
            execute=execute,
            recorder=_NullRecorder(),
        )
    )
    return outcome, trace


def evaluate(
    cases: list[Case],
    adapter: ModelAdapter,
    *,
    grammar_markers: bool | None = None,
) -> list[CaseResult]:
    schema_fingerprint = production_contract_fingerprint()
    results: list[CaseResult] = []
    for case in cases:
        rendered = render_case(case)
        home = load_home(case.household)
        grammar, skip_reason = _case_grammar(rendered["tools"], grammar_markers)
        outcome, trace = _run_case_turn(
            case, rendered, adapter, schema_fingerprint, grammar
        )
        first_turn = read_completion(
            trace.raws[0], rendered["tools"], rendered["exposed_domains"]
        )
        final_turn = Turn(
            raw=trace.raws[0],
            text=outcome.speech,
            calls=[
                {"name": call.name, "arguments": call.arguments}
                for call in outcome.executed
            ],
            parse_error=trace.parse_error,
            violations=trace.violations,
        )
        result = score(case.id, rendered["expectation"], final_turn, home=home)
        result.first_pass = score(
            case.id, rendered["expectation"], first_turn, home=home
        ).outcome
        result.info_flags = _info_flags(outcome)
        result.grammar_skip_reason = skip_reason
        result.request_latencies_ms = trace.latencies_ms
        if trace.transport_error:
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
        "checkpoint": getattr(adapter, "model", None)
        or (extra_metadata or {}).get("checkpoint"),
        "code_revision": _git_revision(),
        "case_hash": fingerprint_cases(cases),
        "input_hash": fingerprint_inputs(cases),
        "suite_hash": hashlib.sha256(
            "\n".join(case.id for case in cases).encode()
        ).hexdigest(),
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
        metadata["server"] = getattr(adapter, "server")
    (dest / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = {
        "summary": summary,
        "promotion": {"passed": not failures, "failures": failures} if gates else None,
    }
    (dest / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
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
