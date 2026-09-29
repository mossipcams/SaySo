
from __future__ import annotations

import json
from typing import Any

from generators.capability_registry import CAPABILITIES, SCRIPT_ACTION_TOOL
from generators.tools import _operation_tool

ACTION = "action"
STATUS = "status"
CLARIFY = "clarify"
ABSENCE = "absence"
UNSUPPORTED = "unsupported"

POSITIVE_OUTCOMES = frozenset({ACTION, STATUS})
NEGATIVE_OUTCOMES = frozenset({CLARIFY, ABSENCE, UNSUPPORTED})
OUTCOMES = POSITIVE_OUTCOMES | NEGATIVE_OUTCOMES

_NO_ACTION_OUTCOMES = {
    "clarify": CLARIFY,
    "not_understood": CLARIFY,
    "area_unavailable": ABSENCE,
    "unsupported": UNSUPPORTED,
    "device_unsupported": UNSUPPORTED,
    "refuse": UNSUPPORTED,
}


def row_calls(row: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message in row.get("messages", []):
        for call in message.get("tool_calls") or []:
            function = call.get("function", call)
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            calls.append({"name": function.get("name"), "arguments": arguments or {}})
    return calls


def bare_tool_name(name: str | None) -> str:
    return (name or "").rsplit("__", 1)[-1]


def row_script_tools(row: dict[str, Any]) -> set[str]:
    return {
        tool["function"]["name"]
        for tool in row.get("tools") or []
        if not bare_tool_name(tool["function"]["name"]).startswith(("Hass", "Get"))
    }


def expected_tool(capability: str, operation: str) -> str | None:
    if capability == "scripts" and operation == "run":
        return SCRIPT_ACTION_TOOL
    try:
        return _operation_tool(operation, capability)
    except KeyError:
        return None


def _domain_matches(call: dict[str, Any], capability: str) -> bool:
    cap = CAPABILITIES[capability]
    arguments = call["arguments"]
    domain = arguments.get("domain")
    if isinstance(domain, str):
        domain = [domain]
    if domain:
        return cap.domain in domain
    device_class = arguments.get("device_class")
    if device_class:
        return bool(cap.device_class and cap.device_class in device_class)
    return True


def targeting_mode(call: dict[str, Any]) -> str:
    arguments = call["arguments"]
    if arguments.get("name"):
        return "individual"
    if arguments.get("floor"):
        return "floor"
    if arguments.get("area"):
        return "area"
    return "context"


def _target_matches_expected(row: dict[str, Any], call: dict[str, Any]) -> bool:
    name = call["arguments"].get("name")
    expected = (row.get("metadata") or {}).get("expected_target_names")
    if not name or expected is None:
        return True
    return str(name).casefold() in {str(target).casefold() for target in expected}


def classify_row(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("metadata") or {}
    capability = meta.get("capability")
    operation = meta.get("operation")
    calls = row_calls(row)
    reason = meta.get("no_action_reason")

    if not calls:
        outcome = _NO_ACTION_OUTCOMES.get(reason or "", CLARIFY)
    elif any(bare_tool_name(call["name"]) == "GetLiveContext" for call in calls):
        outcome = STATUS
    else:
        outcome = ACTION

    facets: dict[str, Any] = {
        "tier": meta.get("tier"),
        "capability": capability,
        "operation": operation,
        "outcome": outcome,
        "tools": sorted({call["name"] for call in calls if call["name"]}),
        "targeting": targeting_mode(calls[0]) if calls else "none",
        "positive": False,
        "no_action_reason": reason,
        "real_home": str(meta.get("home_id") or "").startswith("real_home"),
    }
    if capability not in CAPABILITIES or outcome not in POSITIVE_OUTCOMES:
        return facets

    wanted = expected_tool(capability, operation)
    if wanted is None:
        return facets
    script_tools = row_script_tools(row)
    for call in calls:
        name = bare_tool_name(call["name"])
        if wanted == SCRIPT_ACTION_TOOL:
            if name in script_tools:
                facets["positive"] = True
                facets["targeting"] = "individual"
                break
            continue
        if name != wanted or not _domain_matches(call, capability):
            continue
        arguments = call["arguments"]
        has_target = bool(arguments.get("name") or arguments.get("area") or arguments.get("floor"))
        if (not has_target and capability != "timers") or not _target_matches_expected(row, call):
            continue
        facets["positive"] = True
        facets["targeting"] = targeting_mode(call)
        break
    return facets


