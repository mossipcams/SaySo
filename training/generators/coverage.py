"""What supervision an accepted row actually carries.

Quota accounting and the dataset audit both need the same answer to one question:
does this row teach the operation its metadata claims? Metadata alone cannot say
so -- a refusal tagged ``media_players``/``turn_on`` carries exactly that metadata
and teaches the opposite. Everything here reads the rendered row instead.
"""

from __future__ import annotations

import json
from typing import Any

from generators.capability_registry import CAPABILITIES, SCRIPT_ACTION_TOOL
from generators.tools import _operation_tool

# Outcomes a row can carry. Exactly one applies.
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
    "area_unavailable": ABSENCE,
    "unsupported": UNSUPPORTED,
    "device_unsupported": UNSUPPORTED,
    "refuse": UNSUPPORTED,
}


def row_calls(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Assistant tool calls as ``{"name", "arguments"}`` with arguments parsed."""
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


def row_script_tools(row: dict[str, Any]) -> set[str]:
    """Per-home script tools offered to this row (named after the script, not Hass*)."""
    return {
        tool["function"]["name"]
        for tool in row.get("tools") or []
        if not tool["function"]["name"].startswith(("Hass", "Get"))
    }


def expected_tool(capability: str, operation: str) -> str | None:
    """Tool name a positive row for this operation must call.

    Scripts are their own tools, so their real names are per home;
    ``SCRIPT_ACTION_TOOL`` stands in and is resolved against the row's tool list.
    """
    if capability == "scripts" and operation == "run":
        return SCRIPT_ACTION_TOOL
    try:
        return _operation_tool(operation, capability)
    except KeyError:
        return None


def _domain_matches(call: dict[str, Any], capability: str) -> bool:
    """The call must land on this capability's domain, not merely name its tool."""
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
    # Remaining tools are single-domain by contract (timers, vacuum, climate,
    # media, volume), so matching the tool already fixed the domain.
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
    """Return the supervision facets of one accepted row.

    ``positive`` is true only when the assistant turn calls the tool the claimed
    operation maps to, on the claimed domain, against an actual target.
    """
    meta = row.get("metadata") or {}
    capability = meta.get("capability")
    operation = meta.get("operation")
    calls = row_calls(row)
    reason = meta.get("no_action_reason")

    if not calls:
        outcome = _NO_ACTION_OUTCOMES.get(reason or "", CLARIFY)
    elif any(call["name"] == "GetLiveContext" for call in calls):
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
        name = call["name"]
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


def positive_key(facets: dict[str, Any]) -> tuple[int, str, str] | None:
    if not facets["positive"]:
        return None
    return (facets["tier"], facets["capability"], facets["operation"])


def negative_key(facets: dict[str, Any]) -> tuple[int, str, str, str] | None:
    if facets["positive"] or facets["outcome"] in POSITIVE_OUTCOMES:
        return None
    return (facets["tier"], facets["capability"], facets["operation"], facets["outcome"])
