"""Independent validation of generated training rows."""

from __future__ import annotations

import json
import re
from typing import Any

from adapters.schema import tool_schema_map, validate_tool_arguments, v2_openai_tools
from generators.tools import script_tool_name
from generators.capability_registry import CAPABILITIES, SupportLevel
from generators.gold import _type_label
from generators.stt_noise import _int_to_words

_BANNED = re.compile(r"<tool_call>|evals/cases/|tool_call_start", re.I)


def validate_spec(spec: dict[str, Any]) -> str | None:
    """Validate authoritative behavior against home and pinned v2 schema."""
    expected = spec.get("expected") or {}
    calls = expected.get("calls") or []
    if expected.get("kind") == "no_action" and calls:
        return "no_action_has_calls"
    cap = CAPABILITIES.get(spec.get("capability"))
    operation = next((op for op in cap.operations if op.name == spec.get("operation")), None) if cap else None
    if expected.get("kind") == "no_action":
        if spec.get("capability") == "timers" and expected.get("response") != "unsupported":
            return "timer_inventory_refusal"
        if expected.get("response") == "unsupported" and operation and operation.support != SupportLevel.UNAVAILABLE:
            requested = expected.get("requested", {}).get("calls", [])
            withheld = set(expected.get("unavailable_tools", []))
            if not requested or any(call["name"] not in withheld for call in requested):
                return "unsupported_without_withheld_tool"
    if any(call.get("name") == "GetLiveContext" for call in calls) and expected.get("kind") != "status":
        return "state_query_requires_status_label"
    entities = {entity["name"]: entity for entity in spec.get("home", {}).get("entities", [])}
    if expected.get("response") == "area_unavailable":
        unavailable = expected.get("unavailable") or {}
        area = unavailable.get("area", "").casefold()
        domains = {cap.domain for cap in CAPABILITIES.values() if _type_label(cap.name) == unavailable.get("type")}
        if any(e.get("area", "").casefold() == area and
               (e.get("domain") in domains or unavailable.get("type") == "devices") for e in entities.values()):
            return "contradictory_absence"
    if spec.get("category") in {"multi_action", "exclusion"} and len(calls) < 2:
        return "missing_multiple_actions"
    if spec.get("category") == "exclusion" and not spec.get("excluded_names"):
        return "missing_excluded_target"
    schemas = tool_schema_map(v2_openai_tools())
    excluded = set(spec.get("excluded_names") or [])
    # Per-script tools are named after the script and are not in the pinned catalog.
    script_names = {
        script_tool_name(entity)
        for entity in spec.get("home", {}).get("entities", [])
        if entity.get("domain") == "script"
    }
    for call in calls:
        name = call.get("name")
        arguments = call.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return "invalid_call_shape"
        if name in script_names:
            # Home Assistant builds these from the script's fields; ours take none.
            if arguments:
                return "script_tool_takes_no_arguments"
            if any(script_tool_name(e) == name and e["name"] in excluded
                   for e in entities.values() if e.get("domain") == "script"):
                return "excluded_entity_called"
            continue
        reason = validate_tool_arguments(name, arguments, schemas)
        if reason:
            return reason
        target = arguments.get("name")
        if target is not None and target not in entities and name not in {
            "HassStartTimer", "HassPauseTimer", "HassTimerStatus",
        }:
            return "unknown_canonical_entity"
        if target in excluded:
            return "excluded_entity_called"
    for canonical, spoken in (spec.get("spoken_targets") or {}).items():
        if canonical not in entities or not str(spoken).strip():
            return "invalid_spoken_target"
    return None


def validate_utterance(spec: dict[str, Any]) -> str | None:
    """Check utterance aligns with expected behavior."""
    utterance = spec.get("utterance")
    if not isinstance(utterance, str) or not utterance.strip():
        return "missing_utterance"
    if _BANNED.search(utterance):
        return "banned_marker"
    expected = spec.get("expected") or {}
    lowered = utterance.casefold()
    if expected.get("kind") == "action":
        calls = expected.get("calls") or []
        for call in calls:
            arguments = call.get("arguments") or {}
            for scope in ("area", "floor"):
                if arguments.get(scope) and str(arguments[scope]).casefold() not in lowered:
                    return "missing_expected_scope"
            for key, value in arguments.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                if key in {"hours", "minutes", "seconds"} and value == 0:
                    continue
                spellings = [str(value)]
                if float(value).is_integer():
                    spellings.append(str(int(value)))
                    words = _int_to_words(int(value))
                    if words:
                        spellings.append(words)
                if not any(re.search(r"(?<![\w.])" + re.escape(word) + r"(?!\w|\.\d)", lowered)
                           for word in spellings):
                    return "missing_expected_value"
        has_area_target = any(
            isinstance(c.get("arguments"), dict) and c["arguments"].get("area") and not c["arguments"].get("name")
            for c in calls
        )
        if (
            calls
            and not has_area_target
            and calls[0]["name"] not in {
                "HassCancelAllTimers",
                "HassStartTimer",
                "HassPauseTimer",
                "HassTimerStatus",
            }
        ):
            for name in spec.get("target_names") or []:
                spoken = spec.get("spoken_targets", {}).get(name, name).casefold()
                if spoken not in lowered and name.casefold() not in lowered:
                    area = (calls[0].get("arguments") or {}).get("area")
                    if not area or str(area).casefold() not in lowered:
                        return "missing_expected_target"
        for name in spec.get("excluded_names") or []:
            if "leave" not in lowered or name.casefold() not in lowered:
                return "missing_exclusion"
    if expected.get("kind") == "status" and "status" not in lowered and "what" not in lowered:
        return "status_not_query"
    return None


def validate_token_budget(spec: dict[str, Any], budget: int) -> str | None:
    """Reject rows that exceed token budget (char proxy)."""
    from generators.context import serialize_context

    context_len = len(serialize_context(spec.get("home", {})))
    utterance_len = len(spec.get("utterance") or "")
    if context_len + utterance_len > budget * 8:
        return "token_budget_exceeded"
    return None


def validate_row(spec: dict[str, Any], *, token_budget: int = 4096) -> str | None:
    reason = validate_spec(spec)
    if reason:
        return reason
    reason = validate_utterance(spec)
    if reason:
        return reason
    return validate_token_budget(spec, token_budget)
