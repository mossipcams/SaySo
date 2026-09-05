"""Independent validation of generated training rows."""

from __future__ import annotations

import json
import re
from typing import Any

from adapters.schema import tool_schema_map, validate_tool_arguments, v2_openai_tools

_BANNED = re.compile(r"<tool_call>|evals/cases/|tool_call_start", re.I)


def validate_spec(spec: dict[str, Any]) -> str | None:
    """Validate authoritative behavior against home and pinned v2 schema."""
    expected = spec.get("expected") or {}
    calls = expected.get("calls") or []
    if expected.get("kind") == "no_action" and calls:
        return "no_action_has_calls"
    entities = {entity["name"]: entity for entity in spec.get("home", {}).get("entities", [])}
    schemas = tool_schema_map(v2_openai_tools())
    excluded = set(spec.get("excluded_names") or [])
    for call in calls:
        name = call.get("name")
        arguments = call.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return "invalid_call_shape"
        reason = validate_tool_arguments(name, arguments, schemas)
        if reason:
            return reason
        target = arguments.get("name")
        if target is not None and target not in entities:
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
            if "leave" not in lowered and name.casefold() in lowered:
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
