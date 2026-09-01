"""Strict model-output parser for ControlPlan extraction."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ValidationError

from sayso_server.control_plan import ControlPlan, NoActionPlan

_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)


def parse_model_output(text: str, *, intent: str) -> BaseModel:
    """Parse model text into a validated ControlPlan or a no-action fallback."""
    extracted = _extract_json_text(text)
    if extracted is None:
        return _invalid_plan(intent)

    try:
        payload = json.loads(extracted)
    except json.JSONDecodeError:
        return _invalid_plan(intent)

    if _is_tool_call_wrapper(payload):
        return _invalid_plan(intent)

    try:
        return ControlPlan.model_validate(payload)
    except ValidationError:
        return _invalid_plan(intent)


def _extract_json_text(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None

    fence_match = _FENCE_PATTERN.match(stripped)
    if fence_match is not None:
        return fence_match.group(1).strip()
    return stripped


def _is_tool_call_wrapper(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    if "tool_calls" in payload:
        return True
    if payload.get("type") == "function" and "function" in payload:
        return True
    if "name" in payload and "arguments" in payload and "outcome" not in payload:
        return True
    return False


def _invalid_plan(intent: str) -> NoActionPlan:
    resolved_intent = intent.strip() or "unknown"
    return NoActionPlan(
        outcome="no-action",
        intent=resolved_intent,
        reason="model_output_invalid",
    )
