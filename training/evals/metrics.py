"""Evaluation metrics for SaySo tool-call training."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Fold common apostrophe-like characters to ASCII U+0027 for arg comparison.
_APOSTROPHE_LIKE = str.maketrans(
    {
        "\u2018": "'",  # LEFT SINGLE QUOTATION MARK
        "\u2019": "'",  # RIGHT SINGLE QUOTATION MARK
        "\u201a": "'",  # SINGLE LOW-9 QUOTATION MARK
        "\u201b": "'",  # SINGLE HIGH-REVERSED-9 QUOTATION MARK
        "\u02bc": "'",  # MODIFIER LETTER APOSTROPHE
        "\u2032": "'",  # PRIME
        "\uff07": "'",  # FULLWIDTH APOSTROPHE
    }
)


def fold_apostrophes(text: str) -> str:
    """Normalize apostrophe-like characters for stable string comparison."""
    return text.translate(_APOSTROPHE_LIKE)


def normalize_json_value(value: Any) -> Any:
    """Recursively normalize JSON for stable comparison."""
    if isinstance(value, dict):
        return {k: normalize_json_value(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [normalize_json_value(item) for item in value]
    if isinstance(value, str):
        stripped = fold_apostrophes(value.strip())
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
        return normalize_json_value(parsed)
    return value


def parse_tool_arguments(raw: Any) -> dict[str, Any] | None:
    """Parse tool call arguments to a dict."""
    if isinstance(raw, dict):
        return normalize_json_value(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return normalize_json_value(parsed)
    return None


def extract_assistant_tool_calls(
    messages: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Return tool call batches from assistant messages."""
    batches: list[list[dict[str, Any]]] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if calls:
            batches.append(calls)
    return batches


def tool_call_signature(call: dict[str, Any]) -> tuple[str, str]:
    """Return (name, canonical_args_json) for one tool call."""
    fn = call.get("function") or {}
    name = str(fn.get("name", ""))
    args = parse_tool_arguments(fn.get("arguments")) or {}
    return name, json.dumps(args, sort_keys=True, separators=(",", ":"))


@dataclass
class ExampleScore:
    """Per-example evaluation result."""

    expected: dict[str, Any]
    actual: dict[str, Any]
    failure_category: str | None = None
    latency_ms: float | None = None
    checkpoint_id: str | None = None


@dataclass
class MetricSummary:
    """Aggregate metric scores."""

    total: int = 0
    protocol_valid: int = 0
    tool_name_exact: int = 0
    args_exact: int = 0
    args_parse_as_json: int = 0
    single_action_success: int = 0
    multi_action_exact: int = 0
    no_tool_correct: int = 0
    expected_single_action: int = 0
    expected_multi_action: int = 0
    expected_no_tool: int = 0
    no_call_agreement: int = 0
    unsupported_correct: int = 0
    final_response_present: int = 0
    per_example: list[ExampleScore] = field(default_factory=list)

    def rates(self) -> dict[str, float]:
        if self.total == 0:
            return {key: 0.0 for key in [
                "protocol", "tool_name", "args_exact", "args_parse_as_json",
                "single_action", "multi_action", "no_tool", "call_no_call_agreement",
                "unsupported", "final_response",
            ]}
        return {
            "protocol": self.protocol_valid / self.total,
            "tool_name": self.tool_name_exact / self.total,
            "args_exact": self.args_exact / self.total,
            "args_parse_as_json": self.args_parse_as_json / self.total,
            "single_action": (
                self.single_action_success / self.expected_single_action
                if self.expected_single_action
                else 0.0
            ),
            "multi_action": (
                self.multi_action_exact / self.expected_multi_action
                if self.expected_multi_action
                else 0.0
            ),
            "no_tool": (
                self.no_tool_correct / self.expected_no_tool
                if self.expected_no_tool
                else 0.0
            ),
            "call_no_call_agreement": self.no_call_agreement / self.total,
            "unsupported": self.unsupported_correct / self.total,
            "final_response": self.final_response_present / self.total,
        }


def score_tool_call_protocol(messages: list[dict[str, Any]]) -> bool:
    """Return True when assistant tool-call turns use empty content."""
    for message in messages:
        if message.get("role") != "assistant":
            continue
        if message.get("tool_calls"):
            content = message.get("content")
            if content not in ("", None):
                return False
    return True


def score_expected_vs_actual(
    expected_messages: list[dict[str, Any]],
    actual_messages: list[dict[str, Any]],
    *,
    schemas: dict[str, dict[str, Any]] | None = None,
) -> tuple[bool, bool, bool, str | None]:
    """Score tool name, args, and multi-action match. Returns (name_ok, args_ok, multi_ok, category)."""
    expected_batches = extract_assistant_tool_calls(expected_messages)
    actual_batches = extract_assistant_tool_calls(actual_messages)

    if not expected_batches and not actual_batches:
        return True, True, True, None

    expected_calls = [call for batch in expected_batches for call in batch]
    actual_calls = [call for batch in actual_batches for call in batch]

    if len(expected_calls) != len(actual_calls):
        if not expected_calls and actual_calls:
            return False, False, False, "unexpected_tool_call"
        if expected_calls and not actual_calls:
            return False, False, False, "missing_tool_call"
        return False, False, False, "tool_count_mismatch"

    expected_sigs = {tool_call_signature(c) for c in expected_calls}
    actual_sigs = {tool_call_signature(c) for c in actual_calls}

    name_ok = {s[0] for s in expected_sigs} == {s[0] for s in actual_sigs}
    args_ok = expected_sigs == actual_sigs
    multi_ok = args_ok

    category = None
    if not name_ok:
        category = "tool_name_mismatch"
    elif not args_ok:
        category = "args_mismatch"

    if schemas:
        for call in actual_calls:
            fn = call.get("function") or {}
            name = fn.get("name")
            args = parse_tool_arguments(fn.get("arguments"))
            if not isinstance(name, str) or args is None:
                return name_ok, False, False, "schema_invalid"
            schema = schemas.get(name) or {}
            props = schema.get("properties") or {}
            if isinstance(props, dict):
                for key in args:
                    if key not in props:
                        return name_ok, False, False, "extra_argument"

    return name_ok, args_ok, multi_ok, category


def _tool_call_count(messages: list[dict[str, Any]]) -> int:
    """Count assistant tool calls across message batches."""
    return sum(len(batch) for batch in extract_assistant_tool_calls(messages))


def _actual_args_parse_as_json(messages: list[dict[str, Any]]) -> bool:
    """Return True when every actual tool call has parseable JSON object arguments."""
    calls = [call for batch in extract_assistant_tool_calls(messages) for call in batch]
    if not calls:
        return True
    for call in calls:
        fn = call.get("function") or {}
        if parse_tool_arguments(fn.get("arguments")) is None:
            return False
    return True


def _example_messages(item: ExampleScore, *, side: str) -> list[dict[str, Any]]:
    if side == "expected":
        return item.expected.get("messages") or []
    actual = item.actual
    if isinstance(actual, dict) and actual.get("error") is not None:
        return []
    if isinstance(actual, dict) and actual.get("role"):
        return [actual]
    if isinstance(actual, list):
        return actual
    return []


def _actual_is_inference_error(item: ExampleScore) -> bool:
    """Return True when the actual payload is an inference failure, not a model turn."""
    if item.failure_category == "inference_error":
        return True
    actual = item.actual
    return isinstance(actual, dict) and actual.get("error") is not None


def summarize_scores(scores: list[ExampleScore]) -> MetricSummary:
    """Build aggregate summary from per-example scores."""
    summary = MetricSummary(total=len(scores), per_example=scores)
    for item in scores:
        category = item.failure_category
        expected_messages = _example_messages(item, side="expected")
        actual_messages = _example_messages(item, side="actual")
        expected_count = _tool_call_count(expected_messages)
        actual_count = _tool_call_count(actual_messages)

        if expected_count == 0:
            summary.expected_no_tool += 1
        elif expected_count == 1:
            summary.expected_single_action += 1
        else:
            summary.expected_multi_action += 1

        if not _actual_is_inference_error(item):
            if (expected_count == 0) == (actual_count == 0):
                summary.no_call_agreement += 1

            if _actual_args_parse_as_json(actual_messages):
                summary.args_parse_as_json += 1

        if category not in {"inference_error", "protocol_invalid"}:
            summary.protocol_valid += 1

        if category is None:
            summary.tool_name_exact += 1
            summary.args_exact += 1
            if expected_count == 0:
                summary.no_tool_correct += 1
            elif expected_count == 1:
                summary.single_action_success += 1
            else:
                summary.multi_action_exact += 1
        elif category == "args_mismatch":
            summary.tool_name_exact += 1
    return summary
