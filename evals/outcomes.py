"""Shared evaluation result types.

A completion is read exactly as the integration reads it: the default
backend's ``parse_completion_result``, then ``tool_contract.check_tool_calls``
against the tools the model was offered. Nothing is normalized on the way: an
unparseable completion is ``malformed_output`` (never "no call"), and a bare
``HassTurnOn`` when production offers ``intent__HassTurnOn`` is ``unknown_tool``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from sayso_contract import completion, exceptions, tool_contract

# Production's DEFAULT_TEMPERATURE / DEFAULT_MAX_OUTPUT_TOKENS (const.py imports
# Home Assistant, so the values are restated; tests pin them to const.py).
PRODUCTION_TEMPERATURE = 0
PRODUCTION_MAX_OUTPUT_TOKENS = 160

# Tools that read state. Any other call changes something.
QUERY_TOOLS = frozenset(
    {"homeassistant__GetLiveContext", "llm__GetDateTime", "intent__HassTimerStatus"}
)


class Outcome(StrEnum):
    VALID_TOOL_CALL = "valid_tool_call"
    VALID_MULTI_TOOL_CALL = "valid_multi_tool_call"
    VALID_CLARIFICATION = "valid_clarification"
    VALID_REFUSAL = "valid_refusal"
    MALFORMED_OUTPUT = "malformed_output"
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    UNEXPECTED_TOOL_CALL = "unexpected_tool_call"
    MISSING_TOOL_CALL = "missing_tool_call"
    WRONG_RESPONSE_TYPE = "wrong_response_type"


PASSING_OUTCOMES = frozenset(
    {
        Outcome.VALID_TOOL_CALL,
        Outcome.VALID_MULTI_TOOL_CALL,
        Outcome.VALID_CLARIFICATION,
        Outcome.VALID_REFUSAL,
    }
)


class ResponseType(StrEnum):
    ACTION = "action"
    STATUS = "status"
    CLARIFICATION = "clarification"
    REFUSAL = "refusal"


@dataclass(frozen=True, slots=True)
class Expectation:
    """What one case requires. ``calls`` are exact production calls."""

    category: str
    response_type: ResponseType
    calls: tuple[dict[str, Any], ...] = ()
    # Entities that must not be affected: {"name", "area", "domain"}.
    forbidden_entities: tuple[dict[str, Any], ...] = ()
    # The area an area-scoped call must use: the satellite's for an implicit
    # request, the named one when the request names an area.
    target_area: str | None = None


@dataclass(slots=True)
class Turn:
    """One model completion, read through the production parser and validator."""

    raw: Any
    text: str | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)
    parse_error: str | None = None
    violations: list[dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class CaseResult:
    case_id: str
    category: str
    outcome: Outcome
    passed: bool
    flags: list[str]
    turn: Turn
    expected: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "outcome": self.outcome.value,
            "passed": self.passed,
            "flags": self.flags,
            "raw_output": self.turn.raw,
            "text": self.turn.text,
            "calls": self.turn.calls,
            "parse_error": self.turn.parse_error,
            "violations": self.turn.violations,
            "expected": self.expected,
        }


def read_completion(
    raw: Any,
    tools: list[dict[str, Any]],
    exposed_domains: frozenset[str] | set[str] | None,
) -> Turn:
    """Parse and validate one completion body the way the integration does."""
    turn = Turn(raw=raw)
    try:
        result = completion.parse_completion_result(raw)
    except exceptions.SaySoInvalidResponseError as err:
        turn.parse_error = str(err)
        return turn
    turn.text = result.content
    turn.calls = [{"name": call.name, "arguments": call.arguments} for call in result.tool_calls]
    turn.violations = [
        asdict(violation)
        for violation in tool_contract.check_tool_calls(
            [(call["name"], call["arguments"]) for call in turn.calls],
            tools,
            exposed_domains,
        )
    ]
    return turn
