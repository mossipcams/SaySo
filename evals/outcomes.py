
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from sayso_contract import completion, exceptions, tool_contract

PRODUCTION_TEMPERATURE = 0
PRODUCTION_MAX_OUTPUT_TOKENS = 160

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

    category: str
    response_type: ResponseType
    calls: tuple[dict[str, Any], ...] = ()
    forbidden_entities: tuple[dict[str, Any], ...] = ()
    target_area: str | None = None


@dataclass(slots=True)
class Turn:

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
