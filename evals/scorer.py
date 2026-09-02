"""Score offline evaluation cases against recorded model outcomes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from custom_components.sayso.client import ToolCall
from custom_components.sayso.diagnostics import BoundaryFailureCode
from custom_components.sayso.schema import ToolArgumentValidationError


class CheckName(StrEnum):
    """Named scorer dimensions exercised by the fixed case set."""

    TOOL_NAME = "tool_name"
    TOOL_ARGS = "tool_args"
    TOOL_ORDER = "tool_order"
    WRONG_TOOL = "wrong_tool"
    INVALID_CALL = "invalid_call"
    CLARIFICATION = "clarification"
    PARTIAL_FAILURE = "partial_failure"
    SPOKEN_RESULT = "spoken_result"


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One versioned offline evaluation case."""

    id: str
    category: str
    scenario: str
    description: str
    expect: dict[str, Any]
    checks: tuple[CheckName, ...]


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Pass/fail outcome for one scorer dimension."""

    name: CheckName
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class CaseScore:
    """Aggregate score for one case."""

    case_id: str
    category: str
    scenario: str
    passed: bool
    checks: dict[str, CheckResult]
    detail: str = ""


@dataclass(frozen=True, slots=True)
class EvalActual:
    """Recorded outcome to score against one case."""

    tool_calls: tuple[ToolCall, ...] = ()
    spoken: str | None = None
    validation_errors: tuple[ToolArgumentValidationError, ...] = ()
    boundary_code: BoundaryFailureCode | None = None
    execution_failures: tuple[str, ...] = ()


def _canonical_arguments(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, sort_keys=True, separators=(",", ":"))


def _normalize_spoken(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(text.strip().casefold().split())


def _expected_tool_calls(expect: dict[str, Any]) -> list[dict[str, Any]]:
    tool_calls = expect.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    return [entry for entry in tool_calls if isinstance(entry, dict)]


def _check_tool_name(
    expect: dict[str, Any],
    actual: EvalActual,
) -> CheckResult:
    expected = _expected_tool_calls(expect)
    if not expected:
        return CheckResult(
            CheckName.TOOL_NAME,
            passed=True,
            detail="no tool expectations",
        )
    actual_names = [call.name for call in actual.tool_calls]
    expected_names = [entry["name"] for entry in expected]
    passed = actual_names == expected_names
    return CheckResult(
        CheckName.TOOL_NAME,
        passed=passed,
        detail=f"expected={expected_names} actual={actual_names}",
    )


def _check_tool_args(
    expect: dict[str, Any],
    actual: EvalActual,
) -> CheckResult:
    expected = _expected_tool_calls(expect)
    if not expected:
        return CheckResult(
            CheckName.TOOL_ARGS,
            passed=True,
            detail="no argument expectations",
        )
    if len(actual.tool_calls) != len(expected):
        return CheckResult(
            CheckName.TOOL_ARGS,
            passed=False,
            detail="tool call count mismatch",
        )
    for index, expected_entry in enumerate(expected):
        actual_call = actual.tool_calls[index]
        expected_args = expected_entry.get("arguments", {})
        if not isinstance(expected_args, dict):
            expected_args = {}
        passed = _canonical_arguments(actual_call.arguments) == _canonical_arguments(
            expected_args
        )
        if not passed:
            return CheckResult(
                CheckName.TOOL_ARGS,
                passed=False,
                detail=(
                    f"{actual_call.name}: expected={expected_args} "
                    f"actual={actual_call.arguments}"
                ),
            )
    return CheckResult(CheckName.TOOL_ARGS, passed=True, detail="arguments match")


def _check_tool_order(
    expect: dict[str, Any],
    actual: EvalActual,
) -> CheckResult:
    expected = _expected_tool_calls(expect)
    if len(expected) < 2:
        return CheckResult(
            CheckName.TOOL_ORDER,
            passed=True,
            detail="single-call case",
        )
    actual_names = [call.name for call in actual.tool_calls]
    expected_names = [entry["name"] for entry in expected]
    passed = actual_names == expected_names
    return CheckResult(
        CheckName.TOOL_ORDER,
        passed=passed,
        detail=f"expected={expected_names} actual={actual_names}",
    )


def _check_wrong_tool(
    expect: dict[str, Any],
    actual: EvalActual,
) -> CheckResult:
    forbidden = expect.get("forbidden_tools", [])
    expected_tool = expect.get("expected_tool")
    if not isinstance(forbidden, list):
        forbidden = []
    actual_names = {call.name for call in actual.tool_calls}
    forbidden_hits = sorted(name for name in actual_names if name in forbidden)
    if forbidden_hits:
        return CheckResult(
            CheckName.WRONG_TOOL,
            passed=False,
            detail=f"forbidden tools used: {forbidden_hits}",
        )
    if isinstance(expected_tool, str) and expected_tool not in actual_names:
        return CheckResult(
            CheckName.WRONG_TOOL,
            passed=False,
            detail=f"expected tool {expected_tool} not used",
        )
    return CheckResult(CheckName.WRONG_TOOL, passed=True, detail="tool selection ok")


def _check_invalid_call(
    expect: dict[str, Any],
    actual: EvalActual,
) -> CheckResult:
    expected_code = expect.get("boundary_code")
    if expected_code is None:
        return CheckResult(
            CheckName.INVALID_CALL,
            passed=not actual.validation_errors,
            detail="unexpected validation errors"
            if actual.validation_errors
            else "no validation errors",
        )
    if actual.boundary_code is None:
        return CheckResult(
            CheckName.INVALID_CALL,
            passed=False,
            detail="missing boundary code",
        )
    passed = actual.boundary_code.value == str(expected_code)
    return CheckResult(
        CheckName.INVALID_CALL,
        passed=passed,
        detail=f"expected={expected_code} actual={actual.boundary_code.value}",
    )


def _check_clarification(
    expect: dict[str, Any],
    actual: EvalActual,
) -> CheckResult:
    if actual.tool_calls:
        return CheckResult(
            CheckName.CLARIFICATION,
            passed=False,
            detail="tool calls present",
        )
    spoken = actual.spoken
    if not spoken or not spoken.strip():
        return CheckResult(
            CheckName.CLARIFICATION,
            passed=False,
            detail="missing spoken clarification",
        )
    contains = expect.get("spoken_contains")
    if isinstance(contains, list):
        normalized = _normalize_spoken(spoken)
        missing = [
            phrase
            for phrase in contains
            if _normalize_spoken(str(phrase)) not in normalized
        ]
        if missing:
            return CheckResult(
                CheckName.CLARIFICATION,
                passed=False,
                detail=f"missing phrases: {missing}",
            )
    return CheckResult(CheckName.CLARIFICATION, passed=True, detail="clarification ok")


def _check_partial_failure(
    expect: dict[str, Any],
    actual: EvalActual,
) -> CheckResult:
    expected_failures = expect.get("execution_failures", [])
    if not isinstance(expected_failures, list):
        expected_failures = []
    actual_failures = sorted(actual.execution_failures)
    expected_sorted = sorted(str(item) for item in expected_failures)
    if actual_failures != expected_sorted:
        return CheckResult(
            CheckName.PARTIAL_FAILURE,
            passed=False,
            detail=(
                f"expected failures={expected_sorted} "
                f"actual={actual_failures}"
            ),
        )
    expected_boundary = expect.get("boundary_code")
    if (
        expected_boundary is not None
        and actual.boundary_code is not None
        and actual.boundary_code.value != str(expected_boundary)
    ):
        return CheckResult(
            CheckName.PARTIAL_FAILURE,
            passed=False,
            detail="boundary code mismatch",
        )
    return CheckResult(
        CheckName.PARTIAL_FAILURE,
        passed=True,
        detail="partial failure matched",
    )


def _check_spoken_result(
    expect: dict[str, Any],
    actual: EvalActual,
) -> CheckResult:
    expected_spoken = expect.get("spoken")
    if not isinstance(expected_spoken, str):
        contains = expect.get("spoken_contains")
        if isinstance(contains, list) and contains:
            normalized = _normalize_spoken(actual.spoken)
            missing = [
                phrase
                for phrase in contains
                if _normalize_spoken(str(phrase)) not in normalized
            ]
            passed = not missing
            return CheckResult(
                CheckName.SPOKEN_RESULT,
                passed=passed,
                detail=f"missing phrases: {missing}" if missing else "spoken ok",
            )
        return CheckResult(
            CheckName.SPOKEN_RESULT,
            passed=True,
            detail="no spoken expectation",
        )
    passed = _normalize_spoken(actual.spoken) == _normalize_spoken(expected_spoken)
    return CheckResult(
        CheckName.SPOKEN_RESULT,
        passed=passed,
        detail=f"expected={expected_spoken!r} actual={actual.spoken!r}",
    )


_CHECKERS = {
    CheckName.TOOL_NAME: _check_tool_name,
    CheckName.TOOL_ARGS: _check_tool_args,
    CheckName.TOOL_ORDER: _check_tool_order,
    CheckName.WRONG_TOOL: _check_wrong_tool,
    CheckName.INVALID_CALL: _check_invalid_call,
    CheckName.CLARIFICATION: _check_clarification,
    CheckName.PARTIAL_FAILURE: _check_partial_failure,
    CheckName.SPOKEN_RESULT: _check_spoken_result,
}


def score_case(case: EvalCase, actual: EvalActual) -> CaseScore:
    """Score one case against a recorded actual outcome."""
    checks: dict[str, CheckResult] = {}
    passed = True
    for check_name in case.checks:
        checker = _CHECKERS[check_name]
        result = checker(case.expect, actual)
        checks[result.name.value] = result
        passed = passed and result.passed
    return CaseScore(
        case_id=case.id,
        category=case.category,
        scenario=case.scenario,
        passed=passed,
        checks=checks,
    )


def case_score_to_dict(score: CaseScore) -> dict[str, Any]:
    """Convert one case score to a JSON-serializable mapping."""
    return {
        "case_id": score.case_id,
        "category": score.category,
        "scenario": score.scenario,
        "passed": score.passed,
        "checks": {
            name: {"passed": result.passed, "detail": result.detail}
            for name, result in sorted(score.checks.items())
        },
    }
