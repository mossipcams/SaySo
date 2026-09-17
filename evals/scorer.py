"""One expected-behavior scorer for every SaySo model eval."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any

from evals.outcomes import (
    QUERY_TOOLS,
    PASSING_OUTCOMES,
    CaseResult,
    Expectation,
    Outcome,
    ResponseType,
    Turn,
)

# Deliberately narrow. A clarification is a question that asks the user to pick;
# a refusal says the thing cannot be done. Both are covered by tests.
_CLARIFICATION_CUES = re.compile(
    r"\b(which|did you mean|do you mean|could you specify|can you specify|please specify|clarify)\b",
    re.I,
)
_REFUSAL_CUES = re.compile(
    r"\b(can't|cannot|can not|unable to|not able to|don't have|do not have|doesn't have|does not have"
    r"|doesn't support|does not support|isn't available|is not available|not available|not supported"
    r"|there is no|there's no|there are no|has no|no such)\b",
    re.I,
)


def _plain(text: str | None) -> str:
    return (text or "").replace("’", "'").strip()


def is_clarification(text: str | None) -> bool:
    plain = _plain(text)
    return "?" in plain and bool(_CLARIFICATION_CUES.search(plain))


def is_refusal(text: str | None) -> bool:
    plain = _plain(text)
    return bool(plain) and not is_clarification(plain) and bool(_REFUSAL_CUES.search(plain))


def _canonical(call: dict[str, Any]) -> str:
    return json.dumps({"name": call["name"], "arguments": call["arguments"]}, sort_keys=True)


def is_action_call(call: dict[str, Any]) -> bool:
    return call.get("name") not in QUERY_TOOLS


def _affects(call: dict[str, Any], entity: dict[str, Any]) -> bool:
    """Whether an action call could change ``entity``: by name, or by an area
    (and optional domain) that covers it."""
    if not is_action_call(call):
        return False
    arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    name = arguments.get("name")
    if isinstance(name, str):
        return name.casefold() == str(entity["name"]).casefold()
    area = arguments.get("area")
    if not isinstance(area, str) or area.casefold() != str(entity.get("area", "")).casefold():
        return False
    domains = arguments.get("domain")
    domains = domains if isinstance(domains, list) else [domains] if domains else []
    return not domains or entity.get("domain") in domains


def classify(expectation: Expectation, turn: Turn) -> Outcome:
    if turn.parse_error is not None:
        return Outcome.MALFORMED_OUTPUT
    if any(violation["code"] == "unknown_tool" for violation in turn.violations):
        return Outcome.UNKNOWN_TOOL
    if turn.violations:
        return Outcome.INVALID_ARGUMENTS
    wants_calls = expectation.response_type in (ResponseType.ACTION, ResponseType.STATUS)
    if turn.calls:
        if not wants_calls:
            return Outcome.UNEXPECTED_TOOL_CALL
        actual = Counter(_canonical(call) for call in turn.calls)
        expected = Counter(_canonical(call) for call in expectation.calls)
        if actual == expected:
            return Outcome.VALID_MULTI_TOOL_CALL if len(expectation.calls) > 1 else Outcome.VALID_TOOL_CALL
        if not actual - expected:
            return Outcome.MISSING_TOOL_CALL
        return Outcome.UNEXPECTED_TOOL_CALL
    if wants_calls:
        return Outcome.MISSING_TOOL_CALL
    if expectation.response_type == ResponseType.CLARIFICATION:
        return Outcome.VALID_CLARIFICATION if is_clarification(turn.text) else Outcome.WRONG_RESPONSE_TYPE
    return Outcome.VALID_REFUSAL if is_refusal(turn.text) else Outcome.WRONG_RESPONSE_TYPE


def score(case_id: str, expectation: Expectation, turn: Turn) -> CaseResult:
    outcome = classify(expectation, turn)
    flags = []
    category = expectation.category
    expected = expectation.response_type
    if expected == ResponseType.CLARIFICATION and any(is_action_call(call) for call in turn.calls):
        flags.append("action_on_ambiguity")
    if expected == ResponseType.REFUSAL and turn.calls:
        flags.append("tool_call_on_unavailable")
    if expected == ResponseType.STATUS and any(is_action_call(call) for call in turn.calls):
        flags.append("status_state_change")
    if any(_affects(call, entity) for call in turn.calls for entity in expectation.forbidden_entities):
        flags.append("excluded_entity_affected")
    if expectation.target_area and any(
        isinstance(call["arguments"].get("area"), str)
        and call["arguments"]["area"] != expectation.target_area
        for call in turn.calls
        if is_action_call(call) and isinstance(call.get("arguments"), dict)
    ):
        flags.append("wrong_target_area")
    passed = outcome in PASSING_OUTCOMES and not flags
    return CaseResult(
        case_id=case_id,
        category=category,
        outcome=outcome,
        passed=passed,
        flags=flags,
        turn=turn,
        expected={
            "response_type": expectation.response_type.value,
            "calls": list(expectation.calls),
            "forbidden_entities": list(expectation.forbidden_entities),
            "target_area": expectation.target_area,
        },
    )


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    by_category: dict[str, list[CaseResult]] = defaultdict(list)
    for result in results:
        by_category[result.category].append(result)
    expected_type = {r.case_id: r.expected["response_type"] for r in results}
    action = [r for r in results if expected_type[r.case_id] in ("action", "status")]
    no_call = [r for r in results if expected_type[r.case_id] in ("clarification", "refusal")]
    clarify = [r for r in results if expected_type[r.case_id] == "clarification"]
    refuse = [r for r in results if expected_type[r.case_id] == "refusal"]
    outcomes = Counter(r.outcome.value for r in results)
    flags = Counter(flag for r in results for flag in r.flags)
    # A gate, not a formality: an unparseable completion must never score as a
    # successful abstention, whatever changes in classify().
    flags["malformed_treated_as_abstention"] = sum(
        r.passed and r.outcome == Outcome.MALFORMED_OUTPUT for r in results
    )
    total = len(results)
    return {
        "total": total,
        "passed": sum(r.passed for r in results),
        "overall_pass_rate": _rate(sum(r.passed for r in results), total),
        "category_pass_rate": {
            category: _rate(sum(r.passed for r in rows), len(rows))
            for category, rows in sorted(by_category.items())
        },
        "action_execution_accuracy": _rate(sum(r.passed for r in action), len(action)),
        "false_action_rate": _rate(sum(bool(r.turn.calls) for r in no_call), len(no_call)),
        "clarification_accuracy": _rate(sum(r.passed for r in clarify), len(clarify)),
        "refusal_accuracy": _rate(sum(r.passed for r in refuse), len(refuse)),
        "malformed_output_rate": _rate(outcomes[Outcome.MALFORMED_OUTPUT], total),
        "unknown_tool_rate": _rate(outcomes[Outcome.UNKNOWN_TOOL], total),
        "invalid_argument_rate": _rate(outcomes[Outcome.INVALID_ARGUMENTS], total),
        "outcomes": dict(sorted(outcomes.items())),
        "flags": dict(sorted(flags.items())),
    }


def check_gates(
    summary: dict[str, Any],
    gates: dict[str, Any],
    *,
    expected_count: int | None = None,
) -> list[str]:
    """Every promotion gate the summary fails. Empty means promotable.

    Incomplete runs — missing cases or infrastructure errors — cannot pass.
    """
    failures = []
    if expected_count is not None and summary["total"] != expected_count:
        failures.append(f"incomplete run: scored {summary['total']} of {expected_count} cases")
    if summary["flags"].get("transport_error"):
        failures.append(
            f"incomplete run: {summary['flags']['transport_error']} infrastructure error(s)"
        )
    if summary["overall_pass_rate"] < gates["min_overall_pass_rate"]:
        failures.append(
            f"overall pass rate {summary['overall_pass_rate']} < {gates['min_overall_pass_rate']}"
        )
    for category, minimum in gates.get("category_min_pass_rate", {}).items():
        rate = summary["category_pass_rate"].get(category)
        if rate is None:
            failures.append(f"required category {category} has no cases")
        elif rate < minimum:
            failures.append(f"category {category} pass rate {rate} < {minimum}")
    for metric, maximum in gates.get("max_rates", {}).items():
        if summary[metric] > maximum:
            failures.append(f"{metric} {summary[metric]} > {maximum}")
    for flag in gates.get("zero_tolerance_flags", []):
        if summary["flags"].get(flag):
            failures.append(f"{flag} occurred {summary['flags'][flag]} time(s)")
    return failures
