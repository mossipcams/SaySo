"""Pytest checks for the independent behavioral voice-command suite."""

from __future__ import annotations

from evals.behavior.test_suite import load_behavior_cases, validate_behavior_cases


def test_behavior_suite_has_three_hundred_unique_cases() -> None:
    case_set = load_behavior_cases()
    validate_behavior_cases(case_set)
    assert all(case.id.startswith("sayso-behavior-v1-") for case in case_set.cases)
    assert {case.category for case in case_set.cases} >= {"core_control", "query"}


def test_behavior_suite_tool_calls_are_schema_grounded() -> None:
    case_set = load_behavior_cases()
    tool_names = {
        tool_call["name"]
        for case in case_set.cases
        for tool_call in case.expect.get("tool_calls", [])
        if isinstance(tool_call, dict)
    }
    assert tool_names <= {
        "GetDateTime",
        "GetLiveContext",
        "HassCancelAllTimers",
        "HassFanSetSpeed",
        "HassLightSet",
        "HassTurnOff",
        "HassTurnOn",
    }
