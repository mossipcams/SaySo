"""Tests for evaluation metrics."""

from __future__ import annotations

from evals.metrics import (
    ExampleScore,
    normalize_json_value,
    score_expected_vs_actual,
    score_tool_call_protocol,
    summarize_scores,
)


def test_normalize_json_sorts_keys() -> None:
    assert normalize_json_value({"b": 2, "a": 1}) == {"a": 1, "b": 2}


def test_protocol_requires_empty_tool_call_content() -> None:
    ok = [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "HassTurnOn", "arguments": "{}"}}]}]
    bad = [{"role": "assistant", "content": "oops", "tool_calls": [{"function": {"name": "HassTurnOn", "arguments": "{}"}}]}]
    assert score_tool_call_protocol(ok)
    assert not score_tool_call_protocol(bad)


def test_multi_action_order_independent() -> None:
    expected = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "HassTurnOff", "arguments": '{"name": "Kitchen"}'}},
            {"function": {"name": "HassTurnOn", "arguments": '{"name": "Front Door"}'}},
        ]}
    ]
    actual = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "HassTurnOn", "arguments": '{"name": "Front Door"}'}},
            {"function": {"name": "HassTurnOff", "arguments": '{"name": "Kitchen"}'}},
        ]}
    ]
    _name_ok, args_ok, multi_ok, _cat = score_expected_vs_actual(expected, actual)
    assert args_ok
    assert multi_ok


def _assistant_with_calls(*names: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": name, "arguments": "{}"}} for name in names
            ],
        }
    ]


def test_score_expected_vs_actual_uses_unexpected_tool_call_when_gold_is_no_call() -> None:
    expected: list[dict] = [{"role": "assistant", "content": "Which light?"}]
    actual = _assistant_with_calls("HassTurnOn")
    _name_ok, args_ok, multi_ok, category = score_expected_vs_actual(expected, actual)
    assert not _name_ok
    assert not args_ok
    assert not multi_ok
    assert category == "unexpected_tool_call"


def test_score_expected_vs_actual_uses_missing_tool_call_when_actual_is_empty() -> None:
    expected = _assistant_with_calls("HassTurnOn")
    actual: list[dict] = [{"role": "assistant", "content": ""}]
    _name_ok, args_ok, multi_ok, category = score_expected_vs_actual(expected, actual)
    assert not _name_ok
    assert not args_ok
    assert not multi_ok
    assert category == "missing_tool_call"


def test_score_expected_vs_actual_uses_tool_count_mismatch_for_nonempty_unequal_counts() -> None:
    expected = _assistant_with_calls("HassTurnOn", "HassTurnOff")
    actual = _assistant_with_calls("HassTurnOn")
    _name_ok, args_ok, multi_ok, category = score_expected_vs_actual(expected, actual)
    assert category == "tool_count_mismatch"
    assert not _name_ok


def test_summarize_scores_counts_pass_single_action_and_no_tool() -> None:
    pass_single = ExampleScore(
        expected={"messages": _assistant_with_calls("HassTurnOn")},
        actual=_assistant_with_calls("HassTurnOn")[0],
        failure_category=None,
    )
    pass_no_tool = ExampleScore(
        expected={"messages": [{"role": "assistant", "content": "clarify"}]},
        actual={"role": "assistant", "content": "clarify"},
        failure_category=None,
    )
    summary = summarize_scores([pass_single, pass_no_tool])
    assert summary.total == 2
    assert summary.protocol_valid == 2
    assert summary.tool_name_exact == 2
    assert summary.args_exact == 2
    assert summary.schema_valid_args == 2
    assert summary.single_action_success == 1
    assert summary.no_tool_correct == 1
    assert summary.multi_action_exact == 0


def test_summarize_scores_counts_unexpected_missing_and_args_mismatch() -> None:
    unexpected = ExampleScore(
        expected={"messages": [{"role": "assistant", "content": "clarify"}]},
        actual=_assistant_with_calls("HassTurnOn")[0],
        failure_category="unexpected_tool_call",
    )
    missing = ExampleScore(
        expected={"messages": _assistant_with_calls("HassTurnOn")},
        actual={"role": "assistant", "content": ""},
        failure_category="missing_tool_call",
    )
    args_mismatch = ExampleScore(
        expected={
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "HassTurnOn",
                                "arguments": '{"name": "Kitchen"}',
                            }
                        }
                    ],
                }
            ]
        },
        actual={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "HassTurnOn",
                        "arguments": '{"name": "Office"}',
                    }
                }
            ],
        },
        failure_category="args_mismatch",
    )
    inference_error = ExampleScore(
        expected={"messages": _assistant_with_calls("HassTurnOn")},
        actual={"error": "boom"},
        failure_category="inference_error",
    )
    summary = summarize_scores([unexpected, missing, args_mismatch, inference_error])
    assert summary.total == 4
    assert summary.protocol_valid == 3
    assert summary.tool_name_exact == 1
    assert summary.args_exact == 0
    assert summary.schema_valid_args == 1
    assert summary.single_action_success == 0
    assert summary.multi_action_exact == 0
    assert summary.no_tool_correct == 0
