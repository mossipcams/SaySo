"""Tests for evaluation metrics."""

from __future__ import annotations

from evals.metrics import (
    normalize_json_value,
    score_expected_vs_actual,
    score_tool_call_protocol,
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
