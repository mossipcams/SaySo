from __future__ import annotations

import pytest

from evals.cases import CaseError, _parse_case, case_index
from evals.runner import render_case


def make_payload(**extra) -> dict:
    payload = {
        "id": "test_history_case",
        "category": "ordinary",
        "tags": ["test"],
        "household": "realistic_eval_household_1",
        "utterance": "Turn the kitchen lights to 50 percent",
        "unavailable": None,
        "expected": {
            "response_type": "action",
            "calls": [
                {
                    "name": "intent__HassSetBrightness",
                    "arguments": {"domain": ["light"], "name": "Kitchen Counter Lights"},
                }
            ],
            "clarification": None,
            "forbidden_entities": [],
            "spoken_aliases": {},
        },
        "provenance": {"source": "test"},
    }
    payload.update(extra)
    return payload


def test_render_order_with_history_and_after_utterance() -> None:
    case = _parse_case(
        make_payload(
            history=[
                {"role": "user", "content": "Start a ten minute timer"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "intent__HassCreateTimer",
                                "arguments": {"minutes": 10},
                            },
                        }
                    ],
                },
            ],
            after_utterance=[
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": '{"success": true, "result": "Timer started"}',
                }
            ],
        ),
        source="test.jsonl",
        line=1,
    )
    messages = render_case(case)["messages"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "tool",
    ]
    assert messages[1]["content"] == "Start a ten minute timer"
    assert messages[3]["content"] == case.utterance
    assert messages[4]["tool_call_id"] == "call_1"


def test_existing_case_renders_two_messages() -> None:
    case = case_index()["realistic_eval_20260908_ordinary_01"]
    messages = render_case(case)["messages"]
    assert len(messages) == 2
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[1]["content"] == case.utterance


def test_history_with_unknown_role_raises() -> None:
    with pytest.raises(CaseError):
        _parse_case(
            make_payload(history=[{"role": "system", "content": "behave"}]),
            source="test.jsonl",
            line=1,
        )
