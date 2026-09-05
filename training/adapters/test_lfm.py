"""Tests for LFM training adapter helpers."""

from __future__ import annotations

import json

import pytest

from adapters.lfm import (
    LFM_BASE_MODEL,
    forbidden_chatml_tool_call_patterns,
    lfm_jsonl_line,
    prepare_lfm_example,
    validate_lfm_config_text,
)
from adapters.schema import TrainingExample, v1_openai_tools


def _minimal_example() -> TrainingExample:
    return TrainingExample(
        messages=[
            {
                "role": "system",
                "content": "You are SaySo.",
                "train_on_turn": False,
            },
            {
                "role": "user",
                "content": "Turn on Kitchen Light",
                "train_on_turn": False,
            },
            {
                "role": "assistant",
                "content": "",
                "train_on_turn": True,
                "tool_calls": [
                    {
                        "id": "call_test",
                        "type": "function",
                        "function": {
                            "name": "HassTurnOn",
                            "arguments": json.dumps({"name": "Kitchen Light"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": json.dumps({"result": "Success"}),
                "train_on_turn": False,
                "tool_call_id": "call_test",
            },
            {
                "role": "assistant",
                "content": "Done.",
                "train_on_turn": True,
            },
        ],
        tools=v1_openai_tools(),
        metadata={"source": "inline_test"},
    )


def test_lfm_base_model_is_lfm25_230m() -> None:
    assert LFM_BASE_MODEL == "LiquidAI/LFM2.5-230M"


def test_prepare_lfm_example_uses_full_v1_catalog() -> None:
    prepared = prepare_lfm_example(_minimal_example())
    assert {tool["function"]["name"] for tool in prepared.tools} == {
        tool["function"]["name"] for tool in v1_openai_tools()
    }


def test_lfm_jsonl_line_keeps_string_arguments_and_openai_envelope() -> None:
    line = json.loads(lfm_jsonl_line(_minimal_example()))
    for tool in line["tools"]:
        assert tool["type"] == "function"
    assistant = next(message for message in line["messages"] if message.get("tool_calls"))
    args = assistant["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, str)
    assert "<tool_call>" not in json.dumps(line)


def test_lfm_rejects_chatml_tool_call_markers_in_content() -> None:
    converted = _minimal_example()
    messages = list(converted.messages)
    messages[-1] = {
        **messages[-1],
        "content": 'Sure.<tool_call>{"name":"HassTurnOn","arguments":{}}</tool_call>',
    }
    broken = converted.__class__(messages=messages, tools=converted.tools)
    with pytest.raises(ValueError, match="ChatML"):
        prepare_lfm_example(broken)


def test_forbidden_chatml_patterns_include_tool_call_tags() -> None:
    patterns = forbidden_chatml_tool_call_patterns()
    assert "<tool_call>" in patterns
    assert "</tool_call>" in patterns


def test_validate_lfm_config_text_rejects_chatml_renderer() -> None:
    with pytest.raises(ValueError):
        validate_lfm_config_text('chat_template_jinja: <tool_call>{"name": "X"}</tool_call>')
