"""Tests for LFM training adapter helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adapters.home_llm_v2 import convert_entry
from adapters.lfm import (
    LFM_BASE_MODEL,
    forbidden_home_llm_label_patterns,
    lfm_jsonl_line,
    prepare_lfm_example,
    validate_lfm_config_text,
)
from adapters.schema import v1_openai_tools

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _load_jsonl(name: str) -> list[dict]:
    lines = (FIXTURES / name).read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_lfm_base_model_is_lfm25_230m() -> None:
    assert LFM_BASE_MODEL == "LiquidAI/LFM2.5-230M"


def test_prepare_lfm_example_uses_full_v1_catalog() -> None:
    entry = _load_jsonl("home_llm_v2_example.jsonl")[0]
    converted = convert_entry(entry, seed=42)
    assert converted is not None
    prepared = prepare_lfm_example(converted)
    assert {tool["function"]["name"] for tool in prepared.tools} == {
        tool["function"]["name"] for tool in v1_openai_tools()
    }


def test_lfm_jsonl_line_keeps_string_arguments_and_openai_envelope() -> None:
    entry = _load_jsonl("home_llm_v2_example.jsonl")[0]
    converted = convert_entry(entry, seed=42)
    assert converted is not None
    line = json.loads(lfm_jsonl_line(converted))
    for tool in line["tools"]:
        assert tool["type"] == "function"
    assistant = next(
        message for message in line["messages"] if message.get("tool_calls")
    )
    args = assistant["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, str)
    assert "<tool_call>" not in json.dumps(line)


def test_lfm_rejects_home_llm_label_markers_in_content() -> None:
    entry = _load_jsonl("home_llm_v2_example.jsonl")[0]
    converted = convert_entry(entry, seed=42)
    assert converted is not None
    messages = list(converted.messages)
    messages[-1] = {
        **messages[-1],
        "content": 'Sure.<tool_call>{"name":"HassTurnOn","arguments":{}}</tool_call>',
    }
    broken = converted.__class__(messages=messages, tools=converted.tools)
    with pytest.raises(ValueError, match="Home-LLM"):
        prepare_lfm_example(broken)


def test_forbidden_home_llm_patterns_include_tool_call_tags() -> None:
    patterns = forbidden_home_llm_label_patterns()
    assert "<tool_call>" in patterns
    assert "</tool_call>" in patterns


def test_validate_lfm_config_text_rejects_home_llm_renderer() -> None:
    with pytest.raises(ValueError):
        validate_lfm_config_text("chat_template_jinja: <tool_call>{\"name\": \"X\"}</tool_call>")
