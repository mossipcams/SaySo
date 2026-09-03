"""Tests for SaySo payload fixture compatibility."""

from __future__ import annotations

import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_sayso_payload_openai_shape() -> None:
    payload = json.loads((FIXTURES / "sayso_payload.json").read_text(encoding="utf-8"))
    assert isinstance(payload["messages"], list)
    assert isinstance(payload["tools"], list)
    assert payload["model"]
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 160
    roles = [m["role"] for m in payload["messages"]]
    assert roles[0] == "system"
    assert roles[1] == "user"
    for tool in payload["tools"]:
        assert tool["type"] == "function"
        fn = tool["function"]
        assert isinstance(fn["name"], str)
        assert isinstance(fn["parameters"], dict)


def test_ha_assist_tools_have_function_wrapper() -> None:
    tools = json.loads((FIXTURES / "ha_assist_tools.json").read_text(encoding="utf-8"))
    for tool in tools:
        assert tool["type"] == "function"
        assert "name" in tool["function"]
        assert "parameters" in tool["function"]


def test_multi_tool_fixture_expectations() -> None:
    spec = json.loads((FIXTURES / "multi_tool_utterance.json").read_text(encoding="utf-8"))
    assert len(spec["initial_tool_calls"]) == 2
    assert spec["follow_up_must_include"]["assistant_batches"] == 1
    assert spec["follow_up_must_include"]["tool_call_content_empty"] is True
