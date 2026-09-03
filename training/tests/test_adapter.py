"""Tests for Home-LLM V2 adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adapters.home_llm_v2 import convert_entry, convert_jsonl_stream
from adapters.schema import ALLOWED_HASS_TOOLS, RejectionStats

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _load_jsonl(name: str) -> list[dict]:
    lines = (FIXTURES / name).read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_converts_home_llm_v2_single_action() -> None:
    entry = _load_jsonl("home_llm_v2_example.jsonl")[0]
    converted = convert_entry(entry, seed=42)
    assert converted is not None
    assert len(converted.tools) == len(ALLOWED_HASS_TOOLS)
    assistant_tool = next(
        m for m in converted.messages if m.get("role") == "assistant" and m.get("tool_calls")
    )
    assert assistant_tool["content"] == ""
    assert assistant_tool["tool_calls"][0]["function"]["name"] == "HassTurnOn"
    assert isinstance(assistant_tool["tool_calls"][0]["function"]["arguments"], str)
    assert "id" in assistant_tool["tool_calls"][0]
    assert converted.tools[0]["type"] == "function"


def test_converts_multiple_tool_calls() -> None:
    entry = _load_jsonl("home_llm_v2_example.jsonl")[1]
    converted = convert_entry(entry, seed=7)
    assert converted is not None
    tool_batch = next(
        m for m in converted.messages if m.get("role") == "assistant" and m.get("tool_calls")
    )
    assert len(tool_batch["tool_calls"]) == 2
    names = {c["function"]["name"] for c in tool_batch["tool_calls"]}
    assert names == {"HassTurnOff", "HassTurnOn"}
    tool_msgs = [m for m in converted.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    assert all(m.get("tool_call_id") for m in tool_msgs)


def test_rejects_legacy_service_tools() -> None:
    for entry in _load_jsonl("legacy_rejected.jsonl"):
        assert convert_entry(entry, seed=1) is None


def test_rejects_unknown_tool_in_call() -> None:
    entry = _load_jsonl("legacy_rejected.jsonl")[2]
    converted = convert_entry(entry, seed=1)
    assert converted is None


def test_rejects_extra_arguments() -> None:
    entry = _load_jsonl("legacy_rejected.jsonl")[1]
    assert convert_entry(entry, seed=1) is None


def test_no_tool_conversational_preserved() -> None:
    entry = _load_jsonl("home_llm_v2_example.jsonl")[2]
    converted = convert_entry(entry, seed=3)
    assert converted is not None
    assistants = [m for m in converted.messages if m.get("role") == "assistant"]
    assert len(assistants) == 1
    assert "tool_calls" not in assistants[0]


def test_deduplication_is_deterministic() -> None:
    entry = _load_jsonl("home_llm_v2_example.jsonl")[0]
    line = json.dumps(entry)
    stream = [line, line, line]
    stats, written = convert_jsonl_stream(iter(stream), seed=99)
    assert written == 1
    assert stats.counts.get("duplicate", 0) == 2


def test_allowed_tools_subset() -> None:
    assert "HassTurnOn" in ALLOWED_HASS_TOOLS
    assert "GetLiveContext" in ALLOWED_HASS_TOOLS
    assert "HassFanSetSpeed" in ALLOWED_HASS_TOOLS
    assert "HassVacuumStart" not in ALLOWED_HASS_TOOLS
    assert "HassClimateSetTemperature" not in ALLOWED_HASS_TOOLS


def test_tool_result_linkage() -> None:
    entry = _load_jsonl("home_llm_v2_example.jsonl")[0]
    converted = convert_entry(entry, seed=5)
    assert converted is not None
    call_id = next(
        m["tool_calls"][0]["id"]
        for m in converted.messages
        if m.get("tool_calls")
    )
    tool_msg = next(m for m in converted.messages if m.get("role") == "tool")
    assert tool_msg["tool_call_id"] == call_id


@pytest.mark.parametrize("seed", [1, 1, 42, 42])
def test_deterministic_call_ids(seed: int) -> None:
    entry = _load_jsonl("home_llm_v2_example.jsonl")[0]
    first = convert_entry(entry, seed=seed)
    second = convert_entry(entry, seed=seed)
    assert first is not None and second is not None
    id1 = first.messages[2]["tool_calls"][0]["id"]
    id2 = second.messages[2]["tool_calls"][0]["id"]
    assert id1 == id2


def test_train_on_turn_assistant_only() -> None:
    entry = _load_jsonl("home_llm_v2_example.jsonl")[0]
    converted = convert_entry(entry, seed=42)
    assert converted is not None
    for message in converted.messages:
        role = message["role"]
        train = message.get("train_on_turn")
        if role in {"system", "user", "tool"}:
            assert train is False, f"{role} must not train"
        elif role == "assistant":
            assert train is True, "assistant turns must train (tool-call and final TTS)"


def test_sayso_view_keeps_string_arguments() -> None:
    entry = _load_jsonl("home_llm_v2_example.jsonl")[0]
    converted = convert_entry(entry, seed=42)
    assert converted is not None
    line = json.loads(converted.to_jsonl_line(view="sayso"))
    args = line["messages"][2]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, str)
    parsed = json.loads(args)
    assert "name" in parsed


def test_axolotl_view_parses_arguments_to_dict() -> None:
    entry = _load_jsonl("home_llm_v2_example.jsonl")[0]
    converted = convert_entry(entry, seed=42)
    assert converted is not None
    line = json.loads(converted.to_jsonl_line(view="axolotl"))
    args = line["messages"][2]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, dict)
    assert "name" in args


def test_legacy_reject_counts_once() -> None:
    entry = _load_jsonl("legacy_rejected.jsonl")[0]
    stats = RejectionStats()
    assert convert_entry(entry, seed=1, stats=stats) is None
    assert stats.counts.get("legacy_tool_name", 0) == 1


def test_schema_type_failure_rejected() -> None:
    entry = _load_jsonl("home_llm_v2_example.jsonl")[0]
    entry = json.loads(json.dumps(entry))
    entry["messages"][2]["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {"name": "Kitchen", "domain": "light"}
    )
    stats = RejectionStats()
    assert convert_entry(entry, seed=1, stats=stats) is None
    assert stats.counts.get("schema_validation_failed", 0) == 1


def test_preserves_split_metadata() -> None:
    entry = _load_jsonl("home_llm_v2_example.jsonl")[0]
    entry["metadata"] = {
        "template": "turn_on_area",
        "phrasing": "polite",
        "seed": 99,
    }
    converted = convert_entry(entry, seed=42)
    assert converted is not None
    assert converted.metadata["template_family"] == "turn_on_area"
    assert converted.metadata["phrasing_family"] == "polite"
    assert converted.metadata["seed"] == 99
