"""LFM2.5-230M training helpers using SaySo's OpenAI tool envelope."""

from __future__ import annotations

import json
from typing import Any

from .schema import (
    HOME_LLM_LABEL_MARKERS,
    TrainingExample,
    assert_openai_tool_envelope,
    assert_tools_subset_of_v1,
    contains_home_llm_label_markers,
    v1_openai_tools,
)

LFM_BASE_MODEL = "LiquidAI/LFM2.5-230M"


def lfm_tool_catalog() -> list[dict[str, Any]]:
    """Canonical v1 tools for LFM training examples."""
    return v1_openai_tools()


def prepare_lfm_example(example: TrainingExample) -> TrainingExample:
    """Ensure a converted example uses the locked v1 catalog and runtime envelope."""
    assert_tools_subset_of_v1(example.tools)
    for message in example.messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str) and contains_home_llm_label_markers(content):
            raise ValueError("assistant content must not use Home-LLM <tool_call> labels")
        for call in message.get("tool_calls") or []:
            if call.get("type") != "function":
                raise ValueError("tool_calls must use type:function")
            fn = call.get("function") or {}
            if not isinstance(fn.get("name"), str):
                raise ValueError("tool_calls must declare function.name")
            args = fn.get("arguments")
            if not isinstance(args, str):
                raise ValueError("LFM/SaySo labels must keep function.arguments as JSON strings")
            if contains_home_llm_label_markers(args):
                raise ValueError("tool call arguments must not use Home-LLM <tool_call> labels")
    return TrainingExample(
        messages=example.messages,
        tools=lfm_tool_catalog(),
        metadata=example.metadata,
    )


def lfm_jsonl_line(example: TrainingExample) -> str:
    """Serialize one LFM training record (SaySo runtime envelope, not Home-LLM labels)."""
    prepared = prepare_lfm_example(example)
    line = prepared.to_jsonl_line(view="lfm")
    if contains_home_llm_label_markers(line):
        raise ValueError("serialized example must not contain Home-LLM label markers")
    return line


def forbidden_home_llm_label_patterns() -> frozenset[str]:
    """Markers that must never appear in LFM fine-tuning labels."""
    return HOME_LLM_LABEL_MARKERS


def validate_lfm_config_text(config_text: str) -> None:
    """Reject Axolotl configs that embed Home-LLM ChatML tool-call rendering."""
    for marker in HOME_LLM_LABEL_MARKERS:
        if marker in config_text:
            raise ValueError(f"LFM config must not render Home-LLM labels ({marker})")


def summarize_tools(tools: list[dict[str, Any]]) -> list[str]:
    """Return sorted tool names for diagnostics."""
    names: list[str] = []
    for tool in tools:
        assert_openai_tool_envelope(tool)
        names.append(tool["function"]["name"])
    return sorted(names)
