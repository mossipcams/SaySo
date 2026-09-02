"""Tests for llama.cpp response parsing."""

from __future__ import annotations

import pytest

from evals.llamacpp import LlamaCppParseError, parse_chat_completion


def test_parse_text_response() -> None:
    body = {"choices": [{"message": {"role": "assistant", "content": "Done."}}]}
    parsed = parse_chat_completion(body)
    assert parsed["message"]["content"] == "Done."
    assert "tool_calls" not in parsed["message"]


def test_parse_tool_calls_response() -> None:
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "HassTurnOff",
                                "arguments": '{"name": "Kitchen"}',
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 128},
    }
    parsed = parse_chat_completion(body)
    assert parsed["message"]["content"] == ""
    assert parsed["message"]["tool_calls"][0]["function"]["name"] == "HassTurnOff"
    assert parsed["prompt_tokens"] == 128


def test_parse_error_response() -> None:
    with pytest.raises(LlamaCppParseError):
        parse_chat_completion({"error": {"message": "boom"}})
