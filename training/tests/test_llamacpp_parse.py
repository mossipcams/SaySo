"""Tests for production llama.cpp completion parsing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sayso_contract import completion, exceptions


def test_parse_text_response() -> None:
    body = {"choices": [{"message": {"role": "assistant", "content": "Done."}}]}
    parsed = completion.parse_completion_result(body)
    assert parsed.content == "Done."
    assert parsed.tool_calls == []


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
                                "name": "intent__HassTurnOff",
                                "arguments": '{"name": "Kitchen"}',
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 128},
    }
    parsed = completion.parse_completion_result(body)
    assert parsed.tool_calls[0].name == "intent__HassTurnOff"
    assert parsed.tool_calls[0].arguments == {"name": "Kitchen"}


def test_parse_error_response() -> None:
    with pytest.raises(exceptions.SaySoInvalidResponseError):
        completion.parse_completion_result({"error": {"message": "boom"}})
