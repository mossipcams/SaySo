"""OpenAI-shaped transcripts sent to llama.cpp.

Everything the model reads is built here: the running conversation, and the
synthetic assistant/tool pair that asks it to correct a call before anything
executes. Both use the same tool-call serialization, so a correction request is
indistinguishable in shape from a real turn.
"""

from __future__ import annotations

import json
from typing import Any

from homeassistant.components import conversation

from .client import ToolCall
from .schema import (
    ToolArgumentFailureCode,
    ToolArgumentValidationError,
    format_synthetic_validation_error,
)


def _tool_call_message(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Serialize one tool call the way llama.cpp expects to read it back."""
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, sort_keys=True),
        },
    }


def chat_log_to_messages(
    content: list[conversation.Content],
) -> list[dict[str, Any]]:
    """Convert Home Assistant chat log entries to llama.cpp messages."""
    messages: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, conversation.SystemContent):
            if item.content:
                messages.append({"role": "system", "content": item.content})
        elif isinstance(item, conversation.UserContent):
            messages.append({"role": "user", "content": item.content})
        elif isinstance(item, conversation.AssistantContent):
            message: dict[str, Any] = {
                "role": "assistant",
                "content": item.content or "",
            }
            if item.tool_calls:
                message["tool_calls"] = [
                    _tool_call_message(call.id, call.tool_name, call.tool_args)
                    for call in item.tool_calls
                ]
            messages.append(message)
        elif isinstance(item, conversation.ToolResultContent):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.tool_call_id,
                    "content": json.dumps(item.tool_result),
                }
            )
    return messages


def filtered_miss_failures(
    tool_calls: list[ToolCall],
) -> list[tuple[ToolCall, ToolArgumentValidationError]]:
    """Describe calls the active schema subset never offered.

    A filtered miss and a rejected argument are the same failure from the
    model's side — it was shown the wrong contract — so they are reported in
    one shape and share one correction budget.
    """
    return [
        (
            tool_call,
            ToolArgumentValidationError(
                code=ToolArgumentFailureCode.SCHEMA_MISMATCH,
                message=(
                    f"Tool {tool_call.name} is not available in the active schema subset"
                ),
                tool_name=tool_call.name,
            ),
        )
        for tool_call in tool_calls
    ]


def build_correction_messages(
    base_messages: list[dict[str, Any]],
    failures: list[tuple[ToolCall, ToolArgumentValidationError]],
    allowed_tools: set[str],
    fingerprint: str,
) -> list[dict[str, Any]]:
    """Append a synthetic assistant/tool transcript for one correction request."""
    allowed_tool_names = sorted(allowed_tools)
    messages = [
        *base_messages,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _tool_call_message(call.id, call.name, call.arguments)
                for call, _error in failures
            ],
        },
    ]
    messages.extend(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(
                format_synthetic_validation_error(
                    validation_error,
                    allowed_tools=allowed_tool_names,
                    fingerprint=fingerprint,
                )
            ),
        }
        for tool_call, validation_error in failures
    )
    return messages
