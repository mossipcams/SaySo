from __future__ import annotations

import json
from typing import Any

from .completion import ToolCall
from .tool_contract import (
    ToolArgumentValidationError,
    format_synthetic_validation_error,
)


def _tool_call_message(
    call_id: str, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, sort_keys=True),
        },
    }


def build_correction_messages(
    base_messages: list[dict[str, Any]],
    failures: list[tuple[ToolCall, ToolArgumentValidationError]],
    allowed_tools: set[str],
    fingerprint: str,
) -> list[dict[str, Any]]:
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
