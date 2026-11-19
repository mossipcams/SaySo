from __future__ import annotations

import json
from typing import Any

from homeassistant.components import conversation

from .boundary import tool_result_data
from .core.correction import (
    _tool_call_message,
    build_correction_messages,
)


def chat_log_to_messages(
    content: list[conversation.Content],
) -> list[dict[str, Any]]:
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
                    "content": json.dumps(tool_result_data(item)),
                }
            )
    return messages

