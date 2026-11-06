from __future__ import annotations

from typing import Any

_QUESTION_ENDINGS = ("?", ";", "？")


def _last_index(messages: list[dict[str, Any]], role: str, before: int) -> int | None:
    for index in range(before - 1, -1, -1):
        if messages[index]["role"] == role:
            return index
    return None


def _asks_question(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return isinstance(content, str) and content.strip().endswith(_QUESTION_ENDINGS)


def model_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    current = _last_index(messages, "user", len(messages))
    if current is None:
        return messages
    start = current
    previous_assistant = _last_index(messages, "assistant", current)
    if previous_assistant is not None and _asks_question(messages[previous_assistant]):
        previous_user = _last_index(messages, "user", previous_assistant)
        start = previous_assistant if previous_user is None else previous_user
    return [
        message
        for index, message in enumerate(messages)
        if index >= start or message["role"] == "system"
    ]
