"""Parse llama.cpp OpenAI-compatible chat completion responses."""

from __future__ import annotations

import json
from typing import Any


class LlamaCppParseError(ValueError):
    """Raised when a llama.cpp response cannot be parsed."""


def parse_chat_completion(body: dict[str, Any]) -> dict[str, Any]:
    """Parse one /v1/chat/completions response into messages-compatible output."""
    if not isinstance(body, dict):
        raise LlamaCppParseError("response must be a JSON object")

    if "error" in body:
        error = body["error"]
        if isinstance(error, dict):
            message = error.get("message", "unknown error")
        else:
            message = str(error)
        raise LlamaCppParseError(message)

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LlamaCppParseError("missing choices")

    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise LlamaCppParseError("missing message")

    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise LlamaCppParseError("invalid content type")

    tool_calls: list[dict[str, Any]] = []
    raw_calls = message.get("tool_calls")
    if raw_calls is not None:
        if not isinstance(raw_calls, list):
            raise LlamaCppParseError("invalid tool_calls")
        for item in raw_calls:
            if not isinstance(item, dict):
                raise LlamaCppParseError("invalid tool call entry")
            fn = item.get("function")
            if not isinstance(fn, dict):
                raise LlamaCppParseError("invalid function entry")
            name = fn.get("name")
            arguments = fn.get("arguments")
            if not isinstance(name, str) or not name:
                raise LlamaCppParseError("missing tool name")
            if isinstance(arguments, dict):
                args_str = json.dumps(arguments, sort_keys=True)
            elif isinstance(arguments, str):
                args_str = arguments
            else:
                raise LlamaCppParseError("invalid arguments")
            tool_calls.append(
                {
                    "id": item.get("id") or f"call_{len(tool_calls)+1}",
                    "type": "function",
                    "function": {"name": name, "arguments": args_str},
                }
            )

    if content is None and not tool_calls:
        raise LlamaCppParseError("empty assistant message")

    result: dict[str, Any] = {"role": "assistant"}
    if tool_calls:
        result["content"] = ""
        result["tool_calls"] = tool_calls
    else:
        result["content"] = content or ""

    usage = body.get("usage")
    prompt_tokens = None
    if isinstance(usage, dict):
        pt = usage.get("prompt_tokens")
        if isinstance(pt, int):
            prompt_tokens = pt

    return {
        "message": result,
        "prompt_tokens": prompt_tokens,
        "model": body.get("model"),
    }
