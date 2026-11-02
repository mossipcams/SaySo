
from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .area_context import _AREA_BLOCK_MARKER
from .exceptions import SaySoInvalidResponseError
from .lfm_parse import LfmPythonParseError, parse_lfm_python_tool_calls

@dataclass(frozen=True, slots=True)
class ToolCall:

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ChatCompletionResult:

    content: str | None
    tool_calls: list[ToolCall]
    request_payload: dict[str, Any] | None = None
    request_bytes: int | None = None
    prompt_tokens: int | None = None
    prefix_hash: str | None = None
    reused_tokens: int | None = None


def parse_tool_calls(
    raw: Any,
    *,
    subject: str,
    mint_id: Callable[[], str] | None = None,
) -> list[ToolCall]:
    invalid = SaySoInvalidResponseError(f"{subject} returned invalid tool calls")
    if raw is None:
        return []
    if not isinstance(raw, list):
        if mint_id is None:
            raise invalid
        return []
    if not raw:
        return []

    calls: list[ToolCall] = []
    for item in raw:
        if not isinstance(item, dict):
            raise invalid
        function = item.get("function")
        if not isinstance(function, dict):
            raise invalid
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise invalid

        call_id = item.get("id")
        if not (isinstance(call_id, str) and call_id):
            if mint_id is None:
                raise invalid
            call_id = mint_id()

        calls.append(
            ToolCall(
                id=call_id,
                name=name,
                arguments=_decode_arguments(function.get("arguments"), subject),
            )
        )
    return calls


def _decode_arguments(raw: Any, subject: str) -> dict[str, Any]:
    invalid = SaySoInvalidResponseError(f"{subject} returned invalid tool call arguments")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as err:
            raise invalid from err
    if not isinstance(raw, dict):
        raise invalid
    return raw


def parse_choice_message(raw: Any, subject: str) -> tuple[str | None, dict[str, Any]]:
    choices = raw.get("choices") if isinstance(raw, dict) else None
    if not isinstance(choices, list) or not choices:
        raise SaySoInvalidResponseError(f"{subject} returned no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise SaySoInvalidResponseError(f"{subject} returned no choices")
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise SaySoInvalidResponseError(f"{subject} returned invalid content")
    return content, message


def prompt_tokens_of(raw: Any) -> int | None:
    usage = raw.get("usage") if isinstance(raw, dict) else None
    tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    return tokens if isinstance(tokens, int) else None


def cache_trace_of(raw: dict[str, Any]) -> tuple[str | None, int | None]:
    cache = raw.get("sayso_cache")
    if not isinstance(cache, dict):
        return None, None
    prefix = cache.get("prefix_hash")
    reused = cache.get("reused_tokens")
    return (
        prefix if isinstance(prefix, str) else None,
        reused if isinstance(reused, int) and not isinstance(reused, bool) else None,
    )


def prefix_hash(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
) -> str:
    system = ""
    if messages and messages[0].get("role") == "system":
        system = messages[0].get("content") or ""
    index = system.rfind(_AREA_BLOCK_MARKER)
    stable = system[:index] if index >= 0 else system
    digest = hashlib.sha256((stable + json.dumps(tools, sort_keys=True)).encode())
    return digest.hexdigest()[:12]


_TOOL_CALL_START = "<|tool_call_start|>"
_TOOL_CALL_END = "<|tool_call_end|>"

_SUBJECT = "Local inference"


def call_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


_CALL_SHAPED = re.compile(r"^\[?\s*[A-Za-z_][A-Za-z0-9_]*\(")


def extract_tool_calls(content: str) -> tuple[str | None, list[ToolCall]]:
    if not content:
        return None, []

    body = content
    if _TOOL_CALL_START in body:
        prefix, _, rest = body.partition(_TOOL_CALL_START)
        body, _, suffix = rest.partition(_TOOL_CALL_END)
        leftover = f"{prefix}{suffix}".strip()
    else:
        stripped = body.strip()
        bracketed = stripped.startswith("[") and stripped.endswith("]")
        if not (bracketed or _CALL_SHAPED.match(stripped)):
            return content, []
        body, leftover = stripped, ""

    try:
        parsed = parse_lfm_python_tool_calls(body)
    except LfmPythonParseError as err:
        raise SaySoInvalidResponseError(
            f"Could not parse model tool call {body!r}: {err}"
        ) from err

    calls = [
        ToolCall(
            id=call_id(),
            name=call["name"],
            arguments=call["arguments"],
        )
        for call in parsed
    ]
    return (leftover or None), calls


def parse_completion_result(raw: Any) -> ChatCompletionResult:
    if not isinstance(raw, dict):
        raise SaySoInvalidResponseError("Local inference returned no result")

    content, message = parse_choice_message(raw, _SUBJECT)

    tool_calls = parse_tool_calls(
        message.get("tool_calls"), subject=_SUBJECT, mint_id=call_id
    )
    if tool_calls:
        text: str | None = content
    else:
        text, tool_calls = extract_tool_calls(content or "")

    if text is None and not tool_calls:
        raise SaySoInvalidResponseError(
            "Local inference returned neither content nor tool calls"
        )

    prefix, reused = cache_trace_of(raw)
    return ChatCompletionResult(
        content=text,
        tool_calls=tool_calls,
        prompt_tokens=prompt_tokens_of(raw),
        prefix_hash=prefix,
        reused_tokens=reused,
    )
