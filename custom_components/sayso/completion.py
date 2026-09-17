"""Turn a completion body into text and tool calls.

This is how SaySo reads model output: structured ``tool_calls`` when the
backend produced them, LFM2's native Python-style calls otherwise, and a closed
failure for anything that looks like a call but does not parse. The offline
eval parses with these same functions and imports this module without Home
Assistant.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .exceptions import SaySoInvalidResponseError
from .lfm_parse import LfmPythonParseError, parse_lfm_python_tool_calls

@dataclass(frozen=True, slots=True)
class ToolCall:
    """An OpenAI-style tool call from llama.cpp."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ChatCompletionResult:
    """Parsed assistant output from a chat completion."""

    content: str | None
    tool_calls: list[ToolCall]
    request_payload: dict[str, Any] | None = None
    request_bytes: int | None = None
    prompt_tokens: int | None = None


def parse_tool_calls(
    raw: Any,
    *,
    subject: str,
    mint_id: Callable[[], str] | None = None,
) -> list[ToolCall]:
    """Parse OpenAI-shaped ``tool_calls`` from any SaySo inference backend.

    ``mint_id`` supplies an id when the backend omits one. The HTTP client
    passes none, which also makes parsing strict: a server that sends a
    non-list ``tool_calls`` or an unlabelled call is not speaking the protocol,
    and that must fail closed rather than be read as "no tool calls" and let
    the accompanying text be spoken as if an action ran. The embedded backend
    stays lenient and falls back to parsing LFM2's native text format.
    """
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
    """Accept an argument object, or the JSON string llama.cpp sends instead."""
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
    """Return the first choice's content and message from a completion body."""
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
    """Return ``usage.prompt_tokens`` when the backend reported it."""
    usage = raw.get("usage") if isinstance(raw, dict) else None
    tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    return tokens if isinstance(tokens, int) else None


# LFM2 wraps tool calls in these markers. llama-cpp-python ships libllama only,
# not llama.cpp's common/chat.cpp, so the server-side parser that llama-server
# applies with --jinja is not available and SaySo strips them itself.
_TOOL_CALL_START = "<|tool_call_start|>"
_TOOL_CALL_END = "<|tool_call_end|>"

# Prefix for the errors this backend raises, so a trace says which one failed.
_SUBJECT = "Local inference"


def call_id() -> str:
    """Mint a tool-call id for a backend that does not supply one."""
    return f"call_{uuid.uuid4().hex[:8]}"


_CALL_SHAPED = re.compile(r"^\[?\s*[A-Za-z_][A-Za-z0-9_]*\(")


def extract_tool_calls(content: str) -> tuple[str | None, list[ToolCall]]:
    """Split assistant text into leftover prose and structured tool calls.

    Returns ``(text, [])`` when the model answered in prose. Malformed tool-call
    syntax raises, because a half-understood action must fail closed rather than
    execute something approximate.
    """
    if not content:
        return None, []

    body = content
    if _TOOL_CALL_START in body:
        prefix, _, rest = body.partition(_TOOL_CALL_START)
        body, _, suffix = rest.partition(_TOOL_CALL_END)
        leftover = f"{prefix}{suffix}".strip()
    else:
        stripped = body.strip()
        # A bare bracketed call list is the same payload without the markers.
        # Text that opens like a call but is cut off ("[HassTurnOn(name='x'")
        # is a broken call, not prose to speak.
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
    """Convert llama-cpp-python output into SaySo's transport-neutral result."""
    if not isinstance(raw, dict):
        raise SaySoInvalidResponseError("Local inference returned no result")

    content, message = parse_choice_message(raw, _SUBJECT)

    # Honour structured tool_calls when a handler produced them, and fall back
    # to parsing LFM2's native text format otherwise. llama-cpp-python omits
    # call ids, so one is minted here rather than failing the turn.
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

    return ChatCompletionResult(
        content=text,
        tool_calls=tool_calls,
        prompt_tokens=prompt_tokens_of(raw),
    )
