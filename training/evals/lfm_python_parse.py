"""Parse LFM Base ChatML Python-style tool calls with apostrophe-safe quoting."""

from __future__ import annotations

import re
from typing import Any


class LfmPythonParseError(ValueError):
    """Raised when an LFM Python-style tool call cannot be parsed."""


_CLOSER_AFTER_QUOTE = frozenset({",", ")", "]", "}"})


def _parse_single_quoted_string(text: str, start: int) -> tuple[str, int]:
    """Parse a single-quoted string starting at ``start`` (the opening quote index)."""
    if start >= len(text) or text[start] != "'":
        raise LfmPythonParseError("expected opening single quote")
    index = start + 1
    chars: list[str] = []
    while index < len(text):
        char = text[index]
        if char != "'":
            chars.append(char)
            index += 1
            continue
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if not next_char or next_char in _CLOSER_AFTER_QUOTE:
            return "".join(chars), index + 1
        chars.append("'")
        index += 1
    raise LfmPythonParseError("unterminated single-quoted string")


def _skip_ws(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _parse_value(text: str, index: int) -> tuple[Any, int]:
    index = _skip_ws(text, index)
    if index >= len(text):
        raise LfmPythonParseError("unexpected end of value")
    if text[index] == "'":
        return _parse_single_quoted_string(text, index)
    if text[index] == "[":
        return _parse_list(text, index)
    if text[index] in "-0123456789":
        end = index
        while end < len(text) and text[end] in "-0123456789":
            end += 1
        return int(text[index:end]), end
    raise LfmPythonParseError(f"unsupported value at {index!r}")


def _parse_list(text: str, index: int) -> tuple[list[Any], int]:
    if text[index] != "[":
        raise LfmPythonParseError("expected '['")
    index += 1
    items: list[Any] = []
    index = _skip_ws(text, index)
    if index < len(text) and text[index] == "]":
        return items, index + 1
    while True:
        value, index = _parse_value(text, index)
        items.append(value)
        index = _skip_ws(text, index)
        if index >= len(text):
            raise LfmPythonParseError("unterminated list")
        if text[index] == "]":
            return items, index + 1
        if text[index] != ",":
            raise LfmPythonParseError("expected ',' or ']' in list")
        index = _skip_ws(text, index + 1)


def _parse_argument_list(text: str, index: int) -> tuple[dict[str, Any], int]:
    index = _skip_ws(text, index)
    if index < len(text) and text[index] == ")":
        return {}, index + 1
    arguments: dict[str, Any] = {}
    while True:
        match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)=", text[index:])
        if not match:
            raise LfmPythonParseError("expected keyword argument")
        key = match.group(1)
        index += match.end()
        value, index = _parse_value(text, index)
        arguments[key] = value
        index = _skip_ws(text, index)
        if index >= len(text):
            raise LfmPythonParseError("unterminated argument list")
        if text[index] == ")":
            return arguments, index + 1
        if text[index] != ",":
            raise LfmPythonParseError("expected ',' or ')' in argument list")
        index = _skip_ws(text, index + 1)


def parse_lfm_python_tool_call(text: str) -> dict[str, Any]:
    """Parse one ``ToolName(key='value')`` call into ``{name, arguments}``."""
    stripped = text.strip()
    match = re.match(r"([A-Za-z][A-Za-z0-9_]*)\(", stripped)
    if not match:
        raise LfmPythonParseError("expected tool name")
    name = match.group(1)
    index = match.end()
    arguments, end = _parse_argument_list(stripped, index)
    end = _skip_ws(stripped, end)
    if end != len(stripped):
        raise LfmPythonParseError("trailing content after tool call")
    return {"name": name, "arguments": arguments}


def parse_lfm_python_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse a bracketed or bare sequence of Python-style tool calls."""
    stripped = text.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        stripped = stripped[1:-1].strip()
    if not stripped:
        return []
    calls: list[dict[str, Any]] = []
    index = 0
    while index < len(stripped):
        index = _skip_ws(stripped, index)
        if index >= len(stripped):
            break
        match = re.match(r"[A-Za-z][A-Za-z0-9_]*\(", stripped[index:])
        if not match:
            raise LfmPythonParseError("expected tool call")
        depth = 0
        end = index
        while end < len(stripped):
            char = stripped[end]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    end += 1
                    break
            end += 1
        if depth != 0:
            raise LfmPythonParseError("unbalanced parentheses")
        calls.append(parse_lfm_python_tool_call(stripped[index:end]))
        index = _skip_ws(stripped, end)
        if index < len(stripped):
            if stripped[index] != ",":
                raise LfmPythonParseError("expected ',' between tool calls")
            index += 1
    return calls
