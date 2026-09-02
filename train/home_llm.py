"""Parse Home-LLM synthetic JSONL rows (messages + tools)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

DEVICE_LINE = re.compile(
    r"^(?P<entity_id>[a-z][a-z0-9_]*\.[a-z0-9_]+)\s+'(?P<friendly_name>[^']+)'\s*=\s*(?P<state>.+)$"
)

DEVICES_SECTION_MARKERS = (
    "Devices:",
    "Geräte:",
    "Appareils:",
    "Dispositivos:",
    "Urządzenia:",
)


@dataclass(frozen=True)
class ParsedDevice:
    entity_id: str
    friendly_name: str
    domain: str
    state_value: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class HomeLlmRow:
    user_text: str
    devices: tuple[ParsedDevice, ...]
    tool_calls: tuple[ToolCallRecord, ...]
    declared_tool_names: frozenset[str]
    has_trainable_assistant: bool


def parse_home_llm_row(raw: dict[str, Any]) -> HomeLlmRow | None:
    """Extract structured fields from one Home-LLM JSONL object."""
    messages = raw.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    declared_tool_names = _declared_tool_names(raw.get("tools"))
    devices = _parse_devices_from_messages(messages)
    user_text = _extract_user_text(messages)
    if not user_text:
        return None

    tool_calls, has_trainable = _extract_tool_calls(messages)
    return HomeLlmRow(
        user_text=user_text,
        devices=tuple(devices),
        tool_calls=tuple(tool_calls),
        declared_tool_names=declared_tool_names,
        has_trainable_assistant=has_trainable,
    )


def _declared_tool_names(tools: object) -> frozenset[str]:
    if not isinstance(tools, list):
        return frozenset()
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            names.add(fn["name"])
        elif isinstance(tool.get("name"), str):
            names.add(tool["name"])
    return frozenset(names)


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def _extract_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        text = _message_text(message).strip()
        if not text:
            continue
        if "User instruction" in text or "User instruction:" in text:
            _, _, tail = text.partition("User instruction")
            tail = tail.lstrip(" :")
            if tail.strip():
                return tail.strip().lower()
        if "\n" in text and any(marker in text for marker in DEVICES_SECTION_MARKERS):
            continue
        return text.lower()
    return ""


def _parse_devices_from_messages(messages: list[dict[str, Any]]) -> list[ParsedDevice]:
    devices: list[ParsedDevice] = []
    seen: set[str] = set()
    for message in messages:
        if message.get("role") not in {"system", "user"}:
            continue
        text = _message_text(message)
        for line in text.splitlines():
            parsed = _parse_device_line(line.strip())
            if parsed is None or parsed.entity_id in seen:
                continue
            seen.add(parsed.entity_id)
            devices.append(parsed)
    return devices


def _devices_section(text: str) -> str:
    for marker in DEVICES_SECTION_MARKERS:
        if marker in text:
            return text.split(marker, 1)[1]
    return text


def _parse_device_line(line: str) -> ParsedDevice | None:
    match = DEVICE_LINE.match(line)
    if match is None:
        return None
    entity_id = match.group("entity_id")
    domain = entity_id.split(".", 1)[0]
    state_raw = match.group("state")
    state_value, attributes = _parse_state(state_raw, domain)
    return ParsedDevice(
        entity_id=entity_id,
        friendly_name=match.group("friendly_name"),
        domain=domain,
        state_value=state_value,
        attributes=attributes,
    )


def _parse_state(state_raw: str, domain: str) -> tuple[str, dict[str, Any]]:
    parts = [part.strip() for part in state_raw.split(";") if part.strip()]
    if not parts:
        return "unknown", {}
    value = parts[0].lower()
    attributes: dict[str, Any] = {}
    for part in parts[1:]:
        if part.endswith("%") and part[:-1].isdigit():
            attributes["brightness"] = int(part[:-1])
            continue
        if part.endswith("F") and part[:-1].replace(".", "", 1).isdigit():
            attributes["temperature"] = float(part[:-1])
            continue
        if part.endswith("C") and part[:-1].replace(".", "", 1).isdigit():
            attributes["temperature"] = float(part[:-1])
            continue
        if domain == "climate" and part in {"heat", "cool", "auto", "off", "fan_only", "heat_cool"}:
            attributes["hvac_mode"] = part
            continue
        attributes.setdefault("extra", []).append(part)
    return value, attributes


def _extract_tool_calls(
    messages: list[dict[str, Any]],
) -> tuple[list[ToolCallRecord], bool]:
    tool_calls: list[ToolCallRecord] = []
    has_trainable = False
    for message in messages:
        if message.get("role") != "assistant":
            continue
        if message.get("train_on_turn", True):
            has_trainable = True
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            fn = call.get("function")
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if not isinstance(name, str):
                continue
            args_raw = fn.get("arguments", "{}")
            if isinstance(args_raw, str):
                try:
                    arguments = json.loads(args_raw)
                except json.JSONDecodeError:
                    arguments = {}
            elif isinstance(args_raw, dict):
                arguments = args_raw
            else:
                arguments = {}
            tool_calls.append(ToolCallRecord(name=name, arguments=arguments))
    return tool_calls, has_trainable
