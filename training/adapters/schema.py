"""SaySo training schema types and validation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[2]
V1_SCHEMA_ARTIFACT = REPO_ROOT / "schemas" / "sayso-tool-schema-v1.json"
TRAINING_V1_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sayso_tool_schema_v1.json"

# Legacy service-call argument keys that must be rejected.
LEGACY_ARGUMENT_KEYS: frozenset[str] = frozenset(
    {
        "entity_id",
        "target_device",
        "service",
        "service_name",
    }
)

# Legacy tool name prefixes/patterns.
LEGACY_TOOL_PREFIXES: tuple[str, ...] = (
    "light.",
    "switch.",
    "fan.",
    "lock.",
    "cover.",
    "climate.",
    "media_player.",
    "vacuum.",
    "timer.",
    "todo.",
)

CHATML_TOOL_CALL_MARKERS: frozenset[str] = frozenset(
    {
        "<tool_call>",
        "</tool_call>",
    }
)


@lru_cache(maxsize=1)
def load_v1_schema() -> dict[str, Any]:
    """Load the locked SaySo tool schema artifact (read-only source of truth)."""
    if not V1_SCHEMA_ARTIFACT.is_file():
        raise FileNotFoundError(f"Missing locked schema artifact: {V1_SCHEMA_ARTIFACT}")
    return json.loads(V1_SCHEMA_ARTIFACT.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_v1_tools() -> tuple[dict[str, Any], ...]:
    """Return immutable OpenAI-style tools from the locked v1 artifact."""
    schema = load_v1_schema()
    tools = schema.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError("Locked v1 schema must contain a non-empty tools list")
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError(f"Tool at index {index} must be type:function")
        fn = tool.get("function")
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            raise ValueError(f"Tool at index {index} must declare function.name")
    return tuple(tools)


def v1_tool_names() -> frozenset[str]:
    """Tool names declared in the locked v1 artifact."""
    return frozenset(tool["function"]["name"] for tool in load_v1_tools())


ALLOWED_HASS_TOOLS: frozenset[str] = v1_tool_names()


def v1_openai_tools() -> list[dict[str, Any]]:
    """Full v1 catalog in the OpenAI type:function envelope SaySo sends at runtime."""
    return [dict(tool) for tool in load_v1_tools()]


def v1_tool_by_name() -> dict[str, dict[str, Any]]:
    """Map tool name -> canonical OpenAI tool definition from v1."""
    return {tool["function"]["name"]: dict(tool) for tool in load_v1_tools()}


def assert_openai_tool_envelope(tool: dict[str, Any]) -> None:
    """Validate one tools[] entry matches the runtime llama.cpp envelope."""
    if tool.get("type") != "function":
        raise ValueError("tool entry must have type 'function'")
    fn = tool.get("function")
    if not isinstance(fn, dict):
        raise ValueError("tool entry must include function object")
    if not isinstance(fn.get("name"), str):
        raise ValueError("tool.function.name must be a string")
    params = fn.get("parameters")
    if not isinstance(params, dict):
        raise ValueError("tool.function.parameters must be an object")


def assert_tools_subset_of_v1(tools: list[dict[str, Any]]) -> None:
    """Ensure every tool name is declared in the locked v1 catalog."""
    allowed = ALLOWED_HASS_TOOLS
    for tool in tools:
        assert_openai_tool_envelope(tool)
        name = tool["function"]["name"]
        if name not in allowed:
            raise ValueError(f"tool {name!r} is not in locked v1 catalog")


def contains_chatml_tool_call_markers(text: str) -> bool:
    """Return True when text uses forbidden ChatML tool-call labels."""
    return any(marker in text for marker in CHATML_TOOL_CALL_MARKERS)


@dataclass(frozen=True, slots=True)
class RejectionStats:
    """Counts of rejected examples by reason."""

    counts: dict[str, int] = field(default_factory=dict)

    def record(self, reason: str) -> None:
        object.__setattr__(
            self,
            "counts",
            {**self.counts, reason: self.counts.get(reason, 0) + 1},
        )

    def merge(self, other: RejectionStats) -> RejectionStats:
        merged = dict(self.counts)
        for reason, count in other.counts.items():
            merged[reason] = merged.get(reason, 0) + count
        return RejectionStats(counts=merged)


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """One SaySo-compatible training record."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_jsonl_line(self, *, view: str = "sayso") -> str:
        """Serialize to JSONL.

        view="sayso" keeps OpenAI-style function.arguments as JSON strings
        (runtime / SaySo compatibility). view="lfm" is an alias of sayso.
        view="axolotl" parses argument strings into dicts so the FunctionGemma
        jinja template renders native key:value calls.
        """
        if view == "axolotl":
            messages = _axolotl_messages(self.messages)
        elif view in {"sayso", "lfm"}:
            messages = self.messages
        else:
            raise ValueError(f"unknown view: {view}")
        payload = {"messages": messages, "tools": self.tools}
        if self.metadata:
            payload["metadata"] = self.metadata
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _axolotl_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy with tool-call argument strings parsed to dicts."""
    out: list[dict[str, Any]] = []
    for message in messages:
        msg = dict(message)
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            calls: list[dict[str, Any]] = []
            for tc in msg["tool_calls"]:
                call = dict(tc)
                fn = dict(call.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    parsed = normalize_tool_arguments(args)
                    if parsed is not None:
                        fn["arguments"] = parsed
                call["function"] = fn
                calls.append(call)
            msg["tool_calls"] = calls
        out.append(msg)
    return out


def extract_text_content(content: Any) -> str:
    """Normalize message content from list or string payloads."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def tool_schema_map(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build name -> JSON Schema properties map from OpenAI-style tools."""
    result: dict[str, dict[str, Any]] = {}
    for tool in tools:
        fn = tool.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        params = fn.get("parameters")
        if isinstance(name, str) and isinstance(params, dict):
            result[name] = params
    return result


def allowed_properties(schema: dict[str, Any]) -> frozenset[str]:
    """Return property names declared on a tool parameters schema."""
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return frozenset()
    return frozenset(str(key) for key in properties)


def validate_tool_arguments(
    tool_name: str,
    args: dict[str, Any],
    schemas: dict[str, dict[str, Any]],
) -> str | None:
    """Return rejection reason when arguments fail schema or policy checks."""
    for key in args:
        if key in LEGACY_ARGUMENT_KEYS:
            return "legacy_argument_key"

    schema = schemas.get(tool_name)
    if schema is None:
        return "unknown_tool_schema"

    allowed = allowed_properties(schema)
    for key in args:
        if key not in allowed:
            return "extra_argument"

    required = schema.get("required") or []
    if isinstance(required, list):
        for key in required:
            if key not in args:
                return "missing_required_argument"

    try:
        validator = Draft202012Validator(schema)
        validator.validate(args)
    except jsonschema.ValidationError:
        return "schema_validation_failed"
    return None


def is_legacy_tool_name(name: str) -> bool:
    """Return True when the tool name looks like a legacy service call."""
    if name in LEGACY_TOOL_PREFIXES or "." in name:
        return True
    return any(name.startswith(prefix) for prefix in LEGACY_TOOL_PREFIXES)


def normalize_tool_arguments(args: Any) -> dict[str, Any] | None:
    """Parse tool arguments to a dict, or None when invalid."""
    if isinstance(args, dict):
        return dict(args)
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def shorten_response(text: str, *, max_words: int = 24) -> str:
    """Shorten verbose assistant confirmations for TTS-friendly training."""
    stripped = text.strip()
    if not stripped:
        return stripped
    words = stripped.split()
    if len(words) <= max_words:
        return stripped
    return " ".join(words[:max_words]).rstrip(".,;:") + "."
