"""Canonical compiled tool schemas: the part of compilation that is pure JSON.

Home Assistant tool objects are converted to OpenAPI in :mod:`.schema`; from
the canonical source JSON onward everything here is deterministic JSON work.
The offline eval and the training generator compile their tool lists through
these same functions, and they import this module without Home Assistant.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from .exceptions import SaySoInvalidToolEnvelopeError

_FUNCTION_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


@dataclass(frozen=True, slots=True)
class CompiledToolSchema:
    """Compiled OpenAI tools and compatibility fingerprint for one model turn."""

    tools: tuple[dict[str, Any], ...]
    fingerprint: str


def normalize_schema(
    node: Any,
    *,
    name: str | None = None,
    top_level: bool = False,
) -> Any:
    """Recursively remove redundant OpenAPI metadata from compiled schemas."""
    if isinstance(node, list):
        return [normalize_schema(item) for item in node]

    if not isinstance(node, dict):
        return node

    normalized: dict[str, Any] = {}
    for key, value in node.items():
        if top_level and key == "$schema":
            continue
        if key == "properties" and isinstance(value, dict):
            normalized[key] = {
                prop_name: normalize_schema(value[prop_name], name=prop_name)
                for prop_name in value
            }
            continue
        if key == "function" and isinstance(value, dict):
            fn_name = value.get("name")
            fn_name_str = fn_name if isinstance(fn_name, str) else None
            normalized[key] = normalize_schema(value, name=fn_name_str)
            continue
        if key == "parameters" and isinstance(value, dict):
            normalized[key] = normalize_schema(value, top_level=True)
            continue
        normalized[key] = normalize_schema(value)

    if name is not None and normalized.get("title") == name:
        normalized.pop("title", None)

    description = normalized.get("description")
    if isinstance(description, str) and not description.strip():
        normalized.pop("description", None)

    return normalized


def canonicalize_schema(node: Any) -> Any:
    """Recursively sort mapping keys and required arrays for stable serialization."""
    if isinstance(node, list):
        return [canonicalize_schema(item) for item in node]

    if not isinstance(node, dict):
        return node

    canonical: dict[str, Any] = {}
    for key in sorted(node):
        value = node[key]
        if key == "required" and isinstance(value, list):
            canonical[key] = sorted(value)
        else:
            canonical[key] = canonicalize_schema(value)
    return canonical


def canonicalize_compiled_tools(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Canonicalize compiled tools and sort them by function name."""
    canonical_tools = [canonicalize_schema(tool) for tool in tools]
    return sorted(canonical_tools, key=lambda tool: tool["function"]["name"])


def emit_canonical_json(tools: list[dict[str, Any]]) -> bytes:
    """Emit byte-identical canonical JSON for compiled tools."""
    canonical_tools = canonicalize_compiled_tools(tools)
    return json.dumps(
        canonical_tools,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def schema_fingerprint(tools: list[dict[str, Any]]) -> str:
    """Return the SHA-256 fingerprint of the canonical compiled-tool JSON."""
    digest = hashlib.sha256(emit_canonical_json(tools)).hexdigest()
    return f"sha256:{digest}"


def validate_compiled_tool_envelope(
    tools: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> None:
    """Reject invalid outer tool envelopes before caching or transport."""
    seen_names: set[str] = set()
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise SaySoInvalidToolEnvelopeError(
                f"Tool entry at index {index} must be an object"
            )
        if tool.get("type") != "function":
            raise SaySoInvalidToolEnvelopeError(
                f"Tool entry at index {index} must have type 'function'"
            )

        function = tool.get("function")
        if not isinstance(function, dict):
            raise SaySoInvalidToolEnvelopeError(
                f"Tool entry at index {index} must include a function object"
            )

        name = function.get("name")
        if not isinstance(name, str) or not _FUNCTION_NAME_RE.fullmatch(name):
            raise SaySoInvalidToolEnvelopeError(
                f"Tool entry at index {index} has an invalid function name"
            )
        if name in seen_names:
            raise SaySoInvalidToolEnvelopeError(
                f"Duplicate function name {name!r} in compiled tool envelope"
            )
        seen_names.add(name)

        parameters = function.get("parameters")
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise SaySoInvalidToolEnvelopeError(
                f"Tool {name!r} must have parameters with type 'object'"
            )

        try:
            json.dumps(tool, ensure_ascii=False)
        except TypeError as err:
            raise SaySoInvalidToolEnvelopeError(
                f"Tool {name!r} is not JSON-serializable"
            ) from err


def function_envelope(
    name: str, parameters: Any, description: str | None
) -> dict[str, Any]:
    """Wrap compiled parameters in the canonical OpenAI function envelope.

    The single place a tool takes its wire shape, so compiling from a live HA
    tool and rebuilding from cached source JSON cannot drift apart.
    """
    tool_spec: dict[str, Any] = {
        "name": name,
        "parameters": normalize_schema(parameters, top_level=True),
    }
    if description:
        tool_spec["description"] = description
    return canonicalize_schema(
        normalize_schema({"type": "function", "function": tool_spec}, top_level=True)
    )


def build_compiled_tools_from_source(
    source_json: str,
) -> tuple[dict[str, Any], ...]:
    """Normalize and canonicalize compiled tools from canonical source JSON."""
    compiled = canonicalize_compiled_tools(
        [
            function_envelope(
                entry["name"], entry["parameters"], entry.get("description")
            )
            for entry in json.loads(source_json)
        ]
    )
    validate_compiled_tool_envelope(compiled)
    return tuple(compiled)


def tool_source_json(entries: list[dict[str, Any]]) -> str:
    """Canonical source JSON for ``name``/``description``/``parameters`` entries."""
    return json.dumps(
        sorted(entries, key=lambda entry: entry["name"]),
        separators=(",", ":"),
        ensure_ascii=False,
        sort_keys=True,
    )


def compile_source_tools(entries: list[dict[str, Any]]) -> CompiledToolSchema:
    """Compile tool source entries exactly as live Home Assistant tools compile."""
    tools = build_compiled_tools_from_source(tool_source_json(entries))
    return CompiledToolSchema(tools=tools, fingerprint=schema_fingerprint(list(tools)))
