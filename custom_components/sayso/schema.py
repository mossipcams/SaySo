"""Compile Home Assistant tool schemas for llama.cpp."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from collections.abc import Callable
from typing import Any, Literal

import voluptuous as vol
from homeassistant.helpers import llm
from voluptuous_openapi import convert

from .exceptions import SaySoInvalidToolEnvelopeError

COMPILE_CACHE_MAXSIZE = 32

_FUNCTION_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


@dataclass(frozen=True, slots=True)
class CompiledToolSchema:
    """Compiled OpenAI tools and compatibility fingerprint for one model turn."""

    tools: tuple[dict[str, Any], ...]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class ToolRoutingMetadata:
    """Domain applicability metadata extracted from one HA tool."""

    declared_domains: frozenset[str] | None = None
    retain_always: bool = False


def _vol_in_allowed_values(validator: Any) -> frozenset[str] | None:
    """Return allowed values when a validator is or contains vol.In."""
    if isinstance(validator, vol.In):
        container = validator.container
        if isinstance(container, dict):
            return frozenset(str(key) for key in container)
        return frozenset(str(value) for value in container)

    if isinstance(validator, vol.All):
        for sub_validator in validator.validators:
            allowed = _vol_in_allowed_values(sub_validator)
            if allowed is not None:
                return allowed

    if isinstance(validator, list):
        for sub_validator in validator:
            allowed = _vol_in_allowed_values(sub_validator)
            if allowed is not None:
                return allowed

    return None


def _declared_domains_from_parameters(parameters: vol.Schema) -> frozenset[str] | None:
    """Read explicit domain restrictions from a tool's Voluptuous schema."""
    for marker, validator in parameters.schema.items():
        if _schema_marker_name(marker) != "domain":
            continue
        allowed = _vol_in_allowed_values(validator)
        if allowed is not None:
            return allowed
    return None


def _unwrap_source_tool(tool: llm.Tool) -> llm.Tool:
    """Return the inner HA tool for namespaced wrappers."""
    while isinstance(tool, llm.NamespacedTool):
        tool = tool.tool
    return tool


def _is_query_tool(tool: llm.Tool) -> bool:
    """Return True for HA query/context tools that must always remain available."""
    module = type(tool).__module__
    class_name = type(tool).__name__
    if class_name == "GetLiveContextTool" and module.endswith("homeassistant.llm"):
        return True
    if class_name == "GetDateTimeTool" and module.endswith("llm.llm"):
        return True
    if class_name == "TodoGetItemsTool" and module.endswith("todo.llm"):
        return True
    return False


def extract_tool_routing_metadata(tool: llm.Tool) -> ToolRoutingMetadata:
    """Extract domain metadata from an HA tool without inferring from its name."""
    source = _unwrap_source_tool(tool)

    if _is_query_tool(source):
        return ToolRoutingMetadata(retain_always=True)

    if isinstance(source, llm.ActionTool):
        domain = source._domain
        if domain == "script":
            return ToolRoutingMetadata(retain_always=True)
        return ToolRoutingMetadata(declared_domains=frozenset({domain}))

    declared_domains = _declared_domains_from_parameters(source.parameters)
    return ToolRoutingMetadata(declared_domains=declared_domains)


def clear_compile_cache() -> None:
    """Clear the bounded compile cache. Intended for tests."""
    _cached_compile_tools.cache_clear()


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


def compile_parameters(
    schema: Any,
    *,
    custom_serializer: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Compile a Voluptuous schema to OpenAPI parameters."""
    normalized = normalize_schema(
        convert(schema, custom_serializer=custom_serializer),
        top_level=True,
    )
    return canonicalize_schema(normalized)


def compile_tool(
    tool: llm.Tool,
    *,
    custom_serializer: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Compile one HA tool to an OpenAI-compatible function definition."""
    tool_spec: dict[str, Any] = {
        "name": tool.name,
        "parameters": compile_parameters(
            tool.parameters,
            custom_serializer=custom_serializer,
        ),
    }
    if tool.description:
        tool_spec["description"] = tool.description
    normalized = normalize_schema(
        {"type": "function", "function": tool_spec},
        top_level=True,
    )
    compiled = canonicalize_schema(normalized)
    validate_compiled_tool_envelope((compiled,))
    return compiled


def _emit_tool_source_entry(
    tool: llm.Tool,
    *,
    custom_serializer: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Return one tool's canonical source payload for cache keys."""
    converted = convert(
        tool.parameters,
        custom_serializer=custom_serializer,
    )
    return {
        "description": tool.description or "",
        "name": tool.name,
        "parameters": canonicalize_schema(converted),
    }


def emit_tools_source_json(
    tools: list[llm.Tool],
    *,
    custom_serializer: Callable[[Any], Any] | None = None,
) -> str:
    """Emit canonical source JSON for a tool list."""
    entries = [
        _emit_tool_source_entry(tool, custom_serializer=custom_serializer)
        for tool in tools
    ]
    entries.sort(key=lambda entry: entry["name"])
    return json.dumps(
        entries,
        separators=(",", ":"),
        ensure_ascii=False,
        sort_keys=True,
    )


def _build_compiled_tools_from_source(
    source_json: str,
) -> tuple[dict[str, Any], ...]:
    """Normalize and canonicalize compiled tools from canonical source JSON."""
    entries: list[dict[str, Any]] = json.loads(source_json)
    compiled: list[dict[str, Any]] = []
    for entry in entries:
        tool_spec: dict[str, Any] = {
            "name": entry["name"],
            "parameters": normalize_schema(entry["parameters"], top_level=True),
        }
        description = entry.get("description")
        if isinstance(description, str) and description:
            tool_spec["description"] = description
        normalized = normalize_schema(
            {"type": "function", "function": tool_spec},
            top_level=True,
        )
        compiled.append(canonicalize_schema(normalized))
    compiled_tools = canonicalize_compiled_tools(compiled)
    validate_compiled_tool_envelope(compiled_tools)
    return tuple(compiled_tools)


@lru_cache(maxsize=COMPILE_CACHE_MAXSIZE)
def _cached_compile_tools(source_json: str) -> tuple[dict[str, Any], ...]:
    """Cache normalized compilation keyed by canonical source JSON."""
    return _build_compiled_tools_from_source(source_json)


def compile_tools(
    tools: list[llm.Tool],
    *,
    custom_serializer: Callable[[Any], Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Compile HA tools to OpenAI-compatible function definitions."""
    source_json = emit_tools_source_json(
        tools,
        custom_serializer=custom_serializer,
    )
    return _cached_compile_tools(source_json)


class ToolArgumentFailureCode(StrEnum):
    """Stable codes for tool-argument validation failures."""

    SCHEMA_MISMATCH = "schema_mismatch"
    INVALID_ARGUMENTS = "invalid_arguments"


@dataclass(frozen=True, slots=True)
class ToolArgumentValidationError:
    """Validation failure for one tool call's arguments."""

    code: ToolArgumentFailureCode
    message: str
    tool_name: str


def build_tool_map(tools: list[llm.Tool]) -> dict[str, llm.Tool]:
    """Map tool name to the HA tool definition."""
    return {tool.name: tool for tool in tools}


def _schema_marker_name(marker: Any) -> str | None:
    """Return the argument name represented by a Voluptuous schema marker."""
    if isinstance(marker, (vol.Required, vol.Optional)):
        schema = marker.schema
        return schema if isinstance(schema, str) else None
    if isinstance(marker, str):
        return marker
    if isinstance(marker, vol.Any):
        for sub_marker in marker.validators:
            name = _schema_marker_name(sub_marker)
            if name is not None:
                return name
    return None


def _collect_allowed_argument_names(schema: vol.Schema) -> set[str]:
    """Return every top-level argument name accepted by a Voluptuous schema."""
    allowed: set[str] = set()
    for marker in schema.schema:
        if isinstance(marker, vol.Any):
            for sub_marker in marker.validators:
                name = _schema_marker_name(sub_marker)
                if name is not None:
                    allowed.add(name)
            continue
        name = _schema_marker_name(marker)
        if name is not None:
            allowed.add(name)
    return allowed


def _classify_voluptuous_error(error: vol.Invalid) -> ToolArgumentFailureCode:
    """Map a Voluptuous error to schema mismatch or invalid arguments."""
    if isinstance(error, vol.MultipleInvalid):
        codes = {_classify_voluptuous_error(sub_error) for sub_error in error.errors}
        if ToolArgumentFailureCode.INVALID_ARGUMENTS in codes:
            return ToolArgumentFailureCode.INVALID_ARGUMENTS
        return ToolArgumentFailureCode.SCHEMA_MISMATCH

    message = str(error).lower()
    if "extra keys not allowed" in message:
        return ToolArgumentFailureCode.SCHEMA_MISMATCH
    if "required key not provided" in message:
        return ToolArgumentFailureCode.SCHEMA_MISMATCH
    return ToolArgumentFailureCode.INVALID_ARGUMENTS


def validate_tool_arguments(
    tool: llm.Tool,
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], Literal[None]] | tuple[None, ToolArgumentValidationError]:
    """Validate and normalize tool arguments against the HA Voluptuous schema."""
    allowed_names = _collect_allowed_argument_names(tool.parameters)
    unexpected = sorted(set(arguments) - allowed_names)
    if unexpected:
        return None, ToolArgumentValidationError(
            code=ToolArgumentFailureCode.SCHEMA_MISMATCH,
            message=f"Unexpected argument(s): {unexpected}",
            tool_name=tool.name,
        )

    try:
        normalized = tool.parameters(arguments)
    except vol.Invalid as err:
        return None, ToolArgumentValidationError(
            code=_classify_voluptuous_error(err),
            message=str(err),
            tool_name=tool.name,
        )

    return normalized, None


def format_synthetic_validation_error(
    error: ToolArgumentValidationError,
    *,
    allowed_tools: list[str],
    fingerprint: str,
) -> dict[str, Any]:
    """Build a synthetic tool error payload for a pre-execution correction turn."""
    return {
        "error": {
            "code": error.code,
            "message": error.message,
            "allowed_tools": allowed_tools,
            "schema_fingerprint": fingerprint,
        }
    }


def compile_llm_tools(
    llm_api: llm.APIInstance | None,
) -> CompiledToolSchema | None:
    """Compile HA LLM tools for one model turn."""
    if llm_api is None or not llm_api.tools:
        return None

    serializer = llm_api.custom_serializer or llm.selector_serializer
    tools = compile_tools(llm_api.tools, custom_serializer=serializer)
    return CompiledToolSchema(
        tools=tools,
        fingerprint=schema_fingerprint(tools),
    )
