"""Compile Home Assistant tool schemas for llama.cpp."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from collections.abc import Callable
from typing import Any, Literal

import voluptuous as vol
from homeassistant.helpers import intent, llm

try:  # Home Assistant >= 2026.9 validates with probatio, installed as voluptuous.
    from probatio import to_openapi as _to_openapi
except ImportError:  # pragma: no cover - older Home Assistant
    _to_openapi = None

try:  # Home Assistant <= 2026.8 converted tool schemas with voluptuous_openapi.
    from voluptuous_openapi import UNSUPPORTED, convert
except ImportError:  # pragma: no cover - Home Assistant >= 2026.9 dropped it
    UNSUPPORTED = object()
    convert = None

from .exceptions import SaySoInvalidToolEnvelopeError
# Re-exported: callers and tests import the compiled-schema helpers from here.
from .tool_schema import (  # noqa: F401
    CompiledToolSchema,
    build_compiled_tools_from_source as _build_compiled_tools_from_source,
    canonicalize_compiled_tools,
    canonicalize_schema,
    emit_canonical_json,
    function_envelope,
    normalize_schema,
    schema_fingerprint,
    tool_source_json,
    validate_compiled_tool_envelope,
)

COMPILE_CACHE_MAXSIZE = 32

_UNSUPPORTED_OPENAPI_FALLBACK: dict[str, str] = {"type": "string"}


def _is_unsupported_node(node: Any) -> bool:
    """Return True for voluptuous_openapi or Home Assistant unsupported markers."""
    if node is UNSUPPORTED:
        return True
    return type(node).__name__ == "_Unsupported"


def _convert_parameters(
    schema: Any,
    *,
    custom_serializer: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Convert one tool's parameter schema to OpenAPI, or fail closed.

    Home Assistant 2026.9 swapped voluptuous for ``probatio`` and installs it
    under the ``voluptuous`` name, so ``tool.parameters`` is a
    ``probatio.schema.Schema``. ``voluptuous_openapi.convert`` does not
    understand that and returns its unsupported marker for *every* Home
    Assistant tool. The previous code answered that with an empty
    ``{"type": "object", "properties": {}}``, which offers the model a tool it
    cannot pass a target to -- ``HassTurnOn(name="TV")`` becomes unexpressible
    and the model falls back to prose (issue #52). Prefer probatio's own
    ``to_openapi``, which Home Assistant's bundled integrations use, and raise
    rather than silently shipping a tool stripped of its arguments.

    A genuinely argument-free tool still converts to an empty ``properties``
    mapping and is returned normally; only a failed conversion raises.
    """
    converted: Any = UNSUPPORTED
    if convert is not None:
        converted = convert(schema, custom_serializer=custom_serializer)
    if (_is_unsupported_node(converted) or not isinstance(converted, dict)) and (
        _to_openapi is not None
    ):
        try:
            converted = _to_openapi(schema, custom_serializer=custom_serializer)
        except Exception:  # noqa: BLE001 - reported as a closed failure below
            converted = UNSUPPORTED
    if _is_unsupported_node(converted) or not isinstance(converted, dict):
        raise SaySoInvalidToolEnvelopeError(
            "Home Assistant tool parameters could not be compiled to OpenAPI; "
            "refusing to offer the tool without its arguments"
        )
    sanitized = sanitize_openapi_schema(converted)
    if not isinstance(sanitized, dict):
        raise SaySoInvalidToolEnvelopeError(
            "Compiled tool parameters are not a JSON object"
        )
    return sanitized


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
    if isinstance(tool, llm.IntentTool):
        return tool.name == intent.INTENT_TIMER_STATUS
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


def sanitize_openapi_schema(node: Any) -> Any:
    """Replace UNSUPPORTED and other non-JSON nodes with serializable OpenAPI."""
    if _is_unsupported_node(node):
        return dict(_UNSUPPORTED_OPENAPI_FALLBACK)

    if isinstance(node, dict):
        return {
            key: sanitize_openapi_schema(value) for key, value in node.items()
        }

    if isinstance(node, list):
        return [sanitize_openapi_schema(item) for item in node]

    if isinstance(node, (str, int, float, bool)) or node is None:
        return node

    return dict(_UNSUPPORTED_OPENAPI_FALLBACK)


def compile_parameters(
    schema: Any,
    *,
    custom_serializer: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Compile a Voluptuous schema to OpenAPI parameters."""
    normalized = normalize_schema(
        _convert_parameters(schema, custom_serializer=custom_serializer),
        top_level=True,
    )
    return canonicalize_schema(normalized)


def compile_tool(
    tool: llm.Tool,
    *,
    custom_serializer: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Compile one HA tool to an OpenAI-compatible function definition."""
    compiled = function_envelope(
        tool.name,
        _convert_parameters(tool.parameters, custom_serializer=custom_serializer),
        tool.description,
    )
    validate_compiled_tool_envelope((compiled,))
    return compiled


def _emit_tool_source_entry(
    tool: llm.Tool,
    *,
    custom_serializer: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Return one tool's canonical source payload for cache keys."""
    converted = _convert_parameters(
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
    try:
        return tool_source_json(entries)
    except TypeError as err:
        raise SaySoInvalidToolEnvelopeError(
            "Compiled tool source is not JSON-serializable"
        ) from err


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
    """Map each tool's exact Home Assistant name to its definition.

    Names are exact. Home Assistant 2026.9 names tools ``intent__HassTurnOn``;
    a model that says ``HassTurnOn`` asked for a tool Home Assistant never
    offered, and resolving it anyway would hide that from every eval.
    """
    return {tool.name: tool for tool in tools}


def build_tool_availability_names(tools: list[llm.Tool]) -> set[str]:
    """Return the exact tool names Home Assistant currently offers."""
    return set(build_tool_map(tools))


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
