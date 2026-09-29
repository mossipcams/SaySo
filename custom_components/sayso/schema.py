
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from collections.abc import Callable
from typing import Any, Literal

import voluptuous as vol
from homeassistant.helpers import intent, llm

try:
    from probatio import to_openapi as _to_openapi
except ImportError:
    _to_openapi = None

try:
    from voluptuous_openapi import UNSUPPORTED, convert
except ImportError:
    UNSUPPORTED = object()
    convert = None

from .exceptions import SaySoInvalidToolEnvelopeError
from .tool_schema import (
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
    if node is UNSUPPORTED:
        return True
    return type(node).__name__ == "_Unsupported"


def _convert_parameters(
    schema: Any,
    *,
    custom_serializer: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    converted: Any = UNSUPPORTED
    if convert is not None:
        converted = convert(schema, custom_serializer=custom_serializer)
    if (_is_unsupported_node(converted) or not isinstance(converted, dict)) and (
        _to_openapi is not None
    ):
        try:
            converted = _to_openapi(schema, custom_serializer=custom_serializer)
        except Exception:
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

    declared_domains: frozenset[str] | None = None
    retain_always: bool = False


def _vol_in_allowed_values(validator: Any) -> frozenset[str] | None:
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
    for marker, validator in parameters.schema.items():
        if _schema_marker_name(marker) != "domain":
            continue
        allowed = _vol_in_allowed_values(validator)
        if allowed is not None:
            return allowed
    return None


def _unwrap_source_tool(tool: llm.Tool) -> llm.Tool:
    while isinstance(tool, llm.NamespacedTool):
        tool = tool.tool
    return tool


def _is_query_tool(tool: llm.Tool) -> bool:
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
    _cached_compile_tools.cache_clear()


def sanitize_openapi_schema(node: Any) -> Any:
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
    return _build_compiled_tools_from_source(source_json)


def compile_tools(
    tools: list[llm.Tool],
    *,
    custom_serializer: Callable[[Any], Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    source_json = emit_tools_source_json(
        tools,
        custom_serializer=custom_serializer,
    )
    return _cached_compile_tools(source_json)


class ToolArgumentFailureCode(StrEnum):

    SCHEMA_MISMATCH = "schema_mismatch"
    INVALID_ARGUMENTS = "invalid_arguments"


@dataclass(frozen=True, slots=True)
class ToolArgumentValidationError:

    code: ToolArgumentFailureCode
    message: str
    tool_name: str


def build_tool_map(tools: list[llm.Tool]) -> dict[str, llm.Tool]:
    return {tool.name: tool for tool in tools}


def build_tool_availability_names(tools: list[llm.Tool]) -> set[str]:
    return set(build_tool_map(tools))


def _schema_marker_name(marker: Any) -> str | None:
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
    if llm_api is None or not llm_api.tools:
        return None

    serializer = llm_api.custom_serializer or llm.selector_serializer
    tools = compile_tools(llm_api.tools, custom_serializer=serializer)
    return CompiledToolSchema(
        tools=tools,
        fingerprint=schema_fingerprint(tools),
    )
