"""The model boundary: what a llama.cpp tool call must survive to execute.

Nothing here talks to Home Assistant or to the model. These are the pure
decisions the conversation agent makes about untrusted model output — is the
batch well-formed, which schema was the model actually shown, what stable code
describes the failure — kept apart from the turn loop that acts on them.
"""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.helpers import llm

from .client import ToolCall
from .diagnostics import (
    BoundaryFailureCode,
    BoundaryPhase,
    record_boundary_failure,
)
from .exceptions import (
    SaySoError,
    SaySoInvalidResponseError,
    SaySoTimeoutError,
)
from .schema import (
    CompiledToolSchema,
    ToolArgumentFailureCode,
    ToolArgumentValidationError,
    validate_tool_arguments,
)
from .tracing import ErrorType, Stage, TraceContext

_LOGGER = logging.getLogger(__name__)

# Each boundary code answers the same two questions once: which stage of the
# interaction broke, and how the trace should classify it.
_BOUNDARY_OUTCOMES: dict[BoundaryFailureCode, tuple[Stage, ErrorType]] = {
    BoundaryFailureCode.SCHEMA_MISMATCH: (Stage.TOOL_PARSE, ErrorType.SCHEMA_MISMATCH),
    BoundaryFailureCode.INVALID_ARGUMENTS: (
        Stage.TOOL_PARSE,
        ErrorType.INVALID_ARGUMENTS,
    ),
    BoundaryFailureCode.UNAVAILABLE_TOOL: (
        Stage.TOOL_PARSE,
        ErrorType.UNAVAILABLE_TOOL,
    ),
    BoundaryFailureCode.REQUEST_TIMEOUT: (Stage.INFERENCE, ErrorType.MODEL_TIMEOUT),
    BoundaryFailureCode.ITERATION_LIMIT: (Stage.INFERENCE, ErrorType.ITERATION_LIMIT),
    BoundaryFailureCode.TOOL_EXECUTION_FAILED: (
        Stage.HA_ACTION,
        ErrorType.HA_ACTION_FAILED,
    ),
}

# Ordered most specific first; anything SaySo did not raise stays UNKNOWN.
_INFERENCE_ERROR_TYPES: tuple[tuple[type[BaseException], ErrorType], ...] = (
    (SaySoTimeoutError, ErrorType.MODEL_TIMEOUT),
    (SaySoInvalidResponseError, ErrorType.INVALID_MODEL_OUTPUT),
    (SaySoError, ErrorType.MODEL_UNAVAILABLE),
)


def inference_error_type(err: BaseException) -> ErrorType:
    """Classify why one llama.cpp request failed."""
    for error_class, error_type in _INFERENCE_ERROR_TYPES:
        if isinstance(err, error_class):
            return error_type
    return ErrorType.UNKNOWN


def boundary_schema(
    phase: BoundaryPhase,
    *,
    active_schema: CompiledToolSchema | None,
    complete_schema: CompiledToolSchema | None,
    correction_used: bool = False,
) -> CompiledToolSchema | None:
    """Return the schema whose fingerprint matches the tools sent in this phase."""
    corrected = phase == BoundaryPhase.CORRECTION or (
        phase == BoundaryPhase.EXECUTION and correction_used
    )
    if corrected:
        return complete_schema
    return active_schema if active_schema is not None else complete_schema


def record_boundary(
    entry_id: str,
    code: BoundaryFailureCode,
    phase: BoundaryPhase,
    schema: CompiledToolSchema | None,
    *,
    ha_error: str | None = None,
    trace: TraceContext | None = None,
) -> None:
    """Record one boundary failure and log its stable code and phase."""
    record_boundary_failure(
        entry_id,
        code,
        phase,
        fingerprint=schema.fingerprint if schema else None,
        ha_error=ha_error,
    )
    trace_id = trace.trace_id if trace is not None else None
    _LOGGER.debug(
        "trace_id=%s SaySo boundary failure: code=%s phase=%s",
        trace_id,
        code.value,
        phase.value,
    )
    if code == BoundaryFailureCode.TOOL_EXECUTION_FAILED and ha_error:
        _LOGGER.warning(
            "trace_id=%s SaySo tool execution failed: ha_error=%s", trace_id, ha_error
        )
    if trace is not None:
        stage, error_type = _BOUNDARY_OUTCOMES.get(
            code, (Stage.SAYSO_REQUEST, ErrorType.UNKNOWN)
        )
        trace.fail(stage, error_type, ha_error or code.value)


def validation_failure_code(
    error: ToolArgumentValidationError,
) -> BoundaryFailureCode:
    """Map a tool-argument validation error to a boundary diagnostic code."""
    if error.code == ToolArgumentFailureCode.SCHEMA_MISMATCH:
        return BoundaryFailureCode.SCHEMA_MISMATCH
    return BoundaryFailureCode.INVALID_ARGUMENTS


def is_well_formed_batch(tool_calls: list[ToolCall]) -> bool:
    """Return whether every call in the batch is structurally usable."""
    ids = [call.id for call in tool_calls]
    return (
        all(call.id and call.name for call in tool_calls)
        and len(set(ids)) == len(ids)
        and all(isinstance(call.arguments, dict) for call in tool_calls)
    )


def validate_arguments(
    tool_calls: list[ToolCall],
    tool_map: dict[str, llm.Tool],
    trace: TraceContext,
) -> tuple[
    list[tuple[ToolCall, dict[str, Any]]],
    list[tuple[ToolCall, ToolArgumentValidationError]],
]:
    """Split a batch into calls Home Assistant's schemas accept and ones they reject."""
    validated: list[tuple[ToolCall, dict[str, Any]]] = []
    failures: list[tuple[ToolCall, ToolArgumentValidationError]] = []
    for tool_call in tool_calls:
        normalized_args, error = validate_tool_arguments(
            tool_map[tool_call.name], tool_call.arguments
        )
        if error is None:
            validated.append((tool_call, normalized_args))
            continue
        _LOGGER.debug(
            "trace_id=%s tool argument validation failed for %s (%s): %s",
            trace.trace_id,
            tool_call.name,
            error.code,
            error.message,
        )
        failures.append((tool_call, error))
    return validated, failures


def is_tool_execution_failure(tool_result: dict[str, Any]) -> bool:
    """Return whether HA reported a tool execution exception, not a negative result."""
    return "error" in tool_result and "success" not in tool_result


def first_target(tool_result: dict[str, Any]) -> str | None:
    """Return the first entity Home Assistant reported as successfully targeted."""
    data = tool_result.get("data")
    successes = data.get("success") if isinstance(data, dict) else None
    if not isinstance(successes, list):
        return None
    for target in successes:
        if isinstance(target, dict) and isinstance(target.get("id"), str):
            return target["id"]
    return None


def action_metadata(
    validated_tool_calls: list[tuple[ToolCall, dict[str, Any]]],
    tool_map: dict[str, llm.Tool],
) -> dict[str, Any]:
    """Return small, non-sensitive metadata describing a tool batch.

    Only identifiers are kept: no full arguments, prompts or state dumps.
    """
    if not validated_tool_calls:
        return {}
    tool_call, normalized_args = validated_tool_calls[0]
    metadata: dict[str, Any] = {"tool": tool_map[tool_call.name].name}
    if len(validated_tool_calls) > 1:
        metadata["batch"] = len(validated_tool_calls)
    domain = normalized_args.get("domain")
    if isinstance(domain, list) and domain:
        domain = domain[0]
    if isinstance(domain, str) and domain:
        metadata["domain"] = domain
    name = normalized_args.get("name")
    if isinstance(name, str) and name:
        metadata["target"] = name
    return metadata


def apply_action_summary(trace: TraceContext, metadata: dict[str, Any]) -> None:
    """Promote the executed action onto the interaction summary."""
    trace.tool = metadata.get("tool") or trace.tool
    target = metadata.get("target")
    if isinstance(target, str) and target:
        trace.target = target
        if "." in target:
            trace.domain = target.split(".", 1)[0]
    if trace.domain is None:
        domain = metadata.get("domain")
        if isinstance(domain, str) and domain:
            trace.domain = domain
