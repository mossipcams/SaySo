"""Check a model's tool calls against the compiled schema it was shown.

This is the part of pre-execution validation that needs nothing but the
compiled tools, so the conversation agent and the offline eval run the same
code. Home Assistant's own voluptuous validation still runs after it in
production; this layer only rejects what the advertised contract already rules
out. Tool names are exact: a bare ``HassTurnOn`` is not ``intent__HassTurnOn``.

Pure: no Home Assistant imports.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

ViolationCode = Literal["unknown_tool", "schema_mismatch", "invalid_arguments"]

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


@dataclass(frozen=True, slots=True)
class ContractViolation:
    """Why one call cannot execute."""

    code: ViolationCode
    tool_name: str
    message: str


def _value_error(schema: Mapping[str, Any], value: Any, path: str) -> str | None:
    """First reason ``value`` does not satisfy ``schema``, or None."""
    if "anyOf" in schema and "type" not in schema:
        options = [option for option in schema["anyOf"] if isinstance(option, Mapping)]
        if options and all(_value_error(option, value, path) for option in options):
            return f"{path} matches none of the allowed shapes"
        return None
    expected = schema.get("type")
    if isinstance(expected, str) and expected in _JSON_TYPES:
        # bool is an int in Python; JSON does not agree.
        is_bool = isinstance(value, bool)
        if not isinstance(value, _JSON_TYPES[expected]) or (
            is_bool and expected != "boolean"
        ):
            return f"{path} must be {expected}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{path} must be one of {schema['enum']}"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return f"{path} is below {schema['minimum']}"
        if "maximum" in schema and value > schema["maximum"]:
            return f"{path} is above {schema['maximum']}"
    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        for index, item in enumerate(value):
            if error := _value_error(schema["items"], item, f"{path}[{index}]"):
                return error
    return None


def _domains_of(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def check_tool_call(
    name: Any,
    arguments: Any,
    tools: Mapping[str, Mapping[str, Any]],
    exposed_domains: Iterable[str] | None = None,
) -> ContractViolation | None:
    """Return why one call violates the advertised contract, or None."""
    label = name if isinstance(name, str) else repr(name)
    if not isinstance(name, str) or name not in tools:
        return ContractViolation("unknown_tool", label, f"tool {label} is not offered")
    if not isinstance(arguments, dict):
        return ContractViolation("invalid_arguments", name, "arguments must be an object")

    parameters = tools[name].get("parameters") or {}
    properties = parameters.get("properties") or {}
    unexpected = sorted(set(arguments) - set(properties))
    if unexpected:
        return ContractViolation(
            "schema_mismatch", name, f"unexpected argument(s): {unexpected}"
        )
    missing = sorted(set(parameters.get("required") or []) - set(arguments))
    if missing:
        return ContractViolation(
            "schema_mismatch", name, f"missing required argument(s): {missing}"
        )
    groups = [
        option.get("required") or []
        for option in parameters.get("anyOf") or []
        if isinstance(option, Mapping)
    ]
    if groups and not any(set(group) <= set(arguments) for group in groups):
        return ContractViolation(
            "schema_mismatch", name, f"needs one of the argument sets {groups}"
        )
    for key, value in arguments.items():
        if error := _value_error(properties[key], value, key):
            return ContractViolation("invalid_arguments", name, error)
    if exposed_domains is not None and "domain" in arguments:
        exposed = set(exposed_domains)
        unknown = [d for d in _domains_of(arguments["domain"]) if d not in exposed]
        if unknown:
            return ContractViolation(
                "invalid_arguments", name, f"domain(s) {unknown} are not exposed"
            )
    return None


def tools_by_name(
    compiled_tools: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """Index compiled OpenAI tools by their exact function name."""
    return {tool["function"]["name"]: tool["function"] for tool in compiled_tools}


def check_tool_calls(
    calls: Iterable[tuple[Any, Any]],
    compiled_tools: Sequence[Mapping[str, Any]],
    exposed_domains: Iterable[str] | None = None,
) -> list[ContractViolation]:
    """Every contract violation in a batch, in call order."""
    tools = tools_by_name(compiled_tools)
    domains = None if exposed_domains is None else frozenset(exposed_domains)
    return [
        violation
        for name, arguments in calls
        if (violation := check_tool_call(name, arguments, tools, domains)) is not None
    ]
