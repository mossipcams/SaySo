"""Tests for HA tool schema compilation."""

from __future__ import annotations

import json
from typing import Any

import pytest
import voluptuous as vol
from homeassistant.helpers import llm
from voluptuous_openapi import UNSUPPORTED

from custom_components.sayso.schema import (
    ToolArgumentFailureCode,
    ToolRoutingMetadata,
    _build_compiled_tools_from_source,
    build_tool_map,
    canonicalize_schema,
    clear_compile_cache,
    compile_parameters,
    compile_tool,
    compile_tools,
    emit_canonical_json,
    extract_tool_routing_metadata,
    normalize_schema,
    schema_fingerprint,
    validate_tool_arguments,
)


@pytest.fixture(autouse=True)
def _reset_schema_compile_cache() -> None:
    """Isolate compile cache between schema tests."""
    clear_compile_cache()
    yield
    clear_compile_cache()


class _FakeTool(llm.Tool):
    """Minimal HA tool for schema compilation tests."""

    def __init__(
        self,
        *,
        name: str = "FakeDeviceControl",
        description: str = "Control a fake device with rich constraints.",
        parameters: vol.Schema | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters or _rich_parameters_schema()

    async def async_call(
        self,
        hass: Any,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict[str, Any]:
        return {"ok": True}


def _custom_string_serializer(schema: Any) -> Any:
    if schema is str:
        return {"type": "string", "format": "custom-string"}
    return UNSUPPORTED


def _rich_parameters_schema() -> vol.Schema:
    return vol.Schema(
        {
            vol.Required("name", description="Human-readable device name"): vol.All(
                str, vol.Length(min=1, max=64)
            ),
            vol.Required("mode"): vol.In(["auto", "manual", "off"]),
            vol.Optional("brightness", default=50): vol.All(
                vol.Coerce(int), vol.Range(min=0, max=100)
            ),
            vol.Optional("tags", description="Labels to apply"): [str],
            vol.Optional("config"): {
                vol.Required("nested_id"): vol.Match(r"^[a-z]+$"),
                vol.Optional("note", description="Nested note"): str,
            },
            vol.Optional("started_at"): vol.All(
                str, vol.Match(r"^\d{4}-\d{2}-\d{2}$")
            ),
        }
    )


def test_normalize_schema_removes_redundant_text() -> None:
    """Empty descriptions, $schema, and duplicate titles are stripped."""
    noisy = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "brightness": {
                "type": "integer",
                "title": "brightness",
                "description": "",
                "minimum": 0,
                "maximum": 100,
            },
            "name": {
                "type": "string",
                "title": "Display Name",
                "description": "Human-readable device name",
            },
            "mode": {
                "type": "string",
                "description": "   ",
                "enum": ["auto", "off"],
            },
            "config": {
                "type": "object",
                "title": "config",
                "properties": {
                    "nested_id": {
                        "type": "string",
                        "title": "nested_id",
                        "pattern": "^[a-z]+$",
                    },
                    "note": {
                        "type": "string",
                        "title": "Nested note",
                        "description": "Nested note",
                    },
                },
            },
        },
        "required": ["brightness"],
    }

    cleaned = normalize_schema(noisy, top_level=True)

    assert "$schema" not in cleaned
    brightness = cleaned["properties"]["brightness"]
    assert "title" not in brightness
    assert "description" not in brightness
    assert brightness["minimum"] == 0
    assert brightness["maximum"] == 100

    name = cleaned["properties"]["name"]
    assert name["title"] == "Display Name"
    assert name["description"] == "Human-readable device name"

    mode = cleaned["properties"]["mode"]
    assert "description" not in mode
    assert mode["enum"] == ["auto", "off"]

    config = cleaned["properties"]["config"]
    assert "title" not in config
    nested_id = config["properties"]["nested_id"]
    assert "title" not in nested_id
    assert nested_id["pattern"] == "^[a-z]+$"
    note = config["properties"]["note"]
    assert note["title"] == "Nested note"
    assert note["description"] == "Nested note"

    function_wrapped = normalize_schema(
        {
            "type": "function",
            "function": {
                "name": "FakeDeviceControl",
                "title": "FakeDeviceControl",
                "description": "Control a fake device.",
                "parameters": noisy,
            },
        },
        top_level=True,
    )
    function = function_wrapped["function"]
    assert "title" not in function
    assert function["description"] == "Control a fake device."
    assert "$schema" not in function["parameters"]


def test_compile_parameters_applies_schema_normalization(monkeypatch: Any) -> None:
    """The compiler strips redundant metadata from converted schemas."""
    noisy = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "title": "name",
                "description": "",
                "minLength": 1,
            },
            "mode": {
                "type": "string",
                "description": "Distinct mode hint",
                "enum": ["auto", "manual", "off"],
            },
        },
        "required": ["name", "mode"],
    }
    monkeypatch.setattr(
        "custom_components.sayso.schema.convert",
        lambda *_args, **_kwargs: noisy,
    )

    parameters = compile_parameters(_rich_parameters_schema())

    assert "$schema" not in parameters
    assert "title" not in parameters["properties"]["name"]
    assert "description" not in parameters["properties"]["name"]
    assert parameters["properties"]["name"]["minLength"] == 1
    assert parameters["properties"]["mode"]["description"] == "Distinct mode hint"
    assert parameters["properties"]["mode"]["enum"] == ["auto", "manual", "off"]


def test_compile_tool_strips_duplicate_function_title(monkeypatch: Any) -> None:
    """Function-level titles matching the tool name are removed."""
    noisy = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {},
    }
    monkeypatch.setattr(
        "custom_components.sayso.schema.convert",
        lambda *_args, **_kwargs: noisy,
    )

    compiled = compile_tool(_FakeTool(name="FakeDeviceControl"))

    function = compiled["function"]
    assert function["name"] == "FakeDeviceControl"
    assert "title" not in function
    assert "$schema" not in function["parameters"]


def test_compile_parameters_preserves_executable_constraints() -> None:
    """Executable HA constraints survive voluptuous_openapi compilation."""
    parameters = compile_parameters(
        _rich_parameters_schema(),
        custom_serializer=_custom_string_serializer,
    )

    properties = parameters["properties"]
    assert parameters["type"] == "object"
    assert parameters["required"] == ["mode", "name"]

    name = properties["name"]
    assert name["type"] == "string"
    assert name["format"] == "custom-string"
    assert name["minLength"] == 1
    assert name["maxLength"] == 64
    assert name["description"] == "Human-readable device name"

    assert properties["mode"]["enum"] == ["auto", "manual", "off"]

    brightness = properties["brightness"]
    assert brightness["type"] == "integer"
    assert brightness["minimum"] == 0
    assert brightness["maximum"] == 100
    assert brightness["default"] == 50

    tags = properties["tags"]
    assert tags["type"] == "array"
    assert tags["items"] == {"type": "string", "format": "custom-string"}
    assert tags["description"] == "Labels to apply"

    config = properties["config"]
    assert config["type"] == "object"
    assert config["required"] == ["nested_id"]
    nested_id = config["properties"]["nested_id"]
    assert nested_id["pattern"] == "^[a-z]+$"
    assert nested_id["type"] == "string"
    assert config["properties"]["note"]["description"] == "Nested note"

    started_at = properties["started_at"]
    assert started_at["format"] == "custom-string"
    assert started_at["pattern"] == r"^\d{4}-\d{2}-\d{2}$"


def test_compile_tool_matches_llama_cpp_function_shape() -> None:
    """Compiled tools match the OpenAI function shape sent to llama.cpp."""
    compiled = compile_tool(
        _FakeTool(),
        custom_serializer=_custom_string_serializer,
    )

    assert compiled == {
        "type": "function",
        "function": {
            "name": "FakeDeviceControl",
            "description": "Control a fake device with rich constraints.",
            "parameters": compile_parameters(
                _rich_parameters_schema(),
                custom_serializer=_custom_string_serializer,
            ),
        },
    }


def _schema_with_reordered_keys() -> vol.Schema:
    """Same fields as _rich_parameters_schema() but declared in a different order."""
    return vol.Schema(
        {
            vol.Optional("started_at"): vol.All(
                str, vol.Match(r"^\d{4}-\d{2}-\d{2}$")
            ),
            vol.Required("mode"): vol.In(["auto", "manual", "off"]),
            vol.Optional("config"): {
                vol.Optional("note", description="Nested note"): str,
                vol.Required("nested_id"): vol.Match(r"^[a-z]+$"),
            },
            vol.Optional("tags", description="Labels to apply"): [str],
            vol.Required("name", description="Human-readable device name"): vol.All(
                str, vol.Length(min=1, max=64)
            ),
            vol.Optional("brightness", default=50): vol.All(
                vol.Coerce(int), vol.Range(min=0, max=100)
            ),
        }
    )


def test_canonical_schema_is_order_independent() -> None:
    """Semantically identical tools produce identical canonical JSON and fingerprints."""
    alpha = _FakeTool(name="AlphaTool")
    beta = _FakeTool(name="BetaTool", description="Second tool.")
    reordered = _FakeTool(
        name="GammaTool",
        parameters=_schema_with_reordered_keys(),
    )
    baseline = _FakeTool(
        name="GammaTool",
        parameters=_rich_parameters_schema(),
    )

    forward = compile_tools(
        [alpha, beta, reordered],
        custom_serializer=_custom_string_serializer,
    )
    reverse = compile_tools(
        [reordered, beta, alpha],
        custom_serializer=_custom_string_serializer,
    )
    baseline_tools = compile_tools(
        [alpha, beta, baseline],
        custom_serializer=_custom_string_serializer,
    )

    forward_json = emit_canonical_json(forward)
    reverse_json = emit_canonical_json(reverse)
    baseline_json = emit_canonical_json(baseline_tools)

    assert forward == reverse == baseline_tools
    assert forward_json == reverse_json == baseline_json
    assert forward_json == json.dumps(
        forward,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    assert schema_fingerprint(forward) == schema_fingerprint(reverse)
    assert schema_fingerprint(forward) == schema_fingerprint(baseline_tools)

    altered = compile_tools(
        [
            _FakeTool(
                name="AlphaTool",
                parameters=vol.Schema(
                    {
                        vol.Required("name", description="Human-readable device name"): vol.All(
                            str, vol.Length(min=1, max=64)
                        ),
                        vol.Required("mode"): vol.In(["auto", "manual", "off"]),
                        vol.Optional("brightness", default=50): vol.All(
                            vol.Coerce(int), vol.Range(min=0, max=100)
                        ),
                        vol.Optional("tags", description="Labels to apply"): [str],
                        vol.Optional("config"): {
                            vol.Required("nested_id"): vol.Match(r"^[a-z]+$"),
                            vol.Optional("note", description="Nested note"): str,
                        },
                        vol.Optional("started_at"): vol.All(
                            str, vol.Match(r"^\d{4}-\d{2}-\d{2}$")
                        ),
                        vol.Optional("level"): vol.All(
                            vol.Coerce(int), vol.Range(min=0, max=10)
                        ),
                    }
                ),
            ),
            beta,
            baseline,
        ],
        custom_serializer=_custom_string_serializer,
    )
    assert schema_fingerprint(altered) != schema_fingerprint(forward)


def test_canonicalize_schema_sorts_keys_and_required() -> None:
    """Canonicalization sorts mapping keys and required arrays."""
    scrambled = {
        "type": "object",
        "required": ["mode", "name"],
        "properties": {
            "mode": {"type": "string", "enum": ["auto", "manual", "off"]},
            "name": {"type": "string", "minLength": 1},
            "brightness": {"type": "integer", "minimum": 0, "maximum": 100},
        },
    }

    canonical = canonicalize_schema(scrambled)

    assert list(canonical.keys()) == ["properties", "required", "type"]
    assert list(canonical["properties"].keys()) == ["brightness", "mode", "name"]
    assert canonical["required"] == ["mode", "name"]


def test_compile_tools_returns_canonical_name_order_and_shape() -> None:
    """compile_tools returns canonical OpenAI function entries sorted by name."""
    alpha = _FakeTool(name="AlphaTool")
    beta = _FakeTool(name="BetaTool", description="Second tool.")

    compiled = compile_tools(
        [beta, alpha],
        custom_serializer=_custom_string_serializer,
    )

    assert len(compiled) == 2
    assert compiled[0]["type"] == "function"
    assert compiled[0]["function"]["name"] == "AlphaTool"
    assert compiled[1]["function"]["name"] == "BetaTool"
    assert compiled[1]["function"]["description"] == "Second tool."
    assert "parameters" in compiled[0]["function"]
    assert compiled[0]["function"]["parameters"]["required"] == ["mode", "name"]


def test_compile_tools_caches_identical_canonical_input(monkeypatch: Any) -> None:
    """Identical tool definitions compile once and reuse the cached object."""
    build_calls = 0
    original_build = _build_compiled_tools_from_source

    def counting_build(source_json: str) -> tuple[dict[str, Any], ...]:
        nonlocal build_calls
        build_calls += 1
        return original_build(source_json)

    monkeypatch.setattr(
        "custom_components.sayso.schema._build_compiled_tools_from_source",
        counting_build,
    )

    tool = _FakeTool()
    serializer = _custom_string_serializer
    first = compile_tools([tool], custom_serializer=serializer)
    second = compile_tools([tool], custom_serializer=serializer)

    assert build_calls == 1
    assert first is second


def test_compile_tools_rebuilds_when_description_changes(monkeypatch: Any) -> None:
    """A changed tool description invalidates the compile cache."""
    build_calls = 0
    original_build = _build_compiled_tools_from_source

    def counting_build(source_json: str) -> tuple[dict[str, Any], ...]:
        nonlocal build_calls
        build_calls += 1
        return original_build(source_json)

    monkeypatch.setattr(
        "custom_components.sayso.schema._build_compiled_tools_from_source",
        counting_build,
    )

    serializer = _custom_string_serializer
    compile_tools(
        [_FakeTool(description="First description.")],
        custom_serializer=serializer,
    )
    compile_tools(
        [_FakeTool(description="Second description.")],
        custom_serializer=serializer,
    )

    assert build_calls == 2


def test_build_tool_map_indexes_tools_by_name() -> None:
    """build_tool_map returns a complete-name lookup table."""
    alpha = _FakeTool(name="AlphaTool")
    beta = _FakeTool(name="BetaTool")

    tool_map = build_tool_map([beta, alpha])

    assert set(tool_map) == {"AlphaTool", "BetaTool"}
    assert tool_map["AlphaTool"] is alpha
    assert tool_map["BetaTool"] is beta


def test_build_tool_map_indexes_namespaced_tools_by_suffix_alias() -> None:
    """build_tool_map also indexes HA 2026.9 namespaced tools by their suffix."""
    inner = _FakeTool(name="HassTurnOn")
    namespaced = llm.NamespacedTool("intent", inner)

    tool_map = build_tool_map([namespaced])

    assert tool_map["intent__HassTurnOn"] is namespaced
    assert tool_map["HassTurnOn"] is namespaced


def test_build_tool_map_exact_name_wins_over_suffix_alias_collision() -> None:
    """Exact tool names are never replaced by a suffix alias from another tool."""
    alias_source = _FakeTool(name="custom__HassTurnOn")
    exact = _FakeTool(name="HassTurnOn")

    tool_map = build_tool_map([alias_source, exact])

    assert tool_map["custom__HassTurnOn"] is alias_source
    assert tool_map["HassTurnOn"] is exact


def test_validate_tool_arguments_normalizes_coerced_values() -> None:
    """Valid arguments are normalized by the HA Voluptuous schema."""
    tool = _FakeTool()

    normalized, error = validate_tool_arguments(
        tool,
        {
            "name": "Kitchen",
            "mode": "manual",
            "brightness": "75",
            "tags": ["main"],
        },
    )

    assert error is None
    assert normalized == {
        "name": "Kitchen",
        "mode": "manual",
        "brightness": 75,
        "tags": ["main"],
    }


def test_validate_tool_arguments_reports_missing_required_fields() -> None:
    """Missing required fields are classified as schema mismatch."""
    tool = _FakeTool()

    normalized, error = validate_tool_arguments(
        tool,
        {"name": "Kitchen"},
    )

    assert normalized is None
    assert error is not None
    assert error.code == ToolArgumentFailureCode.SCHEMA_MISMATCH
    assert error.tool_name == "FakeDeviceControl"
    assert "mode" in error.message


def test_validate_tool_arguments_reports_unexpected_fields() -> None:
    """Unexpected top-level fields are classified as schema mismatch."""
    tool = _FakeTool()

    normalized, error = validate_tool_arguments(
        tool,
        {
            "name": "Kitchen",
            "mode": "manual",
            "unexpected": True,
        },
    )

    assert normalized is None
    assert error is not None
    assert error.code == ToolArgumentFailureCode.SCHEMA_MISMATCH
    assert "unexpected" in error.message.lower()


@pytest.mark.parametrize(
    ("arguments", "message_fragment"),
    [
        pytest.param(
            {"name": "Kitchen", "mode": "manual", "brightness": 200},
            "at most 100",
            id="range_violation",
        ),
        pytest.param(
            {"name": "Kitchen", "mode": "turbo"},
            "value must be one of",
            id="enum_violation",
        ),
        pytest.param(
            {"name": 123, "mode": "manual"},
            "expected str",
            id="wrong_type",
        ),
        pytest.param(
            {
                "name": "Kitchen",
                "mode": "manual",
                "config": {"nested_id": "BAD-ID"},
            },
            "does not match regular expression",
            id="nested_failure",
        ),
    ],
)
def test_validate_tool_arguments_reports_invalid_values(
    arguments: dict[str, Any],
    message_fragment: str,
) -> None:
    """Type, enum, range, and nested failures are invalid arguments."""
    tool = _FakeTool()

    normalized, error = validate_tool_arguments(tool, arguments)

    assert normalized is None
    assert error is not None
    assert error.code == ToolArgumentFailureCode.INVALID_ARGUMENTS
    assert message_fragment in error.message


def test_compile_tools_rebuilds_when_constraint_changes(monkeypatch: Any) -> None:
    """A changed executable constraint invalidates the compile cache."""
    build_calls = 0
    original_build = _build_compiled_tools_from_source

    def counting_build(source_json: str) -> tuple[dict[str, Any], ...]:
        nonlocal build_calls
        build_calls += 1
        return original_build(source_json)

    monkeypatch.setattr(
        "custom_components.sayso.schema._build_compiled_tools_from_source",
        counting_build,
    )

    serializer = _custom_string_serializer
    compile_tools([_FakeTool()], custom_serializer=serializer)
    compile_tools(
        [
            _FakeTool(
                parameters=vol.Schema(
                    {
                        vol.Required("name", description="Human-readable device name"): vol.All(
                            str, vol.Length(min=1, max=64)
                        ),
                        vol.Required("mode"): vol.In(["auto", "manual", "off"]),
                        vol.Optional("level"): vol.All(
                            vol.Coerce(int), vol.Range(min=0, max=10)
                        ),
                    }
                )
            )
        ],
        custom_serializer=serializer,
    )

    assert build_calls == 2


def test_extract_tool_routing_metadata_reads_explicit_domain_constraints() -> None:
    """Declared vol.In domain constraints are extracted without name inference."""
    class _DomainRestrictedTool(llm.Tool):
        name = "FanSpeed"
        description = "Set a fan speed."
        parameters = vol.Schema(
            {
                "domain": vol.All(
                    vol.Coerce(list),
                    [vol.In(["fan"])],
                )
            }
        )

        async def async_call(
            self,
            hass: Any,
            tool_input: llm.ToolInput,
            llm_context: llm.LLMContext,
        ) -> dict[str, Any]:
            return {"ok": True}

    class _UnrestrictedTool(llm.Tool):
        name = "light_by_name_only"
        description = "No explicit domain metadata."
        parameters = vol.Schema({})

        async def async_call(
            self,
            hass: Any,
            tool_input: llm.ToolInput,
            llm_context: llm.LLMContext,
        ) -> dict[str, Any]:
            return {"ok": True}

    restricted = extract_tool_routing_metadata(_DomainRestrictedTool())
    unknown = extract_tool_routing_metadata(_UnrestrictedTool())

    assert restricted == ToolRoutingMetadata(declared_domains=frozenset({"fan"}))
    assert unknown == ToolRoutingMetadata()
