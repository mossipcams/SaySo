"""Focused checks for active schema identity during domain routing."""

from __future__ import annotations

import hashlib
from typing import Any

import voluptuous as vol
from homeassistant.helpers import config_validation as cv, llm

from custom_components.sayso.routing import select_schema_for_domain
from custom_components.sayso.schema import (
    CompiledToolSchema,
    compile_tools,
    emit_canonical_json,
    schema_fingerprint,
)


class _FakeDomainTool(llm.Tool):
    """Minimal HA tool with optional domain restriction."""

    def __init__(
        self,
        *,
        name: str,
        domain_validator: Any | None = None,
    ) -> None:
        self.name = name
        self.description = f"Fake tool {name}."
        schema: dict[Any, Any] = {}
        if domain_validator is not None:
            schema["domain"] = domain_validator
        self.parameters = vol.Schema(schema)

    async def async_call(
        self,
        hass: Any,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict[str, Any]:
        return {"ok": True}


def _complete_schema(*tools: llm.Tool) -> CompiledToolSchema:
    source_tools = list(tools)
    compiled = compile_tools(source_tools)
    return CompiledToolSchema(
        tools=compiled,
        fingerprint=schema_fingerprint(list(compiled)),
    )


def test_confident_light_route_returns_subset_with_own_fingerprint() -> None:
    """Filtered routing yields fewer tools and a fingerprint of its exact bytes."""
    source_tools = [
        _FakeDomainTool(name="always_on"),
        _FakeDomainTool(
            name="light_tool",
            domain_validator=vol.All(cv.ensure_list, [vol.In(["light"])]),
        ),
        _FakeDomainTool(
            name="switch_tool",
            domain_validator=vol.All(cv.ensure_list, [vol.In(["switch"])]),
        ),
        _FakeDomainTool(
            name="fan_tool",
            domain_validator=vol.All(cv.ensure_list, [vol.In(["fan"])]),
        ),
    ]
    complete = _complete_schema(*source_tools)

    active = select_schema_for_domain(complete, source_tools, "light")

    assert active is not complete
    assert len(active.tools) < len(complete.tools)
    active_names = {tool["function"]["name"] for tool in active.tools}
    assert "light_tool" in active_names
    assert "always_on" in active_names
    assert "switch_tool" not in active_names
    assert "fan_tool" not in active_names

    active_bytes = emit_canonical_json(list(active.tools))
    expected_fingerprint = f"sha256:{hashlib.sha256(active_bytes).hexdigest()}"
    assert active.fingerprint == expected_fingerprint
    assert active.fingerprint != complete.fingerprint


def test_uncertain_route_returns_complete_schema_unchanged() -> None:
    """Unknown routing must return the same complete CompiledToolSchema object."""
    source_tools = [
        _FakeDomainTool(name="tool_a"),
        _FakeDomainTool(
            name="tool_b",
            domain_validator=vol.All(cv.ensure_list, [vol.In(["switch"])]),
        ),
    ]
    complete = _complete_schema(*source_tools)

    active = select_schema_for_domain(complete, source_tools, None)

    assert active is complete
