"""Focused checks for schema_fingerprint canonical hashing."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from custom_components.sayso.schema import emit_canonical_json, schema_fingerprint

_FINGERPRINT_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def _tool(name: str, *, description: str | None = None) -> dict[str, Any]:
    function: dict[str, Any] = {
        "name": name,
        "parameters": {
            "type": "object",
            "required": ["mode", "name"],
            "properties": {
                "mode": {"type": "string", "enum": ["auto", "off"]},
                "name": {"type": "string", "minLength": 1},
            },
        },
    }
    if description is not None:
        function["description"] = description
    return {"type": "function", "function": function}


def _scrambled_tool(name: str) -> dict[str, Any]:
    """Same tool content with non-canonical key ordering."""
    return {
        "function": {
            "parameters": {
                "properties": {
                    "name": {"minLength": 1, "type": "string"},
                    "mode": {"enum": ["auto", "off"], "type": "string"},
                },
                "required": ["name", "mode"],
                "type": "object",
            },
            "name": name,
        },
        "type": "function",
    }


def test_schema_fingerprint_matches_sha256_prefix_format() -> None:
    """Every fingerprint is sha256: followed by 64 lowercase hex characters."""
    tools = [_tool("AlphaTool"), _tool("BetaTool", description="Second tool.")]
    fingerprint = schema_fingerprint(tools)
    assert _FINGERPRINT_PATTERN.match(fingerprint)


def test_schema_fingerprint_is_order_invariant() -> None:
    """Equivalent tools with different dictionary and tool order share a fingerprint."""
    canonical_tools = [_tool("AlphaTool"), _tool("BetaTool", description="Second tool.")]
    scrambled_tools = [
        _scrambled_tool("BetaTool"),
        _scrambled_tool("AlphaTool"),
    ]
    scrambled_tools[0]["function"]["description"] = "Second tool."

    assert schema_fingerprint(scrambled_tools) == schema_fingerprint(canonical_tools)
    assert schema_fingerprint(list(reversed(canonical_tools))) == schema_fingerprint(
        canonical_tools
    )


def test_schema_fingerprint_changes_when_canonical_bytes_change() -> None:
    """Any change to canonical emitted bytes produces a different fingerprint."""
    baseline = [_tool("AlphaTool"), _tool("BetaTool", description="Second tool.")]
    altered = [
        _tool("AlphaTool"),
        _tool("BetaTool", description="Changed description."),
    ]

    assert schema_fingerprint(altered) != schema_fingerprint(baseline)


def test_schema_fingerprint_hashes_emit_canonical_json_output() -> None:
    """schema_fingerprint hashes only emit_canonical_json() bytes."""
    tools = [_tool("AlphaTool"), _tool("BetaTool", description="Second tool.")]
    canonical_bytes = emit_canonical_json(tools)
    expected = f"sha256:{hashlib.sha256(canonical_bytes).hexdigest()}"
    assert schema_fingerprint(list(reversed(tools))) == expected
