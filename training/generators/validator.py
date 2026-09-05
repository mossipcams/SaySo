"""Additional validator utilities."""

from __future__ import annotations

from typing import Any

from generators.validate import validate_row, validate_spec, validate_utterance

__all__ = ["validate_row", "validate_spec", "validate_utterance", "corrupt_spec"]


def corrupt_spec(spec: dict[str, Any], field: str) -> dict[str, Any]:
    """Produce intentionally corrupted spec for negative testing."""
    corrupted = dict(spec)
    if field == "wrong_tool":
        expected = dict(corrupted.get("expected", {}))
        calls = list(expected.get("calls") or [])
        if calls:
            calls[0] = dict(calls[0])
            calls[0]["name"] = "NonexistentTool"
            expected["calls"] = calls
            corrupted["expected"] = expected
    elif field == "wrong_entity":
        expected = dict(corrupted.get("expected", {}))
        calls = list(expected.get("calls") or [])
        if calls:
            calls[0] = dict(calls[0])
            args = dict(calls[0].get("arguments", {}))
            args["name"] = "Invented Device"
            calls[0]["arguments"] = args
            expected["calls"] = calls
            corrupted["expected"] = expected
    return corrupted
