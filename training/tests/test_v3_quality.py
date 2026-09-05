"""Additional v3 quality eval contract tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters.schema import ALLOWED_HASS_TOOLS  # noqa: E402
from evals.v3_quality import build_gold_examples, build_shadow_examples, expected_tool_calls  # noqa: E402


def test_gold_tool_names_are_schema_v2_only() -> None:
    for row in build_gold_examples():
        for call in expected_tool_calls(row):
            assert call["function"]["name"] in ALLOWED_HASS_TOOLS


def test_shadow_tool_names_are_schema_v2_only() -> None:
    for row in build_shadow_examples(seed=20260906, count=100):
        for call in expected_tool_calls(row):
            assert call["function"]["name"] in ALLOWED_HASS_TOOLS
