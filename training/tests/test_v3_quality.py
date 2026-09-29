
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))

from adapters.schema import ALLOWED_HASS_TOOLS
from evals.cases import cases_with_tag
from evals.runner import render_case


def test_gold_tool_names_are_schema_v2_or_offered_script_tools() -> None:
    for case in cases_with_tag("gold"):
        offered = {tool["function"]["name"] for tool in render_case(case)["tools"]}
        for call in case.expected["calls"]:
            assert call["name"] in ALLOWED_HASS_TOOLS or call["name"] in offered, call["name"]


def test_shadow_tool_names_are_schema_v2_or_offered_script_tools() -> None:
    for case in cases_with_tag("shadow"):
        offered = {tool["function"]["name"] for tool in render_case(case)["tools"]}
        for call in case.expected["calls"]:
            assert call["name"] in ALLOWED_HASS_TOOLS or call["name"] in offered, call["name"]
