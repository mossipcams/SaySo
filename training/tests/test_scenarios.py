"""Tests for scenario generation."""

from __future__ import annotations

from generators.scenarios import build_scenario, semantic_id


def test_scenario_has_semantic_id() -> None:
    scenario = build_scenario(
        index=0,
        seed=10,
        capability="fans",
        operation="set_speed",
        home_size=16,
    )
    assert scenario["semantic_id"] == semantic_id(scenario)
    assert scenario["expected"]["kind"] in {"action", "no_action", "status"}


def test_scenario_expected_before_utterance() -> None:
    scenario = build_scenario(
        index=1,
        seed=10,
        capability="locks",
        operation="lock",
        home_size=16,
    )
    assert "expected" in scenario
    assert "utterance" not in scenario or scenario.get("utterance") is None
