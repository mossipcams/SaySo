"""Tests for validators."""

from __future__ import annotations

import random

from generators.labels import scenario_to_spec
from generators.scenarios import build_scenario
from generators.validate import validate_row, validate_spec
from generators.validator import corrupt_spec


def test_valid_scenario_passes_validation() -> None:
    scenario = build_scenario(
        index=1,
        seed=42,
        capability="lights",
        operation="turn_on",
        home_size=16,
    )
    scenario["utterance"] = "Turn on the kitchen light"
    spec = scenario_to_spec(scenario)
    assert validate_spec(spec) is None


def test_corrupted_entity_rejected() -> None:
    scenario = build_scenario(
        index=2,
        seed=42,
        capability="lights",
        operation="turn_on",
        home_size=16,
    )
    spec = scenario_to_spec(scenario)
    bad = corrupt_spec(spec, "wrong_entity")
    assert validate_spec(bad) == "unknown_canonical_entity"
