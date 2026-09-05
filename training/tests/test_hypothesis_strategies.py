"""Tests for hypothesis strategies (test-only, not production sampling)."""

from __future__ import annotations

from hypothesis import given, settings

from generators.hypothesis_strategies import (
    capability_operation_strategy,
    corrupted_spec_strategy,
    home_strategy,
    invalid_brightness_strategy,
    scenario_strategy,
    valid_brightness_strategy,
    valid_spec_strategy,
)
from generators.validate import validate_spec


def test_strategy_home_produces_entities() -> None:
    home = home_strategy().example()
    assert len(home["entities"]) >= 8


def test_brightness_bounds() -> None:
    assert 0 <= valid_brightness_strategy().example() <= 100
    assert invalid_brightness_strategy().example() not in range(0, 101)


def test_capability_operation_pairs() -> None:
    cap, op = capability_operation_strategy().example()
    assert cap and op


@given(spec=valid_spec_strategy())
@settings(max_examples=40, deadline=None)
def test_valid_specs_pass_validation(spec: dict) -> None:
    assert validate_spec(spec) is None


@given(spec=corrupted_spec_strategy("wrong_entity"))
@settings(max_examples=20, deadline=None)
def test_corrupted_entity_specs_fail_validation(spec: dict) -> None:
    assert validate_spec(spec) == "unknown_canonical_entity"


@given(spec=corrupted_spec_strategy("wrong_tool"))
@settings(max_examples=20, deadline=None)
def test_corrupted_tool_specs_fail_validation(spec: dict) -> None:
    assert validate_spec(spec) is not None


@given(scenario=scenario_strategy(capability="lights", operation="turn_on"))
@settings(max_examples=20, deadline=None)
def test_scenario_semantic_id_ignores_attempt(scenario: dict) -> None:
    from generators.scenarios import semantic_id

    base = semantic_id(scenario)
    mutated = dict(scenario)
    mutated["attempt"] = scenario.get("attempt", 0) + 99
    mutated["scenario_index"] = scenario.get("scenario_index", 0) + 99
    assert semantic_id(mutated) == base
