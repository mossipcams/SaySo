"""Area grounding rows: production format, correct labels, enforced quotas."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from generators.scenarios import area as area_scenarios
from generators.config import GeneratorConfig
from generators.pipeline import run_generation


def _row(scenario: str, index: int = 0) -> dict:
    return area_scenarios.build_row(scenario, index, seed=20260917)


def _calls(row: dict) -> list[tuple[str, dict]]:
    return [
        (call["function"]["name"], json.loads(call["function"]["arguments"]))
        for message in row["messages"]
        for call in message.get("tool_calls") or []
    ]


def _plan(minimum: int) -> dict:
    plan = area_scenarios.load_distribution()
    for rule in plan["scenarios"].values():
        rule.update(per_1000=0, min=minimum)
    return plan


def test_every_scenario_renders_the_production_area_block_only() -> None:
    for scenario in area_scenarios.SCENARIOS:
        system = _row(scenario)["messages"][0]["content"]
        assert "You are in area" not in system
        assert "\nArea context:\nsatellite_area: " in system
        assert "\ntarget_area_source: " in system


def test_implicit_request_resolves_to_the_satellite_area() -> None:
    row = _row("implicit_satellite_area")
    context = row["metadata"]["area_context"]
    assert context["target_area_source"] == "satellite"
    assert context["target_area"] == context["satellite_area"]
    assert _calls(row) == [
        ("intent__HassTurnOff", {"area": context["satellite_area"], "domain": ["light"]})
    ]


@pytest.mark.parametrize(
    "scenario", ["explicit_area", "explicit_area_conflicts_with_satellite", "area_alias"]
)
def test_explicit_area_overrides_the_satellite_area(scenario: str) -> None:
    row = _row(scenario)
    context = row["metadata"]["area_context"]
    assert context["target_area_source"] == "explicit"
    assert context["target_area"] != context["satellite_area"]
    ((_, arguments),) = _calls(row)
    assert arguments["area"] == context["target_area"]


def test_an_area_alias_resolves_to_its_canonical_area() -> None:
    row = _row("area_alias")
    user = row["messages"][1]["content"].casefold()
    target = row["metadata"]["area_context"]["target_area"]
    assert target.casefold() not in user


def test_duplicate_names_across_areas_require_clarification() -> None:
    row = _row("duplicate_name_clarification")
    home_lamps = [
        line for line in row["messages"][0]["content"].splitlines() if line == "- names: Lamp"
    ]
    assert len(home_lamps) == 2
    assert _calls(row) == []
    assert row["messages"][-1]["content"].startswith("Which Lamp do you mean")


def test_missing_satellite_area_renders_none() -> None:
    row = _row("missing_satellite_area")
    assert row["metadata"]["area_context"] == {
        "satellite_area": None,
        "target_area": None,
        "target_area_source": "none",
    }
    assert "\nsatellite_area: none\ntarget_area: none\ntarget_area_source: none" in (
        row["messages"][0]["content"]
    )


def test_multi_area_and_exclusion_rows_are_complete() -> None:
    multi = _row("multi_action_across_areas")
    assert multi["metadata"]["area_context"]["target_area_source"] == "multiple"
    assert len(_calls(multi)) == 2
    exclusion = _row("exclusion_within_area")
    kept = exclusion["metadata"]["excluded_names"][0]
    assert len(_calls(exclusion)) == 2
    assert all(arguments.get("name") != kept for _, arguments in _calls(exclusion))


def test_area_rows_are_deterministic_for_a_fixed_seed() -> None:
    required = {name: 2 for name in area_scenarios.SCENARIOS}
    first = area_scenarios.build_rows(required, seed=11)
    second = area_scenarios.build_rows(required, seed=11)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert json.dumps(first, sort_keys=True) != json.dumps(
        area_scenarios.build_rows(required, seed=12), sort_keys=True
    )


def test_generation_is_deterministic_and_carries_area_rows(tmp_path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan(1)))
    root = Path(__file__).resolve().parents[1]
    config = GeneratorConfig.from_yaml(root / "configs/generation/smoke.yaml", repo_root=root.parent)
    config.count = 120
    config.seed = 31
    config.area_distribution_path = plan_path
    first = run_generation(config)
    second = run_generation(copy.deepcopy(config))
    assert json.dumps(first["rows"], sort_keys=True) == json.dumps(second["rows"], sort_keys=True)
    assert len(first["rows"]) == 120
    achieved = first["stats"]["area_distribution"]["achieved"]
    assert achieved == {name: 1 for name in area_scenarios.SCENARIOS}


def test_zero_row_scenario_fails_before_generation() -> None:
    plan = area_scenarios.load_distribution()
    with pytest.raises(ValueError, match="resolves to zero rows"):
        area_scenarios.assert_required_counts_feasible(plan, 120)


def test_a_short_area_scenario_fails_the_build() -> None:
    plan = _plan(2)
    rows = area_scenarios.build_rows(area_scenarios.required_counts(plan, 100), seed=5)
    area_scenarios.validate_distribution(rows, plan, 100)
    # Keep one of the two required area_alias rows.
    short = [row for row in rows if row["metadata"]["area_scenario"] != "area_alias"]
    short.append(next(row for row in rows if row["metadata"]["area_scenario"] == "area_alias"))
    with pytest.raises(RuntimeError, match="area_alias"):
        area_scenarios.validate_distribution(short, plan, 100)


def test_legacy_area_prompts_fail_the_build() -> None:
    row = _row("explicit_area")
    row["messages"][0]["content"] += "\nYou are in area Kitchen and all generic commands"
    with pytest.raises(RuntimeError, match="production area format"):
        area_scenarios.validate_distribution([row], None, 1)


def test_too_few_satellite_area_rows_fail_the_build() -> None:
    plan = _plan(0)
    plan["min_satellite_area_share_of_actionable_rows"] = 0.9
    rows = [_row("explicit_area"), _row("explicit_area", 1)]
    for row in rows:
        row["metadata"]["area_context"]["satellite_area"] = None
    with pytest.raises(RuntimeError, match="satellite-area share"):
        area_scenarios.validate_distribution(rows, plan, 2)
