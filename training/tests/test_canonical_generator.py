"""Focused tests for canonical generator defects (independent scenarios)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from adapters.schema import v2_openai_tools
from generators.pipeline import generate_row
from generators.config import GeneratorConfig
from generators.deduplication import DuplicateTracker
from generators.manifest import build_manifest, dataset_hash
from generators.planning import build_plan, verify_feasible
from generators.pipeline import run_generation
from generators.gold import gold_matches_family
from generators.scenarios import area as area_scenarios
from generators.rendering import render_example, scenario_to_spec
from generators.scenarios import build_scenario
from generators.tools import namespaced_tool_name, production_catalog
from generators.validation import validate_row, validate_serialized_row

SMOKE_CONFIG = ROOT / "configs" / "generation" / "smoke.yaml"


def _small_config(**overrides) -> GeneratorConfig:
    base = GeneratorConfig.from_yaml(SMOKE_CONFIG, repo_root=REPO)
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_yaml_config_loads() -> None:
    cfg = GeneratorConfig.from_yaml(SMOKE_CONFIG, repo_root=REPO)
    assert cfg.count == 120
    assert abs(sum(cfg.allocations.values()) - 1.0) < 0.02
    # 7168, not the old 5120: until count_row_tokens was repaired the budget
    # gated nothing, and production-shaped rows measure p50 ~5.4k / max ~6.6k.
    assert cfg.token_budget == 7168


def test_yaml_config_requires_repo_root() -> None:
    with pytest.raises(ValueError, match="repo_root is required"):
        GeneratorConfig.from_yaml(SMOKE_CONFIG)


def test_impossible_quota_fails_before_loop() -> None:
    with pytest.raises(ValueError, match="area scenario minimums"):
        verify_feasible(
            10,
            {"ordinary": 1.0},
            near_duplicate_limit=4,
            area_required={"implicit_satellite_area": 20},
            grounding_variants=0,
            grounding_rate=0.0,
        )


def test_generation_is_deterministic() -> None:
    cfg = _small_config(count=40, seed=3)
    first = run_generation(cfg)["rows"]
    second = run_generation(cfg)["rows"]
    assert first == second
    assert dataset_hash(first) == dataset_hash(second)


def test_production_catalog_is_answer_independent() -> None:
    scenario = build_scenario(
        index=1, seed=42, capability="lights", operation="turn_on", home_size=16,
    )
    spec = scenario_to_spec(scenario)
    spec["utterance"] = "Turn on the kitchen light"
    row = render_example(spec)
    offered = {t["function"]["name"] for t in row["tools"]}
    catalog = {t["function"]["name"] for t in production_catalog(spec["home"])}
    assert offered == catalog
    assert namespaced_tool_name("GetDateTime") in offered


def test_datetime_family_labels_get_datetime() -> None:
    scenario = build_scenario(
        index=6,
        seed=19,
        capability="datetime",
        operation="query_time",
        home_size=8,
        family="datetime",
    )
    expected = scenario["expected"]
    assert gold_matches_family(expected, "datetime")
    assert expected["calls"][0]["name"] == "GetDateTime"
    spec = scenario_to_spec(scenario)
    spec["utterance"] = "what time is it"
    row = render_example(spec)
    calls = [
        c["function"]["name"]
        for m in row["messages"]
        for c in (m.get("tool_calls") or [])
    ]
    assert calls == [namespaced_tool_name("GetDateTime")]


def test_status_reads_use_get_live_context() -> None:
    scenario = build_scenario(
        index=2, seed=7, capability="locks", operation="query_state", home_size=16,
    )
    spec = scenario_to_spec(scenario)
    spec["utterance"] = "Is the front door lock locked?"
    assert spec["expected"]["kind"] == "status"
    assert spec["expected"]["calls"][0]["name"] == "GetLiveContext"
    row = render_example(spec)
    calls = [
        c["function"]["name"]
        for m in row["messages"]
        for c in (m.get("tool_calls") or [])
    ]
    assert calls == [namespaced_tool_name("GetLiveContext")]


def test_genuine_ambiguity_clarifies() -> None:
    scenario = build_scenario(
        index=3,
        seed=11,
        capability="lights",
        operation="turn_on",
        home_size=32,
        family="clarify",
    )
    spec = scenario_to_spec(scenario)
    spec["utterance"] = "Turn on the light in the living room"
    assert spec["expected"]["kind"] == "no_action"
    assert spec["expected"].get("response") == "clarify"
    assert not spec["expected"].get("calls")


def test_exclusion_omits_forbidden_target() -> None:
    scenario = build_scenario(
        index=4,
        seed=13,
        capability="lights",
        operation="turn_off",
        home_size=64,
        targeting="multiple",
        robustness="exclusion",
    )
    spec = scenario_to_spec(scenario)
    calls = spec["expected"].get("calls") or []
    assert len(calls) >= 2
    excluded = set(spec.get("excluded_names") or [])
    for call in calls:
        name = call.get("arguments", {}).get("name")
        if name:
            assert name not in excluded


def test_withheld_capability_omits_tool() -> None:
    scenario = build_scenario(
        index=5,
        seed=17,
        capability="climate",
        operation="set_temperature",
        home_size=8,
        family="unavailable",
    )
    spec = scenario_to_spec(scenario)
    spec["utterance"] = "Set the thermostat to 72"
    assert spec["expected"]["response"] == "unsupported"
    withheld = spec["expected"].get("unavailable_tools") or spec.get("removed_tools") or []
    assert withheld == ["HassClimateSetTemperature"]
    assert gold_matches_family(spec["expected"], "unavailable")
    assert not gold_matches_family({**spec["expected"], "unavailable_tools": []}, "unavailable")
    row = render_example(spec)
    offered = {t["function"]["name"] for t in row["tools"]}
    for tool in withheld:
        assert namespaced_tool_name(tool) not in offered


def test_clarify_family_never_labels_action() -> None:
    for seed in range(20):
        scenario = build_scenario(
            index=seed,
            seed=seed,
            capability="lights",
            operation="turn_on",
            home_size=16,
            family="clarify",
        )
        expected = scenario["expected"]
        assert gold_matches_family(expected, "clarify")
        assert expected["kind"] == "no_action"
        assert expected["response"] == "clarify"


def test_follow_up_row_shape() -> None:
    cfg = _small_config(count=40, seed=77)
    slot = {
        "index": 0,
        "family": "follow_up",
        "robustness": "ambiguity",
        "capability": "lights",
        "operation": "turn_on",
        "home_size": 16,
    }
    rng = __import__("random").Random(77)
    row, reason = generate_row(
        slot,
        cfg,
        rng,
        excluded=set(),
        dup_tracker=DuplicateTracker(near_limit=4),
    )
    assert reason is None, reason
    assert row is not None
    users = [m["content"] for m in row["messages"] if m["role"] == "user"]
    assert len(users) >= 2
    assistants = [m for m in row["messages"] if m["role"] == "assistant"]
    first_assistant = assistants[0]
    assert first_assistant.get("train_on_turn") is True
    assert not first_assistant.get("tool_calls")
    assert first_assistant["content"].startswith("Did you mean ")
    trained_calls = [
        c
        for m in row["messages"]
        if m.get("role") == "assistant" and m.get("train_on_turn")
        for c in (m.get("tool_calls") or [])
    ]
    assert trained_calls
    args = json.loads(trained_calls[-1]["function"]["arguments"])
    gold_name = row["metadata"]["expected_target_names"][0]
    assert args.get("name") == gold_name
    assert gold_name in first_assistant["content"]


def test_correction_row_shape() -> None:
    cfg = _small_config(count=40, seed=88)
    for attempt in range(30):
        slot = {
            "index": attempt,
            "family": "correction",
            "robustness": "ordinary",
            "capability": "lights",
            "operation": "turn_on",
            "home_size": 16,
        }
        rng = __import__("random").Random(88 + attempt)
        row, reason = generate_row(
            slot,
            cfg,
            rng,
            excluded=set(),
            dup_tracker=DuplicateTracker(near_limit=4),
        )
        if reason == "family_mismatch":
            continue
        assert reason is None, reason
        assert row is not None
        assistants = [m for m in row["messages"] if m["role"] == "assistant"]
        assert assistants[0].get("train_on_turn") is False
        assert assistants[0].get("tool_calls")
        tool_msgs = [m for m in row["messages"] if m["role"] == "tool"]
        error_payload = json.loads(tool_msgs[0]["content"])
        assert "error" in error_payload and error_payload["error"].get("code")
        trained = next(m for m in assistants if m.get("train_on_turn") and m.get("tool_calls"))
        args = json.loads(trained["tool_calls"][0]["function"]["arguments"])
        gold_name = row["metadata"]["expected_target_names"][0]
        assert args.get("name") == gold_name
        wrong_args = json.loads(assistants[0]["tool_calls"][0]["function"]["arguments"])
        assert wrong_args.get("name") != gold_name
        return
    pytest.fail("could not materialize a correction row in 30 attempts")


def test_clarify_assistant_names_candidates() -> None:
    scenario = build_scenario(
        index=0,
        seed=1,
        capability="lights",
        operation="turn_on",
        home_size=16,
        family="clarify",
    )
    expected = scenario["expected"]
    names = expected.get("candidates") or []
    assert len(names) >= 2
    spec = scenario_to_spec(scenario)
    spec["utterance"] = "Turn on the light"
    row = render_example(spec)
    text = row["messages"][-1]["content"]
    assert text.startswith("Did you mean ")
    assert " or " in text
    for name in names[:2]:
        assert name in text
    assert not any(message.get("tool_calls") for message in row["messages"])


def test_absence_family_never_labels_action_or_script_clarify() -> None:
    for seed in range(20):
        scenario = build_scenario(
            index=seed,
            seed=seed,
            capability="lights",
            operation="turn_on",
            home_size=16,
            family="absence",
        )
        expected = scenario["expected"]
        assert gold_matches_family(expected, "absence")
        assert expected["kind"] == "no_action"
        assert expected["response"] in ("area_unavailable", "device_absent")


def test_unsupported_family_keeps_tool_and_labels_no_action() -> None:
    for seed in range(20):
        scenario = build_scenario(
            index=seed,
            seed=seed,
            capability="media_players",
            operation="turn_on",
            home_size=16,
            family="unsupported",
        )
        expected = scenario["expected"]
        assert gold_matches_family(expected, "unsupported")
        assert expected["kind"] == "no_action"
        assert expected["response"] == "device_unsupported"
        assert not expected.get("unavailable_tools")
        spec = scenario_to_spec(scenario)
        spec["utterance"] = "Turn on the TV"
        row = render_example(spec)
        offered = {t["function"]["name"] for t in row["tools"]}
        assert namespaced_tool_name("HassTurnOn") in offered
        assert not any(message.get("tool_calls") for message in row["messages"])


def test_manifest_reconciles_row_count() -> None:
    cfg = _small_config(count=30, seed=5)
    result = run_generation(cfg)
    manifest = result["manifest"]
    assert manifest["final_row_count"] == len(result["rows"])
    assert sum(manifest["outcome_coverage"].values()) == len(result["rows"])


def test_area_rows_use_production_format() -> None:
    cfg = _small_config(count=120, seed=3)
    result = run_generation(cfg)
    rows = result["rows"]
    area_rows = [r for r in rows if r["metadata"].get("area_scenario")]
    assert area_rows, "expected area scenario rows in smoke recipe"
    for row in area_rows:
        assert "target_area_source:" in row["messages"][0]["content"]
    achieved = result["stats"]["area_distribution"]["achieved"]
    assert achieved == {name: 1 for name in area_scenarios.SCENARIOS}


def test_area_distribution_fails_closed_on_zero_rows() -> None:
    plan = area_scenarios.load_distribution()
    with pytest.raises(ValueError, match="resolves to zero rows"):
        area_scenarios.assert_required_counts_feasible(plan, 120)


def test_no_eval_utterance_contamination() -> None:
    cfg = _small_config(count=50, seed=21)
    rows = run_generation(cfg)["rows"]
    repo_path = REPO
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    from evals.cases import excluded_train_utterances

    held = {u.casefold() for u in excluded_train_utterances()}
    for row in rows:
        user = next(m["content"] for m in row["messages"] if m["role"] == "user")
        assert user.casefold() not in held


def test_cli_smoke_config_runs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "generators.cli",
            "--config",
            str(SMOKE_CONFIG),
            "--dry-run",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["accepted"] == 120


def test_exhausted_retries_would_fail() -> None:
    cfg = _small_config(count=200, seed=1, max_attempts_multiplier=1)
    with pytest.raises(RuntimeError, match="failed to meet allocation"):
        run_generation(cfg)
