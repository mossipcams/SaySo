"""Controller dry-run executor tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals import executor as executor_module
from evals.executor import _HOME_GRAPH_PATH, controller_dry_run_executor
from evals.runner import dry_run_executor
from evals.schema import EvalCase


def _action_case(case_id: str, utterance: str = "Turn off the lights") -> EvalCase:
    return EvalCase.model_validate(
        {
            "case_id": case_id,
            "category": "simple_control",
            "home": "eval-home",
            "origin": "area_living_room",
            "turns": [utterance],
            "expected_control_plan": {
                "outcome": "action",
                "intent": "turn off the lights",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "state": "off",
            },
            "expected_candidate_entities": ["light.living_room_ceiling"],
            "expected_resolved_entities": ["light.living_room_ceiling"],
            "expected_outcome": "valid_action",
            "execution_allowed": True,
        },
    )


def test_dry_run_executor_leaves_control_plan_empty() -> None:
    case = _action_case("dry-empty-001")
    result = dry_run_executor(case)
    assert result.record.recorded_control_plan is None
    assert result.record.ha_executed is False


def test_controller_dry_run_executor_records_control_plan() -> None:
    case = _action_case("controller-plan-001")
    result = controller_dry_run_executor(case)
    assert result.record.recorded_control_plan is not None
    assert result.record.recorded_control_plan.get("outcome") == "query"


def test_controller_dry_run_executor_never_actuates() -> None:
    case = _action_case("controller-no-ha-001")
    result = controller_dry_run_executor(case)
    assert result.record.ha_executed is False
    assert result.record.executed_entities == []


def test_controller_dry_run_executor_records_candidates_with_default_limit() -> None:
    case = _action_case("controller-candidates-001", utterance="turn off the lights")
    result = controller_dry_run_executor(case)
    candidates = result.record.recorded_candidate_entities
    assert "light.living_room_ceiling" in candidates
    assert len(candidates) >= 2


def test_controller_dry_run_executor_sets_failure_fields_on_mismatch() -> None:
    case = _action_case("controller-failure-001")
    result = controller_dry_run_executor(case)
    assert result.record.failure_stage is not None
    assert result.record.failure_reason is not None


def test_controller_dry_run_executor_copies_model_telemetry_from_generation() -> None:
    case = _action_case("controller-telemetry-001")
    result = controller_dry_run_executor(case)
    timing = result.timing
    assert timing.prompt_tokens is not None
    assert timing.prompt_tokens > 0
    assert timing.completion_tokens == 1
    assert timing.model_id == "fake"
    assert timing.retrieve_ms is not None
    assert timing.retrieve_ms >= 0.0
    assert timing.plan_ms is not None
    assert timing.plan_ms >= 0.0


def test_controller_dry_run_executor_marked_non_live_at_definition() -> None:
    assert getattr(controller_dry_run_executor, "live_actuation", None) is False


def test_controller_dry_run_executor_caches_home_graph_between_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_calls = 0
    original_read_text = Path.read_text

    def counting_read_text(self: Path, *args: object, **kwargs: object) -> str:
        nonlocal read_calls
        if self == _HOME_GRAPH_PATH:
            read_calls += 1
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)
    executor_module._load_home_graph.cache_clear()

    controller_dry_run_executor(_action_case("cache-graph-a"))
    controller_dry_run_executor(_action_case("cache-graph-b"))

    assert read_calls == 1


def test_controller_dry_run_executor_reuses_resident_fake_runtime() -> None:
    controller_dry_run_executor(_action_case("resident-runtime-a"))
    first = executor_module._resident_fake_runtime_instance()
    controller_dry_run_executor(_action_case("resident-runtime-b"))
    second = executor_module._resident_fake_runtime_instance()
    assert first is second


def test_run_benchmark_records_control_plan_without_importing_main(
    tmp_path: Path,
) -> None:
    from evals.executor import controller_dry_run_executor as executor
    from evals.runner import run_benchmark

    case = _action_case("benchmark-no-main-001")
    output = tmp_path / "records.jsonl"

    run_benchmark([case], output, executor=executor, execute=False)

    lines = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record_lines = [line for line in lines if "case_id" in line]
    assert len(record_lines) == 1
    assert record_lines[0]["recorded_control_plan"] is not None
