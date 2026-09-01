"""Benchmark runner tests with tiny in-memory fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.metrics import EvalRecord
from evals.runner import (
    BenchmarkRunResult,
    CaseExecutionResult,
    CaseTiming,
    dry_run_executor,
    load_output_case_ids,
    run_benchmark,
)
from evals.schema import EvalCase, load_eval_cases_jsonl


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


def _cases_jsonl(*case_ids: str) -> str:
    return "\n".join(
        json.dumps(
            {
                "case_id": case_id,
                "category": "simple_control",
                "home": "eval-home",
                "origin": "area_living_room",
                "turns": [f"Turn off the lights for {case_id}"],
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
        for case_id in case_ids
    )


def _read_output_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_dry_run_executor_never_actuates() -> None:
    case = _action_case("dry-001")
    result = dry_run_executor(case)
    assert result.record.ha_executed is False
    assert result.record.case_id == "dry-001"
    assert result.timing.total_ms >= 0.0


def test_run_benchmark_writes_eval_record_lines_with_timing(tmp_path: Path) -> None:
    cases = [_action_case("run-001"), _action_case("run-002")]
    output = tmp_path / "records.jsonl"

    def fake_executor(case: EvalCase) -> CaseExecutionResult:
        return CaseExecutionResult(
            record=EvalRecord(
                case_id=case.case_id,
                recorded_control_plan=case.expected_control_plan,
                recorded_candidate_entities=case.expected_candidate_entities,
                recorded_resolved_entities=case.expected_resolved_entities,
                ha_executed=False,
            ),
            timing=CaseTiming(total_ms=12.5),
        )

    summary = run_benchmark(cases, output, fake_executor, seed=7)
    lines = _read_output_lines(output)

    assert summary == BenchmarkRunResult(scored=2, skipped=0, warmup_runs=0, errors=0)
    assert [line["case_id"] for line in lines] == ["run-001", "run-002"]
    for line in lines:
        assert line["total_ms"] == 12.5
        EvalRecord.model_validate(line)


def test_run_benchmark_accepts_jsonl_input_path(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(_cases_jsonl("path-001"), encoding="utf-8")
    output = tmp_path / "records.jsonl"

    summary = run_benchmark(cases_path, output)
    lines = _read_output_lines(output)

    assert summary.scored == 1
    assert lines[0]["case_id"] == "path-001"
    assert lines[0]["ha_executed"] is False


def test_warmup_runs_first_and_is_not_written_to_output(tmp_path: Path) -> None:
    warmup = _action_case("warmup-only")
    scored = [_action_case("scored-001"), _action_case("scored-002")]
    output = tmp_path / "records.jsonl"
    call_log: list[str] = []

    def fake_executor(case: EvalCase) -> CaseExecutionResult:
        call_log.append(case.case_id)
        return CaseExecutionResult(
            record=EvalRecord(case_id=case.case_id, ha_executed=False),
            timing=CaseTiming(total_ms=1.0),
        )

    summary = run_benchmark(scored, output, fake_executor, warmup_count=2, warmup_case=warmup)

    assert summary.warmup_runs == 2
    assert call_log[:2] == ["warmup-only", "warmup-only"]
    assert call_log[2:] == ["scored-001", "scored-002"]
    assert load_output_case_ids(output) == {"scored-001", "scored-002"}


def test_resume_skips_existing_case_ids_without_duplicates(tmp_path: Path) -> None:
    cases = [_action_case("resume-001"), _action_case("resume-002"), _action_case("resume-003")]
    output = tmp_path / "records.jsonl"
    seen: set[str] = set()

    def fake_executor(case: EvalCase) -> CaseExecutionResult:
        if case.case_id in seen:
            msg = f"duplicate execution for {case.case_id}"
            raise AssertionError(msg)
        seen.add(case.case_id)
        return CaseExecutionResult(
            record=EvalRecord(case_id=case.case_id, ha_executed=False),
            timing=CaseTiming(total_ms=3.0),
        )

    first = run_benchmark(cases[:2], output, fake_executor)
    second = run_benchmark(cases, output, fake_executor)

    lines = _read_output_lines(output)
    assert first == BenchmarkRunResult(scored=2, skipped=0, warmup_runs=0, errors=0)
    assert second == BenchmarkRunResult(scored=1, skipped=2, warmup_runs=0, errors=0)
    assert [line["case_id"] for line in lines] == ["resume-001", "resume-002", "resume-003"]
    assert len(lines) == 3


def test_executor_error_records_failure_and_continues(tmp_path: Path) -> None:
    cases = [_action_case("ok-001"), _action_case("fail-002"), _action_case("ok-003")]
    output = tmp_path / "records.jsonl"

    def fake_executor(case: EvalCase) -> CaseExecutionResult:
        if case.case_id == "fail-002":
            msg = "model blew up"
            raise RuntimeError(msg)
        return CaseExecutionResult(
            record=EvalRecord(case_id=case.case_id, ha_executed=False),
            timing=CaseTiming(total_ms=4.0),
        )

    summary = run_benchmark(cases, output, fake_executor)
    lines = _read_output_lines(output)

    assert summary == BenchmarkRunResult(scored=3, skipped=0, warmup_runs=0, errors=1)
    assert [line["case_id"] for line in lines] == ["ok-001", "fail-002", "ok-003"]
    assert lines[1]["schema_failure"] is True
    assert lines[1]["executor_error"] == "model blew up"
    assert "total_ms" in lines[1]
    assert lines[0]["schema_failure"] is False
    assert lines[2]["schema_failure"] is False


def test_load_output_case_ids_empty_when_missing(tmp_path: Path) -> None:
    assert load_output_case_ids(tmp_path / "missing.jsonl") == set()


def test_load_eval_cases_jsonl_fixture_still_parses() -> None:
    text = _cases_jsonl("fixture-001")
    cases = load_eval_cases_jsonl(text)
    assert len(cases) == 1
    assert cases[0].case_id == "fixture-001"
