"""Benchmark runner tests with tiny in-memory fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.config import is_benchmark_config_line
from evals.executor import controller_dry_run_executor
from evals.metrics import EvalRecord
from evals.runner import (
    BenchmarkRunResult,
    CaseExecutionResult,
    CaseExecutor,
    CaseTiming,
    _record_to_jsonl,
    dry_run_executor,
    gate_executor_for_live_safety,
    load_output_case_ids,
    mark_non_live_executor,
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


def _actuating_executor(*, ha_executed: bool = True) -> CaseExecutor:
    def executor(case: EvalCase) -> CaseExecutionResult:
        return CaseExecutionResult(
            record=EvalRecord(
                case_id=case.case_id,
                recorded_resolved_entities=case.expected_resolved_entities,
                executed_entities=case.expected_resolved_entities,
                ha_executed=ha_executed,
            ),
            timing=CaseTiming(total_ms=1.0),
        )

    return executor


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
    lines: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if is_benchmark_config_line(payload):
            continue
        lines.append(payload)
    return lines


def test_dry_run_executor_never_actuates() -> None:
    case = _action_case("dry-001")
    result = dry_run_executor(case)
    assert result.record.ha_executed is False
    assert result.record.case_id == "dry-001"
    assert result.timing.total_ms >= 0.0


def test_record_to_jsonl_includes_present_optional_timing_fields() -> None:
    record = EvalRecord(case_id="timing-001", ha_executed=False)
    timing = CaseTiming(
        total_ms=100.0,
        plan_ms=25.0,
        prompt_tokens=42,
        completion_tokens=3,
        model_id="fake",
    )
    line = _record_to_jsonl(record, timing)
    assert line["total_ms"] == 100.0
    assert line["plan_ms"] == 25.0
    assert line["prompt_tokens"] == 42
    assert line["completion_tokens"] == 3
    assert line["model_id"] == "fake"
    assert "retrieve_ms" not in line


def test_record_to_jsonl_includes_cold_start_when_true() -> None:
    record = EvalRecord(case_id="cold-001", ha_executed=False)
    timing = CaseTiming(total_ms=1.0)
    line = _record_to_jsonl(record, timing, cold_start=True)
    assert line["cold_start"] is True


def test_record_to_jsonl_omits_cold_start_when_not_true() -> None:
    record = EvalRecord(case_id="warm-001", ha_executed=False)
    timing = CaseTiming(total_ms=1.0)
    assert "cold_start" not in _record_to_jsonl(record, timing)
    assert "cold_start" not in _record_to_jsonl(record, timing, cold_start=False)
    assert "cold_start" not in _record_to_jsonl(record, timing, cold_start=None)


def test_run_benchmark_tags_first_scored_row_cold_start_by_default(
    tmp_path: Path,
) -> None:
    cases = [_action_case("cold-001"), _action_case("cold-002")]
    output = tmp_path / "records.jsonl"

    run_benchmark(cases, output)
    lines = _read_output_lines(output)

    assert lines[0]["cold_start"] is True
    assert "cold_start" not in lines[1]


def test_run_benchmark_warmup_count_suppresses_cold_start_tag(tmp_path: Path) -> None:
    warmup = _action_case("warmup-only")
    scored = [_action_case("scored-001"), _action_case("scored-002")]
    output = tmp_path / "records.jsonl"

    run_benchmark(
        scored,
        output,
        warmup_count=1,
        warmup_case=warmup,
    )
    lines = _read_output_lines(output)

    assert all("cold_start" not in line for line in lines)


def test_run_benchmark_resume_tags_only_first_new_row_cold_start(
    tmp_path: Path,
) -> None:
    cases = [
        _action_case("resume-cold-001"),
        _action_case("resume-cold-002"),
        _action_case("resume-cold-003"),
    ]
    output = tmp_path / "records.jsonl"

    run_benchmark(cases[:1], output)
    run_benchmark(cases, output)
    lines = _read_output_lines(output)

    assert [line["case_id"] for line in lines] == [
        "resume-cold-001",
        "resume-cold-002",
        "resume-cold-003",
    ]
    assert lines[0]["cold_start"] is True
    assert lines[1]["cold_start"] is True
    assert "cold_start" not in lines[2]


def test_run_benchmark_cold_start_false_in_config_tags_no_rows(
    tmp_path: Path,
) -> None:
    from evals.config import BenchmarkConfig

    cases = [_action_case("no-cold-001"), _action_case("no-cold-002")]
    output = tmp_path / "records.jsonl"
    config = BenchmarkConfig(cold_start=False)

    run_benchmark(cases, output, config=config)
    lines = _read_output_lines(output)

    assert all("cold_start" not in line for line in lines)


def test_run_benchmark_jsonl_includes_optional_timing_when_present(
    tmp_path: Path,
) -> None:
    case = _action_case("timing-jsonl-001")
    output = tmp_path / "records.jsonl"

    def timing_executor(eval_case: EvalCase) -> CaseExecutionResult:
        return CaseExecutionResult(
            record=EvalRecord(case_id=eval_case.case_id, ha_executed=False),
            timing=CaseTiming(
                total_ms=9.0,
                retrieve_ms=1.5,
                plan_ms=2.0,
                prompt_tokens=10,
                completion_tokens=1,
                model_id="test-model",
            ),
        )

    mark_non_live_executor(timing_executor)
    run_benchmark([case], output, timing_executor)
    line = _read_output_lines(output)[0]
    assert line["total_ms"] == 9.0
    assert line["retrieve_ms"] == 1.5
    assert line["plan_ms"] == 2.0
    assert line["prompt_tokens"] == 10
    assert line["completion_tokens"] == 1
    assert line["model_id"] == "test-model"


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

    summary = run_benchmark(
        cases,
        output,
        fake_executor,
        seed=7,
        execute=True,
        entity_allowlist=["light.living_room_ceiling"],
    )
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

    summary = run_benchmark(
        scored,
        output,
        fake_executor,
        warmup_count=2,
        warmup_case=warmup,
        execute=True,
        entity_allowlist=["light.living_room_ceiling"],
    )

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

    summary = run_benchmark(
        cases,
        output,
        fake_executor,
        execute=True,
        entity_allowlist=["light.living_room_ceiling"],
    )
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


def test_execute_flag_alone_does_not_actuate(tmp_path: Path) -> None:
    case = _action_case("live-execute-only")
    output = tmp_path / "records.jsonl"

    run_benchmark(
        [case],
        output,
        _actuating_executor(),
        execute=True,
        entity_allowlist=(),
    )

    assert _read_output_lines(output)[0]["ha_executed"] is False


def test_allowlist_alone_does_not_actuate(tmp_path: Path) -> None:
    case = _action_case("live-allowlist-only")
    output = tmp_path / "records.jsonl"

    run_benchmark(
        [case],
        output,
        _actuating_executor(),
        execute=False,
        entity_allowlist=["light.living_room_ceiling"],
    )

    assert _read_output_lines(output)[0]["ha_executed"] is False


def test_execute_and_allowlist_with_matching_entities_actuates(tmp_path: Path) -> None:
    case = _action_case("live-both-match")
    output = tmp_path / "records.jsonl"

    run_benchmark(
        [case],
        output,
        _actuating_executor(),
        execute=True,
        entity_allowlist=["light.living_room_ceiling"],
    )

    assert _read_output_lines(output)[0]["ha_executed"] is True


def test_execute_and_allowlist_blocks_out_of_allowlist_entity(tmp_path: Path) -> None:
    case = EvalCase.model_validate(
        {
            **_action_case("live-outside-allowlist").model_dump(),
            "expected_resolved_entities": ["light.bedroom_lamp"],
            "expected_candidate_entities": ["light.bedroom_lamp"],
        },
    )
    output = tmp_path / "records.jsonl"

    run_benchmark(
        [case],
        output,
        _actuating_executor(),
        execute=True,
        entity_allowlist=["light.living_room_ceiling"],
    )

    assert _read_output_lines(output)[0]["ha_executed"] is False


def test_gate_executor_skips_inner_executor_when_safeguards_fail() -> None:
    case = _action_case("count-match")
    outside = EvalCase.model_validate(
        {
            **_action_case("count-outside").model_dump(),
            "expected_resolved_entities": ["light.bedroom_lamp"],
            "expected_candidate_entities": ["light.bedroom_lamp"],
        },
    )
    allowlist = ["light.living_room_ceiling"]
    call_count = 0

    def counting_executor(eval_case: EvalCase) -> CaseExecutionResult:
        nonlocal call_count
        call_count += 1
        return _actuating_executor()(eval_case)

    gated_blocked = gate_executor_for_live_safety(
        counting_executor,
        execute=False,
        entity_allowlist=allowlist,
    )
    assert gated_blocked(case).record.ha_executed is False
    assert call_count == 0

    gated_empty_allowlist = gate_executor_for_live_safety(
        counting_executor,
        execute=True,
        entity_allowlist=(),
    )
    assert gated_empty_allowlist(case).record.ha_executed is False
    assert call_count == 0

    gated_outside = gate_executor_for_live_safety(
        counting_executor,
        execute=True,
        entity_allowlist=allowlist,
    )
    assert gated_outside(outside).record.ha_executed is False
    assert call_count == 0

    gated_live = gate_executor_for_live_safety(
        counting_executor,
        execute=True,
        entity_allowlist=allowlist,
    )
    assert gated_live(case).record.ha_executed is True
    assert call_count == 1


def test_gate_executor_for_live_safety_directly() -> None:
    case = _action_case("gate-direct")
    gated = gate_executor_for_live_safety(
        _actuating_executor(),
        execute=True,
        entity_allowlist=["light.living_room_ceiling"],
    )

    assert gated(case).record.ha_executed is True

    blocked = gate_executor_for_live_safety(_actuating_executor(), execute=True)
    assert blocked(case).record.ha_executed is False


def test_run_benchmark_runs_controller_dry_run_when_execute_false(
    tmp_path: Path,
) -> None:
    case = _action_case("controller-default-001")
    output = tmp_path / "records.jsonl"

    summary = run_benchmark(
        [case],
        output,
        controller_dry_run_executor,
        execute=False,
    )
    lines = _read_output_lines(output)

    assert summary == BenchmarkRunResult(scored=1, skipped=0, warmup_runs=0, errors=0)
    assert lines[0]["ha_executed"] is False
    assert lines[0]["recorded_control_plan"] is not None


def test_gate_executor_passes_through_non_live_executor_when_execute_false() -> None:
    case = _action_case("gate-non-live")
    call_count = 0

    def counting_wrapper(eval_case: EvalCase) -> CaseExecutionResult:
        nonlocal call_count
        call_count += 1
        return controller_dry_run_executor(eval_case)

    mark_non_live_executor(counting_wrapper)
    gated = gate_executor_for_live_safety(
        counting_wrapper,
        execute=False,
        entity_allowlist=["light.living_room_ceiling"],
    )

    result = gated(case)

    assert call_count == 1
    assert result.record.ha_executed is False
    assert result.record.recorded_control_plan is not None
