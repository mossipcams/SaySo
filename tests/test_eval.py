"""Tests for offline evaluation cases and scorer."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from custom_components.sayso.client import (
    LlamaCppClient,
    ToolCall,
    build_chat_completions_payload,
    serialize_chat_completions_payload,
)
from custom_components.sayso.diagnostics import BoundaryFailureCode
from custom_components.sayso.schema import (
    ToolArgumentFailureCode,
    ToolArgumentValidationError,
)
from evals.live import LiveLatencyConfig, measure_live_latency_once, run_live_latency_benchmark
from evals.compare import (
    compare_eval_reports,
    format_comparison_json,
    format_comparison_markdown,
    load_baseline,
    validate_release_report,
)
from evals.metrics import (
    build_live_latency_report,
    compare_metric_reports,
    compute_latency_percentiles,
    derive_latency_tolerance_ms,
    validate_live_latency_report,
    validate_metrics_report,
)
from evals.runner import (
    EvalRecord,
    build_release_report,
    load_cases,
    run_eval,
    run_eval_with_metrics,
)
from evals.scorer import EvalActual, score_case

CASES_PATH = Path(__file__).resolve().parents[1] / "evals" / "cases" / "v1.json"


def _tool_call(name: str, arguments: dict, *, call_id: str = "call_1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def test_cases_file_exists_and_has_unique_ids() -> None:
    """The versioned case set loads with independent, unique IDs."""
    cases = load_cases(CASES_PATH)
    assert cases.version == 1
    assert len(cases.cases) >= 7
    ids = [case.id for case in cases.cases]
    assert len(ids) == len(set(ids))
    for case_id in ids:
        assert case_id.startswith("sayso-eval-v1-")
        assert "datasets" not in case_id


def test_cases_cover_required_categories() -> None:
    """Cases include core control, safety, query, multi-call, failure, and follow-up."""
    cases = load_cases(CASES_PATH)
    categories = {case.category for case in cases.cases}
    required = {
        "core_control",
        "safety_ambiguity",
        "query",
        "multi_call",
        "failure",
        "follow_up",
    }
    assert required.issubset(categories)


def test_score_correct_tool_name_and_args() -> None:
    """Scorer passes when tool name and arguments match exactly."""
    cases = load_cases(CASES_PATH)
    case = next(c for c in cases.cases if c.scenario == "correct_tool_args")
    actual = EvalActual(
        tool_calls=[
            _tool_call(
                "HassTurnOn",
                {"name": "Living Room", "domain": "light"},
            )
        ]
    )
    result = score_case(case, actual)
    assert result.passed is True
    assert result.checks["tool_name"].passed is True
    assert result.checks["tool_args"].passed is True


def test_score_wrong_tool() -> None:
    """Scorer fails when the model selects the wrong tool."""
    cases = load_cases(CASES_PATH)
    case = next(c for c in cases.cases if c.scenario == "wrong_tool")
    actual = EvalActual(
        tool_calls=[_tool_call("HassFanSet", {"name": "Kitchen Fan"})]
    )
    result = score_case(case, actual)
    assert result.passed is False
    assert result.checks["wrong_tool"].passed is False


def test_score_invalid_call() -> None:
    """Scorer detects invalid tool arguments."""
    cases = load_cases(CASES_PATH)
    case = next(c for c in cases.cases if c.scenario == "invalid_call")
    actual = EvalActual(
        tool_calls=[_tool_call("HassTurnOn", {"unexpected": "field"})],
        validation_errors=[
            ToolArgumentValidationError(
                code=ToolArgumentFailureCode.SCHEMA_MISMATCH,
                message="Unexpected argument(s): ['unexpected']",
                tool_name="HassTurnOn",
            )
        ],
        boundary_code=BoundaryFailureCode.SCHEMA_MISMATCH,
    )
    result = score_case(case, actual)
    assert result.passed is True
    assert result.checks["invalid_call"].passed is True


def test_score_clarification() -> None:
    """Scorer passes when the model asks for clarification without tool calls."""
    cases = load_cases(CASES_PATH)
    case = next(c for c in cases.cases if c.scenario == "clarification")
    actual = EvalActual(
        spoken="Which light did you mean, the kitchen or the bedroom?"
    )
    result = score_case(case, actual)
    assert result.passed is True
    assert result.checks["clarification"].passed is True


def test_score_multi_call_order() -> None:
    """Scorer verifies ordered multi-tool batches."""
    cases = load_cases(CASES_PATH)
    case = next(c for c in cases.cases if c.scenario == "multi_call_order")
    actual = EvalActual(
        tool_calls=[
            _tool_call("HassTurnOff", {"name": "Living Room"}, call_id="call_1"),
            _tool_call("HassTurnOn", {"name": "Kitchen Light"}, call_id="call_2"),
        ]
    )
    result = score_case(case, actual)
    assert result.passed is True
    assert result.checks["tool_order"].passed is True


def test_score_partial_failure() -> None:
    """Scorer accepts partial execution failure with a final spoken result."""
    cases = load_cases(CASES_PATH)
    case = next(c for c in cases.cases if c.scenario == "partial_failure")
    actual = EvalActual(
        tool_calls=[
            _tool_call("HassTurnOn", {"name": "Living Room", "domain": "light"}),
            _tool_call("HassTurnOn", {"name": "Missing Device", "domain": "light"}),
        ],
        execution_failures=["Missing Device"],
        boundary_code=BoundaryFailureCode.TOOL_EXECUTION_FAILED,
        spoken="I couldn't reach Missing Device, but turned on the living room light.",
    )
    result = score_case(case, actual)
    assert result.passed is True
    assert result.checks["partial_failure"].passed is True


def test_score_final_spoken_result() -> None:
    """Scorer verifies the final assistant spoken response."""
    cases = load_cases(CASES_PATH)
    case = next(c for c in cases.cases if c.scenario == "final_spoken")
    actual = EvalActual(
        tool_calls=[
            _tool_call("HassTurnOn", {"name": "Living Room", "domain": "light"}),
        ],
        spoken="The living room light is on.",
    )
    result = score_case(case, actual)
    assert result.passed is True
    assert result.checks["spoken_result"].passed is True


def test_run_eval_produces_deterministic_json() -> None:
    """Runner emits byte-stable JSON for the same inputs."""
    cases = load_cases(CASES_PATH)
    actuals = {
        case.id: EvalActual(
            tool_calls=[
                _tool_call("HassTurnOn", {"name": "Living Room", "domain": "light"}),
            ],
            spoken="The living room light is on.",
        )
        for case in cases.cases
        if case.scenario in {"correct_tool_args", "final_spoken"}
    }
    first = run_eval(cases, actuals)
    second = run_eval(cases, actuals)
    first_bytes = json.dumps(first, sort_keys=True, separators=(",", ":")).encode()
    second_bytes = json.dumps(second, sort_keys=True, separators=(",", ":")).encode()
    assert first_bytes == second_bytes
    assert first["version"] == 1
    assert "summary" in first
    assert "results" in first


def test_run_eval_marks_missing_actuals() -> None:
    """Runner reports skipped cases when no actual is supplied."""
    cases = load_cases(CASES_PATH)
    report = run_eval(cases, {})
    assert report["summary"]["skipped"] == len(cases.cases)
    assert all(item["status"] == "skipped" for item in report["results"])


def _recorded_fixture_records() -> tuple[dict[str, EvalRecord], dict[str, EvalRecord]]:
    """Baseline and improved recordings for the same eval cases."""
    cases = load_cases(CASES_PATH)
    core_case = next(c for c in cases.cases if c.scenario == "correct_tool_args")
    wrong_case = next(c for c in cases.cases if c.scenario == "wrong_tool")
    invalid_case = next(c for c in cases.cases if c.scenario == "invalid_call")
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "turn on the living room light"}],
        "temperature": 0,
        "max_tokens": 160,
    }
    request_bytes = len(serialize_chat_completions_payload(payload))
    baseline = {
        core_case.id: EvalRecord(
            actual=EvalActual(
                tool_calls=[
                    _tool_call(
                        "HassTurnOn",
                        {"name": "Living Room", "domain": "light"},
                    )
                ]
            ),
            request_payload=payload,
            request_bytes=request_bytes,
            prompt_tokens=120,
            latency_ms=100.0,
            confidently_routed=True,
        ),
        wrong_case.id: EvalRecord(
            actual=EvalActual(
                tool_calls=[_tool_call("HassFanSet", {"name": "Kitchen Fan"})]
            ),
            request_payload=payload,
            request_bytes=request_bytes + 50,
            prompt_tokens=140,
            latency_ms=200.0,
        ),
        invalid_case.id: EvalRecord(
            actual=EvalActual(
                tool_calls=[_tool_call("HassTurnOn", {"unexpected": "field"})],
                validation_errors=[
                    ToolArgumentValidationError(
                        code=ToolArgumentFailureCode.SCHEMA_MISMATCH,
                        message="Unexpected argument(s): ['unexpected']",
                        tool_name="HassTurnOn",
                    )
                ],
                boundary_code=BoundaryFailureCode.SCHEMA_MISMATCH,
            ),
            request_payload=payload,
            request_bytes=request_bytes + 25,
            prompt_tokens=None,
            latency_ms=300.0,
        ),
    }
    improved_payload = dict(payload)
    improved_payload["messages"] = [
        {"role": "user", "content": "turn on the living room light please"}
    ]
    improved_bytes = len(serialize_chat_completions_payload(improved_payload))
    improved = {
        core_case.id: EvalRecord(
            actual=EvalActual(
                tool_calls=[
                    _tool_call(
                        "HassTurnOn",
                        {"name": "Living Room", "domain": "light"},
                    )
                ]
            ),
            request_payload=improved_payload,
            request_bytes=improved_bytes,
            prompt_tokens=80,
            latency_ms=80.0,
            confidently_routed=True,
        ),
        wrong_case.id: EvalRecord(
            actual=EvalActual(
                tool_calls=[
                    _tool_call(
                        "HassTurnOn",
                        {"name": "Living Room", "domain": "light"},
                    )
                ]
            ),
            request_payload=improved_payload,
            request_bytes=improved_bytes,
            prompt_tokens=85,
            latency_ms=120.0,
        ),
        invalid_case.id: EvalRecord(
            actual=EvalActual(
                tool_calls=[_tool_call("HassTurnOn", {"unexpected": "field"})],
                validation_errors=[
                    ToolArgumentValidationError(
                        code=ToolArgumentFailureCode.SCHEMA_MISMATCH,
                        message="Unexpected argument(s): ['unexpected']",
                        tool_name="HassTurnOn",
                    )
                ],
                boundary_code=BoundaryFailureCode.SCHEMA_MISMATCH,
            ),
            request_payload=improved_payload,
            request_bytes=improved_bytes,
            prompt_tokens=86,
            latency_ms=160.0,
        ),
    }
    return baseline, improved


def test_run_eval_with_metrics_computes_request_tool_and_latency_metrics() -> None:
    """Runner aggregates serialized bytes, prompt tokens, tool quality, and latency."""
    cases = load_cases(CASES_PATH)
    baseline, _ = _recorded_fixture_records()
    report = run_eval_with_metrics(cases, baseline)
    metrics = report["metrics"]

    assert metrics["serialized_request_bytes"]["total"] == sum(
        record.request_bytes for record in baseline.values()
    )
    assert metrics["prompt_tokens"]["total"] == 260
    assert metrics["prompt_tokens"]["missing_usage_count"] == 1
    assert metrics["tool_accuracy"] == pytest.approx(0.5)
    assert metrics["invalid_call_rate"] == pytest.approx(1 / 3)
    assert metrics["latency_ms"]["p50"] == 200.0
    assert metrics["latency_ms"]["p95"] == 290.0


def test_compare_metric_reports_from_same_fixtures() -> None:
    """Baseline and improved reports compare with signed deltas."""
    cases = load_cases(CASES_PATH)
    baseline_records, improved_records = _recorded_fixture_records()
    baseline_report = run_eval_with_metrics(cases, baseline_records)
    improved_report = run_eval_with_metrics(cases, improved_records)
    comparison = compare_metric_reports(
        baseline_report["metrics"],
        improved_report["metrics"],
    )

    assert comparison["delta"]["serialized_request_bytes_total"] < 0
    assert comparison["delta"]["prompt_tokens_total"] < 0
    assert comparison["delta"]["tool_accuracy"] > 0
    assert comparison["delta"]["invalid_call_rate"] == 0
    assert comparison["delta"]["latency_ms_p50"] < 0
    assert comparison["delta"]["latency_ms_p95"] < 0


def test_validate_metrics_report_rejects_missing_fields() -> None:
    """Missing metric fields are rejected before comparison."""
    with pytest.raises(ValueError, match="missing metric fields"):
        validate_metrics_report({"tool_accuracy": 1.0})

    with pytest.raises(ValueError, match="missing latency_ms fields"):
        validate_metrics_report(
            {
                "serialized_request_bytes": {"total": 1, "mean": 1.0},
                "prompt_tokens": {"total": 1, "mean": 1.0, "missing_usage_count": 0},
                "tool_accuracy": 1.0,
                "invalid_call_rate": 0.0,
                "latency_ms": {"p50": 1.0},
            }
        )


def test_compute_latency_percentiles_is_deterministic() -> None:
    """Latency percentiles use stable p50/p95 values."""
    assert compute_latency_percentiles([100.0, 200.0, 300.0, 400.0, 500.0]) == {
        "p50": 300.0,
        "p95": 480.0,
    }


class _FakeSseStream:
    """Async readline stream that simulates delayed SSE token delivery."""

    def __init__(self, *, token_delay_s: float) -> None:
        self._token_delay_s = token_delay_s
        self._lines = [
            b'data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}\n',
            b"data: [DONE]\n",
        ]
        self._index = 0
        self.first_token_sent_at: float | None = None

    async def readline(self) -> bytes:
        if self._index >= len(self._lines):
            return b""
        if self._index == 0:
            await asyncio.sleep(self._token_delay_s)
            self.first_token_sent_at = time.perf_counter()
        line = self._lines[self._index]
        self._index += 1
        return line


def _configure_stream_post(
    mock_session: aiohttp.ClientSession,
    *,
    token_delay_s: float,
) -> tuple[list[float], _FakeSseStream]:
    post_started_at: list[float] = []
    stream = _FakeSseStream(token_delay_s=token_delay_s)

    def _post(*_args: Any, **_kwargs: Any) -> AsyncMock:
        post_started_at.append(time.perf_counter())
        response = AsyncMock()
        response.status = 200
        response.content = stream
        context_manager = AsyncMock()
        context_manager.__aenter__ = AsyncMock(return_value=response)
        context_manager.__aexit__ = AsyncMock(return_value=False)
        return context_manager

    mock_session.post = MagicMock(side_effect=_post)
    return post_started_at, stream


def _configure_routing_post(
    mock_session: aiohttp.ClientSession,
    *,
    token_delay_s: float,
    completion_delay_s: float,
) -> dict[str, int]:
    counts = {"stream": 0, "completion": 0}

    def _post(*_args: Any, **kwargs: Any) -> AsyncMock:
        response = AsyncMock()
        response.status = 200
        if kwargs.get("json", {}).get("stream") is True:
            counts["stream"] += 1
            response.content = _FakeSseStream(token_delay_s=token_delay_s)
            context_manager = AsyncMock()
            context_manager.__aenter__ = AsyncMock(return_value=response)
            context_manager.__aexit__ = AsyncMock(return_value=False)
            return context_manager

        counts["completion"] += 1

        async def _json(**_kwargs: Any) -> dict[str, Any]:
            await asyncio.sleep(completion_delay_s)
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The living room light is on.",
                        }
                    }
                ]
            }

        response.json = AsyncMock(side_effect=_json)
        context_manager = AsyncMock()
        context_manager.__aenter__ = AsyncMock(return_value=response)
        context_manager.__aexit__ = AsyncMock(return_value=False)
        return context_manager

    mock_session.post = MagicMock(side_effect=_post)
    return counts


async def test_probe_ttft_starts_before_post_and_ends_on_first_token(
    mock_session: aiohttp.ClientSession,
) -> None:
    """TTFT starts immediately before POST and ends on the first generated token."""
    token_delay_s = 0.05
    post_started_at, stream = _configure_stream_post(
        mock_session,
        token_delay_s=token_delay_s,
    )
    payload = build_chat_completions_payload(
        [{"role": "user", "content": "turn on the living room light"}],
        model="test-model",
    )
    client = LlamaCppClient(mock_session, "http://127.0.0.1:8080/v1", timeout=30)
    ttft_ms = await client.probe_ttft_ms(payload)

    assert len(post_started_at) == 1
    assert stream.first_token_sent_at is not None
    server_ttft_ms = (stream.first_token_sent_at - post_started_at[0]) * 1000.0
    assert ttft_ms == pytest.approx(server_ttft_ms, rel=0.25, abs=15.0)
    assert mock_session.post.call_args.kwargs["json"]["stream"] is True
    assert "stream" not in payload


async def test_measure_live_latency_once_ends_after_final_ha_result(
    mock_session: aiohttp.ClientSession,
) -> None:
    """End-to-end latency ends after the final HA result, not at first token."""
    token_delay_s = 0.02
    completion_delay_s = 0.04
    ha_delay_s = 0.03
    _configure_routing_post(
        mock_session,
        token_delay_s=token_delay_s,
        completion_delay_s=completion_delay_s,
    )
    payload = build_chat_completions_payload(
        [{"role": "user", "content": "turn on the living room light"}],
        model="test-model",
    )
    client = LlamaCppClient(mock_session, "http://127.0.0.1:8080/v1", timeout=30)

    async def execute_completion() -> str:
        await client.chat_completion(
            payload["messages"],
            model=payload["model"],
            temperature=payload["temperature"],
            max_tokens=payload["max_tokens"],
        )
        await asyncio.sleep(ha_delay_s)
        return "ha-complete"

    sample = await measure_live_latency_once(
        client,
        payload,
        execute_completion=execute_completion,
    )

    min_e2e_ms = (completion_delay_s + ha_delay_s) * 1000.0
    assert sample.end_to_end_ms >= min_e2e_ms * 0.8
    assert sample.end_to_end_ms > sample.ttft_ms


async def test_run_live_latency_benchmark_records_warmups_repetitions_median_p95(
    mock_session: aiohttp.ClientSession,
) -> None:
    """Live runner records warmups, repetitions, median, and p95 latency."""
    token_delay_s = 0.01
    completion_delay_s = 0.02
    ha_delay_s = 0.01
    counts = _configure_routing_post(
        mock_session,
        token_delay_s=token_delay_s,
        completion_delay_s=completion_delay_s,
    )
    client = LlamaCppClient(mock_session, "http://127.0.0.1:8080/v1", timeout=30)
    config = LiveLatencyConfig(warmups=1, repetitions=3)

    async def execute_completion() -> str:
        await client.chat_completion(
            [{"role": "user", "content": "turn on the living room light"}],
            model="test-model",
        )
        await asyncio.sleep(ha_delay_s)
        return "ha-complete"

    report = await run_live_latency_benchmark(
        client,
        messages=[{"role": "user", "content": "turn on the living room light"}],
        model="test-model",
        config=config,
        execute_completion=execute_completion,
    )

    validate_live_latency_report(report)
    assert report["warmups"] == 1
    assert report["repetitions"] == 3
    assert set(report["ttft_ms"]) == {"p50", "p95"}
    assert set(report["end_to_end_ms"]) == {"p50", "p95"}
    assert report["ttft_ms"]["p50"] <= report["ttft_ms"]["p95"]
    assert report["end_to_end_ms"]["p50"] <= report["end_to_end_ms"]["p95"]
    assert counts["stream"] == 4
    assert counts["completion"] == 4


def test_build_live_latency_report_is_deterministic() -> None:
    """Live latency aggregation uses stable median and p95 values."""
    report = build_live_latency_report(
        ttft_samples_ms=[40.0, 50.0, 60.0, 70.0, 80.0],
        end_to_end_samples_ms=[120.0, 140.0, 160.0, 180.0, 200.0],
        warmups=1,
        repetitions=5,
    )
    assert report == {
        "warmups": 1,
        "repetitions": 5,
        "ttft_ms": {"p50": 60.0, "p95": 78.0},
        "end_to_end_ms": {"p50": 160.0, "p95": 196.0},
    }


EVALS_ROOT = Path(__file__).resolve().parents[1] / "evals"
BASELINE_CURRENT_PATH = EVALS_ROOT / "baselines" / "current.json"


def _sample_run_metadata(*, homeassistant: str = "2026.8.3") -> dict[str, Any]:
    return {
        "homeassistant": homeassistant,
        "llama_cpp": "b4567",
        "model": "fixture-model",
        "chat_template": "llama3",
        "hardware": "Apple M2",
        "warmups": 1,
        "repetitions": 5,
    }


def _sample_fingerprints() -> dict[str, Any]:
    return {
        "schema": "sha256:fixture-schema",
        "gguf_sha256": "sha256:fixture-gguf",
        "llama_server_args": ["--jinja", "--model", "fixture.gguf"],
        "cases_file": "evals/cases/v1.json",
        "cases_version": 1,
    }


def _sample_live_latency() -> dict[str, Any]:
    return build_live_latency_report(
        ttft_samples_ms=[40.0, 50.0, 60.0, 70.0, 80.0],
        end_to_end_samples_ms=[120.0, 140.0, 160.0, 180.0, 200.0],
        warmups=1,
        repetitions=5,
    )


def _build_fixture_release_report(
    records: dict[str, EvalRecord],
    *,
    homeassistant: str = "2026.8.3",
    matrix_id: str = "current",
    live_latency: dict[str, Any] | None = None,
    latency_explanations: dict[str, str] | None = None,
) -> dict[str, Any]:
    cases = load_cases(CASES_PATH)
    return build_release_report(
        cases,
        records,
        matrix_id=matrix_id,
        metadata=_sample_run_metadata(homeassistant=homeassistant),
        fingerprints=_sample_fingerprints(),
        live_latency=live_latency or _sample_live_latency(),
        latency_explanations=latency_explanations,
    )


def test_derive_latency_tolerance_from_baseline_only() -> None:
    """Latency tolerance is computed from baseline live latency, not the candidate."""
    live_latency = _sample_live_latency()
    tolerance = derive_latency_tolerance_ms(live_latency)
    assert tolerance == {
        "ttft_p50": 6.0,
        "ttft_p95": 11.7,
        "end_to_end_p50": 16.0,
        "end_to_end_p95": 29.4,
    }

    worse_candidate = build_live_latency_report(
        ttft_samples_ms=[400.0, 500.0, 600.0, 700.0, 800.0],
        end_to_end_samples_ms=[1200.0, 1400.0, 1600.0, 1800.0, 2000.0],
        warmups=1,
        repetitions=5,
    )
    assert derive_latency_tolerance_ms(worse_candidate) != tolerance


def test_compare_eval_reports_rejects_metadata_mismatch() -> None:
    """Report comparison fails when run metadata differs."""
    cases = load_cases(CASES_PATH)
    baseline_records, improved_records = _recorded_fixture_records()
    baseline = _build_fixture_release_report(
        baseline_records,
        homeassistant="2026.8.3",
    )
    candidate = _build_fixture_release_report(
        improved_records,
        homeassistant="2026.8.2",
    )

    comparison = compare_eval_reports(baseline, candidate)

    assert comparison["passed"] is False
    assert comparison["gates"]["metadata_match"]["passed"] is False
    assert "homeassistant" in comparison["gates"]["metadata_match"]["mismatches"]


def test_compare_eval_reports_passes_improved_candidate() -> None:
    """All release gates pass when the candidate improves without metadata drift."""
    baseline_records, improved_records = _recorded_fixture_records()
    baseline = _build_fixture_release_report(baseline_records)
    candidate = _build_fixture_release_report(improved_records)

    comparison = compare_eval_reports(baseline, candidate)

    assert comparison["passed"] is True
    assert all(gate["passed"] for gate in comparison["gates"].values())
    assert comparison["delta"]["tool_accuracy"] > 0
    assert comparison["delta"]["prompt_tokens_total"] < 0


def test_compare_eval_reports_fails_tool_accuracy_regression() -> None:
    """Tool accuracy must not regress versus the baseline."""
    baseline_records, improved_records = _recorded_fixture_records()
    baseline = _build_fixture_release_report(baseline_records)
    regressed = _build_fixture_release_report(improved_records)
    regressed["metrics"]["tool_accuracy"] = 0.0

    comparison = compare_eval_reports(baseline, regressed)

    assert comparison["passed"] is False
    assert comparison["gates"]["tool_accuracy"]["passed"] is False


def test_compare_eval_reports_fails_invalid_call_rate_increase() -> None:
    """Invalid-call rate must not increase versus the baseline."""
    baseline_records, improved_records = _recorded_fixture_records()
    baseline = _build_fixture_release_report(baseline_records)
    worse = _build_fixture_release_report(improved_records)
    worse["metrics"]["invalid_call_rate"] = 1.0

    comparison = compare_eval_reports(baseline, worse)

    assert comparison["passed"] is False
    assert comparison["gates"]["invalid_call_rate"]["passed"] is False


def test_compare_eval_reports_fails_confident_routing_prompt_tokens_increase() -> None:
    """Confidently routed cases must not use more prompt tokens than the baseline."""
    baseline_records, improved_records = _recorded_fixture_records()
    baseline = _build_fixture_release_report(baseline_records)
    worse = _build_fixture_release_report(improved_records)
    worse["confident_routing"]["prompt_tokens_total"] = (
        baseline["confident_routing"]["prompt_tokens_total"] + 1
    )

    comparison = compare_eval_reports(baseline, worse)

    assert comparison["passed"] is False
    assert comparison["gates"]["confident_routing_prompt_tokens"]["passed"] is False


def test_compare_eval_reports_fails_unexplained_latency_regression() -> None:
    """TTFT and end-to-end latency regressions fail without an explanation."""
    baseline_records, improved_records = _recorded_fixture_records()
    baseline = _build_fixture_release_report(baseline_records)
    candidate = _build_fixture_release_report(improved_records)
    candidate["live_latency"]["ttft_ms"]["p95"] = (
        baseline["live_latency"]["ttft_ms"]["p95"]
        + baseline["latency_tolerance_ms"]["ttft_p95"]
        + 1.0
    )

    comparison = compare_eval_reports(baseline, candidate)

    assert comparison["passed"] is False
    assert comparison["gates"]["ttft_latency"]["passed"] is False


def test_compare_eval_reports_allows_explained_latency_regression() -> None:
    """Explained latency regressions do not fail the release gate."""
    baseline_records, improved_records = _recorded_fixture_records()
    baseline = _build_fixture_release_report(baseline_records)
    candidate = _build_fixture_release_report(
        improved_records,
        latency_explanations={"ttft_p95": "intentional model swap"},
    )
    candidate["live_latency"]["ttft_ms"]["p95"] = (
        baseline["live_latency"]["ttft_ms"]["p95"]
        + baseline["latency_tolerance_ms"]["ttft_p95"]
        + 1.0
    )

    comparison = compare_eval_reports(baseline, candidate)

    assert comparison["gates"]["ttft_latency"]["passed"] is True
    assert comparison["gates"]["ttft_latency"]["explained"] is True


def test_compare_report_formats_are_compact() -> None:
    """Comparison emits compact Markdown and JSON summaries."""
    baseline_records, improved_records = _recorded_fixture_records()
    comparison = compare_eval_reports(
        _build_fixture_release_report(baseline_records),
        _build_fixture_release_report(improved_records),
    )

    markdown = format_comparison_markdown(comparison)
    payload = format_comparison_json(comparison)

    assert "| gate |" in markdown
    assert "tool_accuracy" in markdown
    assert payload["passed"] is True
    assert "gates" in payload
    assert "delta" in payload


def test_baseline_file_exists_for_current_ha_matrix_version() -> None:
    """Archived baseline covers the current HA matrix entry."""
    current = load_baseline(BASELINE_CURRENT_PATH)

    validate_release_report(current)

    assert current["matrix_id"] == "current"
    assert current["metadata"]["homeassistant"] == "2026.8.3"
    assert current["fingerprints"]["cases_version"] == 1
    assert "latency_tolerance_ms" in current
