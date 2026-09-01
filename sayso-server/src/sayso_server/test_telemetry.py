"""Telemetry JSONL record tests."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest

from sayso_server.conversation import ConversationStore
from sayso_server.graph_store import HomeGraphStore
from sayso_server.ha_client import FakeHaClient
from sayso_server.home_graph import HomeGraphSnapshot
from sayso_server.orchestrator import execute_control_plan
from sayso_server.results import ActionResultStatus, ExecutionCategory
from sayso_server.runtime import FakeModelRuntime
from sayso_server.telemetry import (
    InteractionTelemetry,
    InteractionTelemetryRecord,
    JsonlTelemetrySink,
    STAGE_NAMES,
    mandatory_field_names,
)
from sayso_server.test_orchestrator import _action_plan, _load_graph
from sayso_server.text_api import OrchestratorTextController

FIXTURES = Path(__file__).resolve().parents[3] / "evals" / "fixtures"


class SteppedClock:
    """Deterministic monotonic clock for telemetry tests."""

    def __init__(self, *, start: float = 100.0, step: float = 0.01) -> None:
        self._value = start
        self._step = step

    def __call__(self) -> float:
        current = self._value
        self._value += self._step
        return current


def _mandatory_keys() -> frozenset[str]:
    return mandatory_field_names()


def _assert_mandatory_fields(record: InteractionTelemetryRecord) -> None:
    payload = record.model_dump(mode="json")
    missing = sorted(_mandatory_keys() - payload.keys())
    assert missing == [], f"missing mandatory telemetry fields: {missing}"
    stage_fields = {f"{stage}_ms" for stage in STAGE_NAMES} | {"total_ms"}
    assert set(record.stages.model_dump(mode="json")) == stage_fields
    for stage in STAGE_NAMES:
        assert getattr(record.stages, f"{stage}_ms") >= 0.0


def test_success_record_satisfies_schema_and_includes_stage_timings() -> None:
    graph = _load_graph()
    ha_client = FakeHaClient()
    clock = SteppedClock()
    telemetry = InteractionTelemetry(
        correlation_id="corr-success",
        satellite_id="macbook",
        area_id="area_living_room",
        clock=clock,
    )
    plan = _action_plan()
    ha_client.queue_results(
        [
            ("req-1", ActionResultStatus.ACCEPTED, None),
            ("req-1", ActionResultStatus.COMPLETED, "state_changed"),
        ],
    )

    with telemetry.time_stage("plan"):
        pass
    outcome = execute_control_plan(
        plan,
        graph,
        origin_area_id="area_living_room",
        ha_client=ha_client,
        request_id="req-1",
        telemetry=telemetry,
    )

    record = telemetry.finish(
        category=outcome.category.value,
        reason=outcome.reason,
        request_id=outcome.request_id,
    )
    _assert_mandatory_fields(record)
    assert record.category == ExecutionCategory.COMPLETED.value
    assert record.request_id == "req-1"
    assert record.stages.plan_ms > 0.0
    assert record.stages.resolve_ms > 0.0
    assert record.stages.validate_ms > 0.0
    assert record.stages.request_ms > 0.0
    assert record.stages.verify_ms > 0.0
    assert record.stages.total_ms > 0.0


def test_rejected_record_satisfies_schema() -> None:
    graph = _load_graph()
    ha_client = FakeHaClient()
    telemetry = InteractionTelemetry(
        correlation_id="corr-rejected",
        satellite_id="macbook",
        area_id="area_living_room",
    )
    plan = _action_plan()
    ha_client.queue_results([("req-2", ActionResultStatus.REJECTED, "domain_mismatch")])

    outcome = execute_control_plan(
        plan,
        graph,
        origin_area_id="area_living_room",
        ha_client=ha_client,
        request_id="req-2",
        telemetry=telemetry,
    )

    record = telemetry.finish(
        category=outcome.category.value,
        reason=outcome.reason,
        request_id=outcome.request_id,
    )
    _assert_mandatory_fields(record)
    assert record.category == ExecutionCategory.REJECTED.value
    assert record.reason == "domain_mismatch"
    assert record.stages.request_ms >= 0.0
    assert record.stages.verify_ms >= 0.0


def test_failed_record_includes_all_mandatory_fields() -> None:
    graph = _load_graph()
    ha_client = FakeHaClient()
    telemetry = InteractionTelemetry(
        correlation_id="corr-failed",
        satellite_id="macbook",
        area_id="area_living_room",
    )
    plan = _action_plan()
    ha_client.queue_results(
        [
            ("req-3", ActionResultStatus.ACCEPTED, None),
            ("req-3", ActionResultStatus.FAILED, "execution_failed"),
        ],
    )

    outcome = execute_control_plan(
        plan,
        graph,
        origin_area_id="area_living_room",
        ha_client=ha_client,
        request_id="req-3",
        telemetry=telemetry,
    )

    record = telemetry.finish(
        category=outcome.category.value,
        reason=outcome.reason,
        request_id=outcome.request_id,
    )
    _assert_mandatory_fields(record)
    assert record.category == ExecutionCategory.FAILED.value
    assert record.reason == "execution_failed"
    assert record.request_id == "req-3"
    assert record.correlation_id == "corr-failed"
    assert record.satellite_id == "macbook"
    assert record.area_id == "area_living_room"


def _sample_telemetry_record(*, correlation_id: str = "corr-jsonl") -> InteractionTelemetryRecord:
    return InteractionTelemetryRecord.model_validate(
        {
            "correlation_id": correlation_id,
            "satellite_id": "macbook",
            "area_id": "area_living_room",
            "input_type": "text",
            "category": "completed",
            "request_id": "req-jsonl",
            "monotonic_started_at": 42.0,
            "stages": {
                "stt_ms": 0.0,
                "plan_ms": 1.0,
                "resolve_ms": 2.0,
                "validate_ms": 3.0,
                "request_ms": 4.0,
                "verify_ms": 5.0,
                "total_ms": 15.0,
            },
        },
    )


def test_jsonl_sink_writes_one_record_per_interaction() -> None:
    buffer = StringIO()
    sink = JsonlTelemetrySink(buffer)
    record = _sample_telemetry_record()

    sink.write(record)
    sink.write(record)

    lines = [line for line in buffer.getvalue().splitlines() if line.strip()]
    assert len(lines) == 2
    parsed = json.loads(lines[0])
    assert parsed["correlation_id"] == "corr-jsonl"
    assert "audio" not in json.dumps(parsed).lower()


def test_jsonl_sink_write_flushes_record_to_disk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from sayso_server.telemetry import TELEMETRY_PATH_ENV_VAR, open_jsonl_telemetry_sink_from_env

    telemetry_path = tmp_path / "telemetry.jsonl"
    monkeypatch.setenv(TELEMETRY_PATH_ENV_VAR, str(telemetry_path))
    sink = open_jsonl_telemetry_sink_from_env()
    assert sink is not None

    sink.write(_sample_telemetry_record(correlation_id="corr-flush"))

    content = telemetry_path.read_text(encoding="utf-8")
    lines = [line for line in content.splitlines() if line.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["correlation_id"] == "corr-flush"


def test_jsonl_sink_close_flushes_and_closes_stream(tmp_path: Path) -> None:
    telemetry_path = tmp_path / "telemetry.jsonl"
    stream = telemetry_path.open("a", encoding="utf-8")
    sink = JsonlTelemetrySink(stream)

    sink.write(_sample_telemetry_record(correlation_id="corr-close"))
    sink.close()

    assert stream.closed
    parsed = json.loads(telemetry_path.read_text(encoding="utf-8").strip())
    assert parsed["correlation_id"] == "corr-close"


def test_jsonl_sink_stringio_still_works_after_flush() -> None:
    buffer = StringIO()
    sink = JsonlTelemetrySink(buffer)
    record = _sample_telemetry_record(correlation_id="corr-stringio")

    sink.write(record)
    sink.write(record)

    lines = [line for line in buffer.getvalue().splitlines() if line.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["correlation_id"] == "corr-stringio"


def test_interaction_telemetry_copies_input_type_into_record() -> None:
    telemetry = InteractionTelemetry(
        correlation_id="corr-audio-type",
        satellite_id="macbook",
        area_id="area_living_room",
        input_type="audio",
    )
    record = telemetry.finish(category="no_action", reason="stub")
    assert record.input_type == "audio"


def test_interaction_telemetry_defaults_input_type_to_text() -> None:
    telemetry = InteractionTelemetry(
        correlation_id="corr-text-default",
        satellite_id="macbook",
        area_id="area_living_room",
    )
    record = telemetry.finish(category="no_action", reason="stub")
    assert record.input_type == "text"


def test_orchestrator_text_controller_emits_telemetry_record() -> None:
    graph_store = HomeGraphStore()
    graph_store.replace_snapshot(_load_graph())
    runtime = FakeModelRuntime()
    runtime.load()
    sink_buffer = StringIO()
    sink = JsonlTelemetrySink(sink_buffer)
    controller = OrchestratorTextController(
        runtime=runtime,
        ha_client=FakeHaClient(),
        graph_store=graph_store,
        conversation_store=ConversationStore(ttl_seconds=300.0),
        telemetry_sink=sink,
    )

    payload = controller.handle(
        satellite_id="macbook",
        area_id="area_living_room",
        text="are any lights on",
        correlation_id="corr-text",
    )

    assert payload["category"] in {
        ExecutionCategory.COMPLETED.value,
        ExecutionCategory.NO_ACTION.value,
    }
    lines = [line for line in sink_buffer.getvalue().splitlines() if line.strip()]
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    record = InteractionTelemetryRecord.model_validate(parsed)
    _assert_mandatory_fields(record)
    assert record.correlation_id == "corr-text"
    assert record.input_type == "text"
    assert record.model is not None
    assert record.stages.plan_ms >= 0.0
    assert record.stages.stt_ms == 0.0


def test_stage_names_include_stt_first() -> None:
    assert STAGE_NAMES[0] == "stt"
    assert "stt" in STAGE_NAMES


def test_text_path_records_zero_stt_ms() -> None:
    telemetry = InteractionTelemetry(
        correlation_id="corr-text-stt-zero",
        satellite_id="macbook",
        area_id="area_living_room",
        input_type="text",
    )
    record = telemetry.finish(category="no_action", reason="stub")
    assert record.stages.stt_ms == 0.0


def test_record_stage_ms_sets_stt_timing() -> None:
    telemetry = InteractionTelemetry(
        correlation_id="corr-stt-stage",
        satellite_id="macbook",
        area_id="area_living_room",
        input_type="audio",
    )
    telemetry.record_stage_ms("stt", 12.5)
    record = telemetry.finish(category="no_action", reason="stub")
    assert record.stages.stt_ms == 12.5


def test_time_stage_stt_accumulates() -> None:
    clock = SteppedClock()
    telemetry = InteractionTelemetry(
        correlation_id="corr-stt-time",
        satellite_id="macbook",
        area_id="area_living_room",
        input_type="audio",
        clock=clock,
    )
    with telemetry.time_stage("stt"):
        pass
    record = telemetry.finish(category="no_action", reason="stub")
    assert record.stages.stt_ms > 0.0


def test_orchestrator_text_controller_records_zero_stt_ms_for_text_input() -> None:
    graph_store = HomeGraphStore()
    graph_store.replace_snapshot(_load_graph())
    runtime = FakeModelRuntime()
    runtime.load()
    sink_buffer = StringIO()
    sink = JsonlTelemetrySink(sink_buffer)
    controller = OrchestratorTextController(
        runtime=runtime,
        ha_client=FakeHaClient(),
        graph_store=graph_store,
        conversation_store=ConversationStore(ttl_seconds=300.0),
        telemetry_sink=sink,
    )

    controller.handle(
        satellite_id="macbook",
        area_id="area_living_room",
        text="are any lights on",
        correlation_id="corr-text-stt",
        input_type="text",
        stt_ms=0.0,
    )

    parsed = json.loads(sink_buffer.getvalue().strip())
    record = InteractionTelemetryRecord.model_validate(parsed)
    assert record.stages.stt_ms == 0.0


def test_orchestrator_text_controller_records_pre_measured_stt_ms() -> None:
    graph_store = HomeGraphStore()
    graph_store.replace_snapshot(_load_graph())
    runtime = FakeModelRuntime()
    runtime.load()
    sink_buffer = StringIO()
    sink = JsonlTelemetrySink(sink_buffer)
    controller = OrchestratorTextController(
        runtime=runtime,
        ha_client=FakeHaClient(),
        graph_store=graph_store,
        conversation_store=ConversationStore(ttl_seconds=300.0),
        telemetry_sink=sink,
    )

    controller.handle(
        satellite_id="macbook",
        area_id="area_living_room",
        text="turn off the floor lamp",
        correlation_id="corr-audio-stt",
        input_type="audio",
        stt_ms=42.0,
    )

    parsed = json.loads(sink_buffer.getvalue().strip())
    record = InteractionTelemetryRecord.model_validate(parsed)
    assert record.input_type == "audio"
    assert record.stages.stt_ms == 42.0
