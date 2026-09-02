"""Evaluation case executors for controller dry-run and live-safety no-op paths."""

from __future__ import annotations

import json
import time
import uuid
from functools import cache
from pathlib import Path

from evals.config import HOME_LLM_270M_MODEL_ID
from evals.latency import timing_boundaries_from_stages
from evals.ledger import classify_failure
from evals.metrics import EvalRecord
from evals.runner import CaseExecutionResult, CaseTiming, mark_non_live_executor
from evals.schema import EvalCase
from sayso_server.candidates import retrieve_candidates
from sayso_server.conversation import ConversationStore
from sayso_server.control_plan import ActionPlan, QueryPlan
from sayso_server.ha_client import FakeHaClient
from sayso_server.home_graph import HomeGraphSnapshot
from sayso_server.orchestrator import execute_control_plan
from sayso_server.results import ActionResultStatus
from sayso_server.runtime import FakeModelRuntime, ModelRuntime, compose_plan_generation
from sayso_server.telemetry import InteractionTelemetry

_HOME_GRAPH_PATH = Path(__file__).resolve().parent / "fixtures" / "home_graph.json"
_SATELLITE_ID = "macbook"
_CANDIDATE_LIMIT = 8
_resident_fake_runtime: ModelRuntime | None = None
_resident_comparison_runtime: ModelRuntime | None = None


@cache
def _load_home_graph() -> HomeGraphSnapshot:
    data = json.loads(_HOME_GRAPH_PATH.read_text(encoding="utf-8"))
    return HomeGraphSnapshot.model_validate(data)


def _ensure_runtime(
    runtime: ModelRuntime | None,
    *,
    factory: type[FakeModelRuntime] | None = None,
) -> tuple[ModelRuntime, float | None]:
    if runtime is not None:
        return runtime, None
    load_started = time.perf_counter()
    created = (factory or FakeModelRuntime)()
    created.load()
    readiness_ms = (time.perf_counter() - load_started) * 1000.0
    return created, readiness_ms


def _resident_fake_runtime_instance() -> tuple[ModelRuntime, float | None]:
    global _resident_fake_runtime
    runtime, readiness_ms = _ensure_runtime(_resident_fake_runtime)
    if _resident_fake_runtime is None:
        _resident_fake_runtime = runtime
    return runtime, readiness_ms


def _comparison_fake_runtime_instance() -> tuple[ModelRuntime, float | None]:
    global _resident_comparison_runtime
    runtime, readiness_ms = _ensure_runtime(
        _resident_comparison_runtime,
        factory=_ComparisonFakeRuntime,
    )
    if _resident_comparison_runtime is None:
        _resident_comparison_runtime = runtime
    return runtime, readiness_ms


class _ComparisonFakeRuntime(FakeModelRuntime):
    """In-tree stand-in for the Home-LLM 270M comparison slot (no model download)."""

    def __init__(self) -> None:
        super().__init__(model_id=HOME_LLM_270M_MODEL_ID, revision="comparison-fixture")

    def generate(self, prompt: str):
        from evals.config import COMPARISON_BASELINE_RUNTIME

        raw = super().generate(prompt)
        return raw.model_copy(
            update={
                "metadata": raw.metadata.model_copy(
                    update={"runtime": COMPARISON_BASELINE_RUNTIME},
                ),
            },
        )


def _case_timing_from_telemetry(
    *,
    telemetry: InteractionTelemetry,
    outcome_category: str,
    outcome_reason: str | None,
    request_id: str | None,
    retrieve_ms: float,
    readiness_ms: float | None,
    wall_total_ms: float,
    generation: object,
) -> CaseTiming:
    record = telemetry.finish(
        category=outcome_category,
        reason=outcome_reason,
        request_id=request_id,
    )
    stages = record.stages
    boundaries = timing_boundaries_from_stages(
        stt_stage_ms=stages.stt_ms,
        plan_stage_ms=stages.plan_ms,
        resolve_stage_ms=stages.resolve_ms,
        validate_stage_ms=stages.validate_ms,
        request_stage_ms=stages.request_ms,
        verify_stage_ms=stages.verify_ms,
    )
    gen = generation  # PlanGenerationResult
    return CaseTiming(
        total_ms=wall_total_ms,
        stt_ms=stages.stt_ms,
        retrieve_ms=retrieve_ms,
        plan_ms=boundaries["plan_ms"],
        resolve_ms=stages.resolve_ms,
        validate_ms=stages.validate_ms,
        request_ms=boundaries["request_ms"],
        verify_ms=boundaries["verify_ms"],
        readiness_ms=readiness_ms,
        prompt_tokens=gen.prompt_tokens,
        completion_tokens=gen.completion_tokens,
        model_id=gen.metadata.model_id,
    )


def execute_controller_dry_run(
    case: EvalCase,
    runtime: ModelRuntime,
    *,
    readiness_ms: float | None = None,
) -> CaseExecutionResult:
    """Run the controller pipeline with ``runtime`` without live Home Assistant actuation."""
    start = time.perf_counter()
    snapshot = _load_home_graph()
    conversation_store = ConversationStore(ttl_seconds=300.0)
    ha_client = FakeHaClient()

    turns = case.turns
    for text in turns[:-1]:
        conversation = conversation_store.get_state(_SATELLITE_ID)
        generation = compose_plan_generation(
            runtime=runtime,
            snapshot=snapshot,
            satellite_id=_SATELLITE_ID,
            area_id=case.origin,
            text=text,
            conversation=conversation,
        )
        execute_control_plan(
            generation.plan,
            snapshot,
            origin_area_id=case.origin,
            ha_client=ha_client,
            request_id=str(uuid.uuid4()),
            conversation_store=conversation_store,
            satellite_id=_SATELLITE_ID,
        )

    last_text = turns[-1]
    conversation = conversation_store.get_state(_SATELLITE_ID)
    retrieve_started = time.perf_counter()
    scored_candidates = retrieve_candidates(
        snapshot,
        origin_area_id=case.origin,
        request=last_text,
        conversation=conversation,
        limit=_CANDIDATE_LIMIT,
    )
    retrieve_ms = (time.perf_counter() - retrieve_started) * 1000.0
    recorded_candidate_entities = [candidate.item.entity_id for candidate in scored_candidates]

    telemetry = InteractionTelemetry(
        correlation_id=str(uuid.uuid4()),
        satellite_id=_SATELLITE_ID,
        area_id=case.origin,
    )
    with telemetry.time_stage("plan"):
        generation = compose_plan_generation(
            runtime=runtime,
            snapshot=snapshot,
            satellite_id=_SATELLITE_ID,
            area_id=case.origin,
            text=last_text,
            conversation=conversation,
        )
    telemetry.set_model_from_generation(generation)
    plan_dump = generation.plan.model_dump(mode="json")
    schema_failure = (
        plan_dump.get("outcome") == "no-action"
        and plan_dump.get("reason") == "model_output_invalid"
    )

    request_id = str(uuid.uuid4())
    if isinstance(generation.plan, ActionPlan):
        ha_client.queue_results(
            [
                (request_id, ActionResultStatus.ACCEPTED, None),
                (request_id, ActionResultStatus.COMPLETED, "state_changed"),
            ],
        )

    prior_request_count = len(ha_client.action_requests)
    outcome = execute_control_plan(
        generation.plan,
        snapshot,
        origin_area_id=case.origin,
        ha_client=ha_client,
        request_id=request_id,
        conversation_store=conversation_store,
        satellite_id=_SATELLITE_ID,
        telemetry=telemetry,
    )

    new_requests = ha_client.action_requests[prior_request_count:]
    recorded_resolved_entities = [request.entity_id for request in new_requests]

    recorded_query_answer: str | None = None
    if isinstance(generation.plan, QueryPlan):
        recorded_query_answer = outcome.reason

    record = EvalRecord(
        case_id=case.case_id,
        recorded_control_plan=plan_dump,
        schema_failure=schema_failure,
        recorded_candidate_entities=recorded_candidate_entities,
        recorded_resolved_entities=recorded_resolved_entities,
        executed_entities=[],
        ha_executed=False,
        recorded_query_answer=recorded_query_answer,
    )
    failure = classify_failure(case, record)
    if failure is not None:
        failure_stage, failure_reason = failure
        record = record.model_copy(
            update={
                "failure_stage": failure_stage,
                "failure_reason": failure_reason,
            },
        )

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    timing = _case_timing_from_telemetry(
        telemetry=telemetry,
        outcome_category=outcome.category.value,
        outcome_reason=outcome.reason,
        request_id=outcome.request_id,
        retrieve_ms=retrieve_ms,
        readiness_ms=readiness_ms,
        wall_total_ms=elapsed_ms,
        generation=generation,
    )
    return CaseExecutionResult(record=record, timing=timing)


def controller_dry_run_executor(case: EvalCase) -> CaseExecutionResult:
    """Run the deterministic controller pipeline with FakeModelRuntime."""
    runtime, readiness_ms = _resident_fake_runtime_instance()
    return execute_controller_dry_run(case, runtime, readiness_ms=readiness_ms)


def comparison_baseline_executor(case: EvalCase) -> CaseExecutionResult:
    """Run the same controller timing path for the Home-LLM 270M comparison slot."""
    runtime, readiness_ms = _comparison_fake_runtime_instance()
    return execute_controller_dry_run(case, runtime, readiness_ms=readiness_ms)


mark_non_live_executor(controller_dry_run_executor)
mark_non_live_executor(comparison_baseline_executor)
