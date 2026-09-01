"""Evaluation case executors for controller dry-run and live-safety no-op paths."""

from __future__ import annotations

import json
import time
import uuid
from functools import cache
from pathlib import Path

from evals.ledger import classify_failure
from evals.metrics import EvalRecord
from evals.runner import CaseExecutionResult, CaseTiming, mark_non_live_executor
from evals.schema import EvalCase
from sayso_server.candidates import retrieve_candidates
from sayso_server.conversation import ConversationStore
from sayso_server.control_plan import QueryPlan
from sayso_server.ha_client import FakeHaClient
from sayso_server.home_graph import HomeGraphSnapshot
from sayso_server.orchestrator import execute_control_plan
from sayso_server.runtime import FakeModelRuntime, ModelRuntime, compose_plan_generation

_HOME_GRAPH_PATH = Path(__file__).resolve().parent / "fixtures" / "home_graph.json"
_SATELLITE_ID = "macbook"
_CANDIDATE_LIMIT = 8
_resident_fake_runtime: ModelRuntime | None = None


@cache
def _load_home_graph() -> HomeGraphSnapshot:
    data = json.loads(_HOME_GRAPH_PATH.read_text(encoding="utf-8"))
    return HomeGraphSnapshot.model_validate(data)


def _resident_fake_runtime_instance() -> ModelRuntime:
    global _resident_fake_runtime
    if _resident_fake_runtime is None:
        runtime = FakeModelRuntime()
        runtime.load()
        _resident_fake_runtime = runtime
    return _resident_fake_runtime


def execute_controller_dry_run(case: EvalCase, runtime: ModelRuntime) -> CaseExecutionResult:
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

    generation = compose_plan_generation(
        runtime=runtime,
        snapshot=snapshot,
        satellite_id=_SATELLITE_ID,
        area_id=case.origin,
        text=last_text,
        conversation=conversation,
    )
    plan_dump = generation.plan.model_dump(mode="json")
    schema_failure = (
        plan_dump.get("outcome") == "no-action"
        and plan_dump.get("reason") == "model_output_invalid"
    )

    prior_request_count = len(ha_client.action_requests)
    outcome = execute_control_plan(
        generation.plan,
        snapshot,
        origin_area_id=case.origin,
        ha_client=ha_client,
        request_id=str(uuid.uuid4()),
        conversation_store=conversation_store,
        satellite_id=_SATELLITE_ID,
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
    return CaseExecutionResult(
        record=record,
        timing=CaseTiming(
            total_ms=elapsed_ms,
            retrieve_ms=retrieve_ms,
            plan_ms=generation.latency_ms,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            model_id=generation.metadata.model_id,
        ),
    )


def controller_dry_run_executor(case: EvalCase) -> CaseExecutionResult:
    """Run the deterministic controller pipeline with FakeModelRuntime."""
    return execute_controller_dry_run(case, _resident_fake_runtime_instance())


mark_non_live_executor(controller_dry_run_executor)
