"""Model runtime contract and fake runtime interchangeability tests."""

from __future__ import annotations

import json
from pathlib import Path

from sayso_server.control_plan import ControlPlan, NoActionPlan
from sayso_server.home_graph import HomeGraphSnapshot
from sayso_server.parser import parse_model_output
from sayso_server.runtime import (
    FakeModelRuntime,
    ModelRuntime,
    RawGenerationResult,
    compose_plan_generation,
)

FIXTURES = Path(__file__).resolve().parents[3] / "evals" / "fixtures"


def _load_graph() -> HomeGraphSnapshot:
    data = json.loads((FIXTURES / "home_graph.json").read_text())
    return HomeGraphSnapshot.model_validate(data)


def _use_runtime(runtime: ModelRuntime, prompt: str) -> RawGenerationResult:
    runtime.load()
    return runtime.generate(prompt)


def test_fake_runtime_exposes_raw_text_and_metrics() -> None:
    prompt = json.dumps({"user_text": "turn off the living room lights"})
    result = _use_runtime(FakeModelRuntime(), prompt)

    assert isinstance(result, RawGenerationResult)
    assert result.prompt_tokens >= 0
    assert result.completion_tokens >= 0
    assert result.latency_ms >= 0
    assert result.metadata.model_id
    assert result.metadata.runtime == "fake"

    plan = parse_model_output(result.text, intent="turn off the living room lights")
    round_tripped = ControlPlan.model_validate(plan.model_dump(mode="json"))
    assert round_tripped == plan


def test_fake_runtime_is_interchangeable() -> None:
    runtime: ModelRuntime = FakeModelRuntime()
    prompt = json.dumps({"user_text": "check if any lights are on"})
    result = _use_runtime(runtime, prompt)

    plan = parse_model_output(result.text, intent="check if any lights are on")
    assert plan.outcome == "query"


def test_generate_requires_load() -> None:
    runtime = FakeModelRuntime()

    try:
        runtime.generate('{"user_text": "hello"}')
    except RuntimeError as exc:
        assert "load" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError when generate called before load")


def test_compose_plan_generation_includes_candidates_in_prompt() -> None:
    graph = _load_graph()
    runtime = FakeModelRuntime()
    runtime.load()

    result = compose_plan_generation(
        runtime=runtime,
        snapshot=graph,
        satellite_id="macbook",
        area_id="area_living_room",
        text="turn off the floor lamp",
    )

    assert result.plan.outcome == "query"
    assert result.plan.intent == "turn off the floor lamp"
    assert result.metadata.runtime == "fake"


def test_compose_plan_generation_invalid_model_output_stays_invalid() -> None:
    graph = _load_graph()

    class GarbageRuntime(ModelRuntime):
        def load(self) -> None:
            return None

        def generate(self, prompt: str) -> RawGenerationResult:
            from sayso_server.runtime import ModelMetadata

            return RawGenerationResult(
                text="not-json",
                prompt_tokens=1,
                completion_tokens=1,
                latency_ms=0.0,
                metadata=ModelMetadata(model_id="garbage", runtime="fake"),
            )

    runtime = GarbageRuntime()
    runtime.load()

    result = compose_plan_generation(
        runtime=runtime,
        snapshot=graph,
        satellite_id="macbook",
        area_id="area_living_room",
        text="turn off the floor lamp",
    )

    assert isinstance(result.plan, NoActionPlan)
    assert result.plan.reason == "model_output_invalid"
