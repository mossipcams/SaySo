"""Model runtime contract and fake runtime interchangeability tests."""

from sayso_server.control_plan import ControlPlan
from sayso_server.runtime import FakeModelRuntime, ModelRuntime, PlanGenerationResult


def _use_runtime(runtime: ModelRuntime, text: str) -> PlanGenerationResult:
    runtime.load()
    return runtime.generate_plan(text)


def test_fake_runtime_exposes_plan_and_metrics() -> None:
    result = _use_runtime(FakeModelRuntime(), "turn off the living room lights")

    assert isinstance(result, PlanGenerationResult)
    assert result.prompt_tokens >= 0
    assert result.completion_tokens >= 0
    assert result.latency_ms >= 0
    assert result.metadata.model_id
    assert result.metadata.runtime == "fake"

    round_tripped = ControlPlan.model_validate(result.plan.model_dump(mode="json"))
    assert round_tripped == result.plan


def test_fake_runtime_is_interchangeable() -> None:
    runtime: ModelRuntime = FakeModelRuntime()
    result = _use_runtime(runtime, "check if any lights are on")

    assert result.plan.outcome == "query"


def test_generate_plan_requires_load() -> None:
    runtime = FakeModelRuntime()

    try:
        runtime.generate_plan("hello")
    except RuntimeError as exc:
        assert "load" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError when generate_plan called before load")
