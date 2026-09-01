"""Resident MLX runtime load-once and warm-metrics tests."""

from __future__ import annotations

import json

from sayso_server.mlx_runtime import DEFAULT_MLX_MODEL_ID, MlxLoadedModel, MlxModelRuntime
from sayso_server.parser import parse_model_output
from sayso_server.runtime import RawGenerationResult


def _valid_plan_json(intent: str) -> str:
    return json.dumps(
        {
            "outcome": "query",
            "intent": intent,
            "domain": "light",
        }
    )


def test_mlx_runtime_loads_once_for_two_generations() -> None:
    load_calls: list[str] = []

    def fake_loader(model_id: str) -> MlxLoadedModel:
        load_calls.append(model_id)
        return MlxLoadedModel(model=object(), tokenizer=object())

    def fake_generate(_loaded: MlxLoadedModel, text: str) -> tuple[str, int, int]:
        return _valid_plan_json(text), 5, 3

    runtime = MlxModelRuntime(loader=fake_loader, generate_fn=fake_generate)
    runtime.load()
    runtime.generate("turn off the living room lights")
    runtime.generate("turn on the kitchen light")

    assert load_calls == [DEFAULT_MLX_MODEL_ID]


def test_mlx_runtime_emits_warm_metrics_after_load() -> None:
    def fake_loader(_model_id: str) -> MlxLoadedModel:
        return MlxLoadedModel(model=object(), tokenizer=object())

    def fake_generate(_loaded: MlxLoadedModel, text: str) -> tuple[str, int, int]:
        return _valid_plan_json(text), 4, 2

    runtime = MlxModelRuntime(loader=fake_loader, generate_fn=fake_generate)
    runtime.load()
    result = runtime.generate("check if any lights are on")
    plan = parse_model_output(result.text, intent="check if any lights are on")

    assert isinstance(result, RawGenerationResult)
    assert result.metadata.model_id == DEFAULT_MLX_MODEL_ID
    assert result.metadata.runtime == "mlx"
    assert result.metadata.warm is True
    assert result.metadata.resident is True
    assert result.prompt_tokens == 4
    assert result.completion_tokens == 2
    assert result.latency_ms >= 0
    assert plan.outcome == "query"


def test_mlx_runtime_latency_excludes_load_time() -> None:
    times = iter([0.0, 10.0, 10.5, 11.0])

    def fake_clock() -> float:
        return next(times)

    def fake_loader(_model_id: str) -> MlxLoadedModel:
        return MlxLoadedModel(model=object(), tokenizer=object())

    def fake_generate(_loaded: MlxLoadedModel, text: str) -> tuple[str, int, int]:
        return _valid_plan_json(text), 1, 1

    runtime = MlxModelRuntime(
        loader=fake_loader,
        generate_fn=fake_generate,
        clock=fake_clock,
    )
    runtime.load()
    result = runtime.generate("hello")

    assert result.latency_ms == 500.0


def test_mlx_runtime_generate_requires_load() -> None:
    runtime = MlxModelRuntime(
        loader=lambda _model_id: MlxLoadedModel(model=object(), tokenizer=object()),
        generate_fn=lambda _loaded, _text: (_valid_plan_json("x"), 1, 1),
    )

    try:
        runtime.generate("hello")
    except RuntimeError as exc:
        assert "load" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError when generate called before load")
