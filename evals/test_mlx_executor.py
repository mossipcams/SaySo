"""Optional MLX executor tests — skip live-model cases when mlx-lm is missing."""

from __future__ import annotations

import importlib

import pytest

from evals.config import DEFAULT_MODEL_ID
from evals.executor import controller_dry_run_executor
from evals.mlx_executor import (
    MLX_EVAL_ENV_VAR,
    build_mlx_model_runtime,
    controller_mlx_executor,
    is_mlx_eval_enabled,
    is_mlx_lm_available,
    resolve_eval_executor,
)
from evals.schema import EvalCase


def _action_case(case_id: str = "mlx-001") -> EvalCase:
    return EvalCase.model_validate(
        {
            "case_id": case_id,
            "category": "simple_control",
            "home": "eval-home",
            "origin": "area_living_room",
            "turns": ["Turn off the ceiling lights"],
            "expected_control_plan": {
                "outcome": "action",
                "intent": "turn off the ceiling lights",
                "domain": "light",
                "targets": ["ceiling lights"],
                "state": "off",
            },
            "expected_candidate_entities": ["light.living_room_ceiling"],
            "expected_resolved_entities": ["light.living_room_ceiling"],
            "expected_outcome": "valid_action",
            "execution_allowed": True,
        },
    )


def test_mlx_executor_module_imports_without_mlx_lm() -> None:
    mod = importlib.import_module("evals.mlx_executor")
    assert callable(mod.is_mlx_lm_available)
    assert mod.is_mlx_lm_available() in {True, False}


def test_is_mlx_eval_enabled() -> None:
    assert is_mlx_eval_enabled({}) is False
    assert is_mlx_eval_enabled({MLX_EVAL_ENV_VAR: "0"}) is False
    assert is_mlx_eval_enabled({MLX_EVAL_ENV_VAR: "1"}) is True
    assert is_mlx_eval_enabled({MLX_EVAL_ENV_VAR: "true"}) is True


def test_resolve_eval_executor_defaults_to_fake_runtime() -> None:
    executor = resolve_eval_executor({})
    assert executor is controller_dry_run_executor


def test_resolve_eval_executor_falls_back_to_fake_when_mlx_lm_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("evals.mlx_executor.is_mlx_lm_available", lambda: False)
    executor = resolve_eval_executor({MLX_EVAL_ENV_VAR: "1"})
    assert executor is controller_dry_run_executor


def test_build_mlx_model_runtime_returns_none_when_mlx_lm_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("evals.mlx_executor.is_mlx_lm_available", lambda: False)
    assert build_mlx_model_runtime() is None


def test_build_mlx_model_runtime_uses_default_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    class FakeMlxRuntime:
        def __init__(self, *, model_id: str, **kwargs: object) -> None:
            captured["model_id"] = model_id

        def load(self) -> None:
            captured["loaded"] = "yes"

    monkeypatch.setattr("evals.mlx_executor.is_mlx_lm_available", lambda: True)

    import sayso_server.mlx_runtime as mlx_runtime_mod

    monkeypatch.setattr(mlx_runtime_mod, "MlxModelRuntime", FakeMlxRuntime)

    runtime = build_mlx_model_runtime()
    assert runtime is not None
    assert captured["model_id"] == DEFAULT_MODEL_ID
    assert captured["loaded"] == "yes"


def test_controller_mlx_executor_uses_injected_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sayso_server.runtime import FakeModelRuntime

    runtime = FakeModelRuntime(model_id=DEFAULT_MODEL_ID)
    runtime.load()
    monkeypatch.setattr(
        "evals.mlx_executor._resident_mlx_runtime_instance",
        lambda: runtime,
    )

    result = controller_mlx_executor(_action_case("mlx-fake-runtime-001"))
    assert result.record.recorded_control_plan is not None
    assert result.record.ha_executed is False
    assert result.timing.model_id == DEFAULT_MODEL_ID


@pytest.mark.skipif(not is_mlx_lm_available(), reason="mlx-lm not installed")
def test_resolve_eval_executor_selects_mlx_when_opt_in() -> None:
    executor = resolve_eval_executor({MLX_EVAL_ENV_VAR: "1"})
    assert executor is controller_mlx_executor
