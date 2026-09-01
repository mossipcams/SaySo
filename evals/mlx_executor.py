"""Optional live MLX executor for eval benchmarks."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

from evals.config import DEFAULT_MODEL_ID
from evals.executor import controller_dry_run_executor, execute_controller_dry_run
from evals.runner import CaseExecutionResult, CaseExecutor, mark_non_live_executor
from evals.schema import EvalCase

if TYPE_CHECKING:
    from sayso_server.runtime import ModelRuntime

MLX_EVAL_ENV_VAR = "SAYSO_EVAL_MLX"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

_resident_mlx_runtime: ModelRuntime | None = None


def is_mlx_eval_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return True when the opt-in live MLX eval path is requested."""
    source = os.environ if environ is None else environ
    return source.get(MLX_EVAL_ENV_VAR, "").strip().lower() in _TRUTHY


def is_mlx_lm_available() -> bool:
    """Return True when ``mlx-lm`` can be imported."""
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        return False
    return True


def build_mlx_model_runtime(*, model_id: str = DEFAULT_MODEL_ID) -> ModelRuntime | None:
    """Construct and load ``MlxModelRuntime`` when ``mlx-lm`` is installed."""
    if not is_mlx_lm_available():
        return None

    from sayso_server.mlx_runtime import MlxModelRuntime

    runtime = MlxModelRuntime(model_id=model_id)
    runtime.load()
    return runtime


def _resident_mlx_runtime_instance() -> ModelRuntime:
    global _resident_mlx_runtime
    if _resident_mlx_runtime is None:
        runtime = build_mlx_model_runtime()
        if runtime is None:
            msg = "mlx-lm is required when SAYSO_EVAL_MLX=1 but is not installed"
            raise RuntimeError(msg)
        _resident_mlx_runtime = runtime
    return _resident_mlx_runtime


def controller_mlx_executor(case: EvalCase) -> CaseExecutionResult:
    """Run the controller pipeline with the resident MLX runtime."""
    return execute_controller_dry_run(case, _resident_mlx_runtime_instance())


mark_non_live_executor(controller_mlx_executor)


def resolve_eval_executor(
    environ: Mapping[str, str] | None = None,
) -> CaseExecutor:
    """Select FakeModelRuntime by default; MLX only when opt-in and importable."""
    if is_mlx_eval_enabled(environ) and is_mlx_lm_available():
        return controller_mlx_executor
    return controller_dry_run_executor
