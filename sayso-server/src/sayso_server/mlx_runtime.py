"""Resident MLX-LM runtime behind the narrow ModelRuntime contract."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from sayso_server.parser import parse_model_output
from sayso_server.runtime import ModelMetadata, ModelRuntime, PlanGenerationResult

DEFAULT_MLX_MODEL_ID = "mlx-community/LFM2.5-230M-OptiQ-4bit"


@dataclass(frozen=True, slots=True)
class MlxLoadedModel:
    model: object
    tokenizer: object


MlxLoader = Callable[[str], MlxLoadedModel]
MlxGenerateFn = Callable[[MlxLoadedModel, str], tuple[str, int, int]]


def _default_loader(model_id: str) -> MlxLoadedModel:
    try:
        from mlx_lm import load
    except ImportError as exc:
        msg = "mlx-lm is required for MLX runtime but is not installed"
        raise RuntimeError(msg) from exc

    model, tokenizer = load(model_id)
    return MlxLoadedModel(model=model, tokenizer=tokenizer)


def _default_generate(loaded: MlxLoadedModel, text: str) -> tuple[str, int, int]:
    try:
        from mlx_lm import generate
    except ImportError as exc:
        msg = "mlx-lm is required for MLX runtime but is not installed"
        raise RuntimeError(msg) from exc

    response = generate(
        loaded.model,
        loaded.tokenizer,
        prompt=text,
        max_tokens=256,
        verbose=False,
    )
    prompt_tokens = len(text.split())
    completion_tokens = max(1, len(response.split()))
    return response, prompt_tokens, completion_tokens


class MlxModelRuntime(ModelRuntime):
    """Load-once MLX runtime that retains the checkpoint across generations."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MLX_MODEL_ID,
        loader: MlxLoader | None = None,
        generate_fn: MlxGenerateFn | None = None,
        clock: Callable[[], float] | None = None,
        revision: str | None = None,
    ) -> None:
        self._model_id = model_id
        self._loader = loader or _default_loader
        self._generate_fn = generate_fn or _default_generate
        self._clock = clock or time.monotonic
        self._revision = revision
        self._loaded: MlxLoadedModel | None = None

    def load(self) -> None:
        if self._loaded is not None:
            return
        self._clock()
        self._loaded = self._loader(self._model_id)
        self._clock()

    def generate_plan(self, text: str) -> PlanGenerationResult:
        if self._loaded is None:
            msg = "model runtime must be loaded before generate_plan"
            raise RuntimeError(msg)

        started = self._clock()
        raw_output, prompt_tokens, completion_tokens = self._generate_fn(self._loaded, text)
        elapsed_ms = max(0.0, (self._clock() - started) * 1000.0)
        plan = parse_model_output(raw_output, intent=text)

        return PlanGenerationResult(
            plan=plan,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=elapsed_ms,
            metadata=ModelMetadata(
                model_id=self._model_id,
                runtime="mlx",
                revision=self._revision,
                warm=True,
                resident=True,
            ),
        )
