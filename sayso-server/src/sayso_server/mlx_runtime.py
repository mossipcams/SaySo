"""Resident MLX-LM runtime behind the narrow ModelRuntime contract."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from sayso_server.prompt import (
    GENERATION_INSTRUCTION,
    LFM_FEW_SHOT_ASSISTANT_JSON,
    LFM_FEW_SHOT_USER_JSON,
    extract_lfm_prompt_user_json,
)
from sayso_server.runtime import ModelMetadata, ModelRuntime, RawGenerationResult

DEFAULT_MLX_MODEL_ID = "mlx-community/LFM2.5-230M-OptiQ-4bit"
MODEL_ID_ENV_VAR = "SAYSO_MODEL_ID"


@dataclass(frozen=True, slots=True)
class MlxLoadedModel:
    model: object
    tokenizer: object


MlxLoader = Callable[[str], MlxLoadedModel]
MlxGenerateFn = Callable[[MlxLoadedModel, str], tuple[str, int, int]]


def ensure_mlx_lm_available() -> None:
    """Fail fast when ``mlx-lm`` is not installed."""

    try:
        import mlx_lm  # noqa: F401
    except ImportError as exc:
        msg = "mlx-lm is required to run sayso_server but is not installed"
        raise RuntimeError(msg) from exc


def build_mlx_runtime_for_server(
    *,
    environ: Mapping[str, str] | None = None,
    loader: MlxLoader | None = None,
) -> MlxModelRuntime:
    """Construct and load the resident MLX runtime for the process entrypoint."""

    ensure_mlx_lm_available()
    source = os.environ if environ is None else environ
    model_id = source.get(MODEL_ID_ENV_VAR, DEFAULT_MLX_MODEL_ID).strip() or DEFAULT_MLX_MODEL_ID
    runtime = MlxModelRuntime(model_id=model_id, loader=loader)
    runtime.load()
    return runtime


def _default_loader(model_id: str) -> MlxLoadedModel:
    try:
        from mlx_lm import load
    except ImportError as exc:
        msg = "mlx-lm is required for MLX runtime but is not installed"
        raise RuntimeError(msg) from exc

    model, tokenizer = load(model_id)
    return MlxLoadedModel(model=model, tokenizer=tokenizer)


def _prepare_generation_prompt(tokenizer: object, prompt: str) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is None:
        return prompt
    messages = [
        {"role": "system", "content": GENERATION_INSTRUCTION},
        {"role": "user", "content": LFM_FEW_SHOT_USER_JSON},
        {"role": "assistant", "content": LFM_FEW_SHOT_ASSISTANT_JSON},
        {"role": "user", "content": extract_lfm_prompt_user_json(prompt)},
    ]
    return apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _default_generate(loaded: MlxLoadedModel, text: str) -> tuple[str, int, int]:
    try:
        from mlx_lm import generate
    except ImportError as exc:
        msg = "mlx-lm is required for MLX runtime but is not installed"
        raise RuntimeError(msg) from exc

    formatted_prompt = _prepare_generation_prompt(loaded.tokenizer, text)
    response = generate(
        loaded.model,
        loaded.tokenizer,
        prompt=formatted_prompt,
        max_tokens=256,
        verbose=False,
    )
    prompt_tokens = len(formatted_prompt.split())
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

    def generate(self, prompt: str) -> RawGenerationResult:
        if self._loaded is None:
            msg = "model runtime must be loaded before generate"
            raise RuntimeError(msg)

        started = self._clock()
        raw_output, prompt_tokens, completion_tokens = self._generate_fn(self._loaded, prompt)
        elapsed_ms = max(0.0, (self._clock() - started) * 1000.0)

        return RawGenerationResult(
            text=raw_output,
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
