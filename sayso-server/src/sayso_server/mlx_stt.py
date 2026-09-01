"""Resident MLX Whisper runtime behind the narrow SpeechToTextRuntime contract."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from sayso_server.stt import (
    STT_SAMPLE_RATE_HZ,
    SttMetadata,
    SpeechToTextRuntime,
    TranscriptionResult,
    pcm_duration_ms,
    validate_pcm16_mono,
)

DEFAULT_MLX_WHISPER_MODEL_ID = "mlx-community/whisper-small-mlx"


@dataclass(frozen=True, slots=True)
class MlxWhisperLoadedModel:
    model: object
    model_id: str


MlxWhisperLoader = Callable[[str], MlxWhisperLoadedModel]
MlxWhisperTranscribeFn = Callable[[MlxWhisperLoadedModel, bytes], str]


def _pcm16_mono_to_mlx_audio(pcm: bytes) -> object:
    try:
        import mlx.core as mx
        import numpy as np
    except ImportError as exc:
        msg = "numpy is required for MLX Whisper STT but is not installed"
        raise RuntimeError(msg) from exc

    return mx.array(np.frombuffer(pcm, dtype=np.int16)).astype(mx.float32) / 32768.0


def _default_loader(model_id: str) -> MlxWhisperLoadedModel:
    try:
        import mlx.core as mx
        from mlx_whisper.load_models import load_model
    except ImportError as exc:
        msg = "mlx-whisper is required for MLX STT but is not installed"
        raise RuntimeError(msg) from exc

    # mlx_whisper.transcribe defaults to fp16=True; loader dtype must match.
    model = load_model(model_id, dtype=mx.float16)
    return MlxWhisperLoadedModel(model=model, model_id=model_id)


def _default_transcribe(loaded: MlxWhisperLoadedModel, pcm: bytes) -> str:
    try:
        from mlx_whisper import transcribe
        from mlx_whisper.transcribe import ModelHolder
    except ImportError as exc:
        msg = "mlx-whisper is required for MLX STT but is not installed"
        raise RuntimeError(msg) from exc

    previous_model = ModelHolder.model
    previous_path = ModelHolder.model_path
    try:
        ModelHolder.model = loaded.model
        ModelHolder.model_path = loaded.model_id
        audio = _pcm16_mono_to_mlx_audio(pcm)
        result = transcribe(
            audio,
            path_or_hf_repo=loaded.model_id,
            verbose=False,
            language="en",
            fp16=True,
        )
    finally:
        ModelHolder.model = previous_model
        ModelHolder.model_path = previous_path

    text = result.get("text")
    if not isinstance(text, str):
        msg = "mlx-whisper transcribe did not return text"
        raise RuntimeError(msg)
    return text.strip()


class MlxWhisperSttRuntime(SpeechToTextRuntime):
    """Load-once MLX Whisper runtime that retains the checkpoint across clips."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MLX_WHISPER_MODEL_ID,
        loader: MlxWhisperLoader | None = None,
        transcribe_fn: MlxWhisperTranscribeFn | None = None,
        clock: Callable[[], float] | None = None,
        revision: str | None = None,
    ) -> None:
        self._model_id = model_id
        self._loader = loader or _default_loader
        self._transcribe_fn = transcribe_fn or _default_transcribe
        self._clock = clock or time.monotonic
        self._revision = revision
        self._loaded: MlxWhisperLoadedModel | None = None

    def load(self) -> None:
        if self._loaded is not None:
            return
        self._clock()
        self._loaded = self._loader(self._model_id)
        self._clock()

    def transcribe(
        self,
        pcm: bytes,
        *,
        sample_rate_hz: int = STT_SAMPLE_RATE_HZ,
    ) -> TranscriptionResult:
        if self._loaded is None:
            msg = "speech-to-text runtime must be loaded before transcribe"
            raise RuntimeError(msg)

        validate_pcm16_mono(pcm, sample_rate_hz=sample_rate_hz)
        duration_ms = pcm_duration_ms(pcm=pcm, sample_rate_hz=sample_rate_hz)

        started = self._clock()
        text = self._transcribe_fn(self._loaded, pcm)
        elapsed_ms = max(0.0, (self._clock() - started) * 1000.0)

        return TranscriptionResult(
            text=text,
            latency_ms=elapsed_ms,
            audio_duration_ms=duration_ms,
            metadata=SttMetadata(
                model_id=self._model_id,
                runtime="mlx-whisper",
                revision=self._revision,
                warm=True,
                resident=True,
            ),
        )


__all__ = [
    "DEFAULT_MLX_WHISPER_MODEL_ID",
    "MlxWhisperLoadedModel",
    "MlxWhisperSttRuntime",
]
