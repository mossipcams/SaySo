"""Resident MLX Whisper STT load-once and warm-metrics tests."""

from __future__ import annotations

from sayso_server.mlx_stt import (
    DEFAULT_MLX_WHISPER_MODEL_ID,
    MlxWhisperLoadedModel,
    MlxWhisperSttRuntime,
)
from sayso_server.stt import TranscriptionResult


def test_mlx_whisper_stt_loads_once_for_two_clips() -> None:
    load_calls: list[str] = []
    pcm = b"\x00\x01" * 800

    def fake_loader(model_id: str) -> MlxWhisperLoadedModel:
        load_calls.append(model_id)
        return MlxWhisperLoadedModel(model=object(), model_id=model_id)

    def fake_transcribe(_loaded: MlxWhisperLoadedModel, _pcm: bytes) -> str:
        return "turn off the living room lights"

    runtime = MlxWhisperSttRuntime(loader=fake_loader, transcribe_fn=fake_transcribe)
    runtime.load()
    runtime.transcribe(pcm)
    runtime.transcribe(pcm)

    assert load_calls == [DEFAULT_MLX_WHISPER_MODEL_ID]


def test_mlx_whisper_stt_emits_warm_metrics_after_load() -> None:
    pcm = b"\x00\x01" * 800

    def fake_loader(_model_id: str) -> MlxWhisperLoadedModel:
        return MlxWhisperLoadedModel(model=object(), model_id=DEFAULT_MLX_WHISPER_MODEL_ID)

    def fake_transcribe(_loaded: MlxWhisperLoadedModel, _pcm: bytes) -> str:
        return "turn off the living room lights"

    runtime = MlxWhisperSttRuntime(loader=fake_loader, transcribe_fn=fake_transcribe)
    runtime.load()
    result = runtime.transcribe(pcm)

    assert isinstance(result, TranscriptionResult)
    assert result.metadata.model_id == DEFAULT_MLX_WHISPER_MODEL_ID
    assert result.metadata.runtime == "mlx-whisper"
    assert result.metadata.warm is True
    assert result.metadata.resident is True
    assert result.latency_ms >= 0
    assert result.audio_duration_ms == 50
    assert result.text == "turn off the living room lights"


def test_mlx_whisper_stt_latency_excludes_load_time() -> None:
    times = iter([0.0, 10.0, 10.5, 11.0])
    pcm = b"\x00\x01" * 800

    def fake_clock() -> float:
        return next(times)

    def fake_loader(_model_id: str) -> MlxWhisperLoadedModel:
        return MlxWhisperLoadedModel(model=object(), model_id=DEFAULT_MLX_WHISPER_MODEL_ID)

    def fake_transcribe(_loaded: MlxWhisperLoadedModel, _pcm: bytes) -> str:
        return "hello"

    runtime = MlxWhisperSttRuntime(
        loader=fake_loader,
        transcribe_fn=fake_transcribe,
        clock=fake_clock,
    )
    runtime.load()
    result = runtime.transcribe(pcm)

    assert result.latency_ms == 500.0


def test_mlx_whisper_stt_transcribe_requires_load() -> None:
    runtime = MlxWhisperSttRuntime(
        loader=lambda model_id: MlxWhisperLoadedModel(model=object(), model_id=model_id),
        transcribe_fn=lambda _loaded, _pcm: "hello",
    )

    try:
        runtime.transcribe(b"\x00\x01" * 4)
    except RuntimeError as exc:
        assert "load" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError when transcribe called before load")


def test_mlx_whisper_default_loader_uses_float16_for_fp16_transcribe() -> None:
    from unittest.mock import patch

    import mlx.core as mx

    from sayso_server.mlx_stt import _default_loader

    captured: dict[str, object] = {}

    def fake_load_model(model_id: str, *, dtype: mx.Dtype) -> object:
        captured["model_id"] = model_id
        captured["dtype"] = dtype
        return object()

    with patch("mlx_whisper.load_models.load_model", fake_load_model):
        loaded = _default_loader(DEFAULT_MLX_WHISPER_MODEL_ID)

    assert captured == {
        "model_id": DEFAULT_MLX_WHISPER_MODEL_ID,
        "dtype": mx.float16,
    }
    assert loaded.model_id == DEFAULT_MLX_WHISPER_MODEL_ID


def test_pcm16_mono_to_mlx_audio_matches_mlx_whisper_load_audio_shape() -> None:
    import mlx.core as mx

    from sayso_server.mlx_stt import _pcm16_mono_to_mlx_audio

    pcm = b"\x00\x01" * 800
    audio = _pcm16_mono_to_mlx_audio(pcm)

    assert isinstance(audio, mx.array)
    assert audio.dtype == mx.float32
    assert audio.shape == (800,)
