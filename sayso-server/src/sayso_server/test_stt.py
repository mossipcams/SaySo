"""Speech-to-text contract and fixture tolerance tests."""

from __future__ import annotations

from pathlib import Path

from sayso_server.audio_api import expected_pcm_byte_length
from sayso_server.mlx_stt import DEFAULT_MLX_WHISPER_MODEL_ID, MlxWhisperLoadedModel, MlxWhisperSttRuntime
from sayso_server.stt import (
    SpeechToTextRuntime,
    TranscriptionResult,
    load_stt_clip_fixture,
    load_stt_clip_pcm,
    normalize_transcript,
    pcm_duration_ms,
    transcript_within_tolerance,
    validate_pcm16_mono,
)

FIXTURES = Path(__file__).resolve().parents[3] / "evals" / "fixtures"


def _use_runtime(runtime: SpeechToTextRuntime, pcm: bytes) -> TranscriptionResult:
    runtime.load()
    return runtime.transcribe(pcm)


def test_validate_pcm16_mono_rejects_empty_and_odd_byte_lengths() -> None:
    try:
        validate_pcm16_mono(b"")
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for empty pcm")

    try:
        validate_pcm16_mono(b"\x00")
    except ValueError as exc:
        assert "sample" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for odd-length pcm")


def test_fixture_clip_matches_declared_pcm_framing() -> None:
    fixture = load_stt_clip_fixture(FIXTURES)
    pcm = load_stt_clip_pcm(fixture, FIXTURES)

    assert fixture.sample_rate_hz == 16_000
    assert fixture.channels == 1
    assert fixture.encoding == "pcm_s16le"
    assert len(pcm) == expected_pcm_byte_length(
        duration_ms=fixture.duration_ms,
        sample_rate_hz=fixture.sample_rate_hz,
        channels=fixture.channels,
    )
    assert pcm_duration_ms(pcm=pcm, sample_rate_hz=fixture.sample_rate_hz) == fixture.duration_ms


def test_known_english_fixture_meets_declared_tolerance_with_fake_backend() -> None:
    fixture = load_stt_clip_fixture(FIXTURES)
    pcm = load_stt_clip_pcm(fixture, FIXTURES)
    expected = fixture.expected_transcript

    def fake_loader(model_id: str) -> MlxWhisperLoadedModel:
        return MlxWhisperLoadedModel(model=object(), model_id=model_id)

    def fake_transcribe(_loaded: MlxWhisperLoadedModel, clip_pcm: bytes) -> str:
        assert clip_pcm == pcm
        return expected

    runtime = MlxWhisperSttRuntime(loader=fake_loader, transcribe_fn=fake_transcribe)
    result = _use_runtime(runtime, pcm)

    assert isinstance(result, TranscriptionResult)
    assert result.metadata.runtime == "mlx-whisper"
    assert result.latency_ms >= 0
    assert result.audio_duration_ms == fixture.duration_ms
    assert transcript_within_tolerance(result.text, expected, fixture.tolerance)


def test_resident_runtime_does_not_reload_between_fixture_clips() -> None:
    fixture = load_stt_clip_fixture(FIXTURES)
    pcm = load_stt_clip_pcm(fixture, FIXTURES)
    load_calls: list[str] = []

    def fake_loader(model_id: str) -> MlxWhisperLoadedModel:
        load_calls.append(model_id)
        return MlxWhisperLoadedModel(model=object(), model_id=model_id)

    def fake_transcribe(_loaded: MlxWhisperLoadedModel, _pcm: bytes) -> str:
        return fixture.expected_transcript

    runtime = MlxWhisperSttRuntime(loader=fake_loader, transcribe_fn=fake_transcribe)
    runtime.load()
    runtime.transcribe(pcm)
    runtime.transcribe(pcm)

    assert load_calls == [DEFAULT_MLX_WHISPER_MODEL_ID]


def test_normalize_transcript_collapses_case_and_whitespace() -> None:
    assert normalize_transcript("  Turn OFF   the lights ") == "turn off the lights"
