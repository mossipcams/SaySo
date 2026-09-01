"""Narrow speech-to-text runtime contract for resident MLX Whisper."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, Field

STT_SAMPLE_RATE_HZ = 16_000
STT_CHANNELS = 1
STT_PCM_ENCODING = "pcm_s16le"
_BYTES_PER_SAMPLE = 2


class SttMetadata(BaseModel):
    model_id: str = Field(min_length=1)
    runtime: str = Field(min_length=1)
    revision: str | None = None
    warm: bool = False
    resident: bool = False


class TranscriptionResult(BaseModel):
    text: str
    latency_ms: float = Field(ge=0)
    audio_duration_ms: int = Field(ge=0)
    metadata: SttMetadata


class SttTolerance(BaseModel):
    mode: str = Field(min_length=1)


class SttClipFixture(BaseModel):
    clip_id: str = Field(min_length=1)
    pcm_file: str = Field(min_length=1)
    sample_rate_hz: int = Field(gt=0)
    channels: int = Field(gt=0)
    encoding: str = Field(min_length=1)
    duration_ms: int = Field(gt=0)
    language: str = Field(min_length=1)
    expected_transcript: str
    tolerance: SttTolerance


class SpeechToTextRuntime(ABC):
    """Load-once runtime that transcribes 16 kHz mono PCM16 audio."""

    @abstractmethod
    def load(self) -> None:
        """Prepare the runtime for transcription."""

    @abstractmethod
    def transcribe(
        self,
        pcm: bytes,
        *,
        sample_rate_hz: int = STT_SAMPLE_RATE_HZ,
    ) -> TranscriptionResult:
        """Transcribe PCM16 mono audio into text and metrics."""


class FakeSpeechToTextRuntime(SpeechToTextRuntime):
    """Deterministic STT stand-in for tests and local wiring."""

    def __init__(self, *, transcript: str = "turn off the floor lamp") -> None:
        self.transcript = transcript
        self.transcribe_calls: list[bytes] = []
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def transcribe(
        self,
        pcm: bytes,
        *,
        sample_rate_hz: int = STT_SAMPLE_RATE_HZ,
    ) -> TranscriptionResult:
        if not self._loaded:
            msg = "speech-to-text runtime must be loaded before transcribe"
            raise RuntimeError(msg)
        validate_pcm16_mono(pcm, sample_rate_hz=sample_rate_hz)
        self.transcribe_calls.append(pcm)
        return TranscriptionResult(
            text=self.transcript,
            latency_ms=0.0,
            audio_duration_ms=pcm_duration_ms(pcm=pcm, sample_rate_hz=sample_rate_hz),
            metadata=SttMetadata(model_id="fake", runtime="fake"),
        )


def validate_pcm16_mono(pcm: bytes, *, sample_rate_hz: int = STT_SAMPLE_RATE_HZ) -> None:
    """Validate PCM framing for the resident STT contract."""

    if sample_rate_hz != STT_SAMPLE_RATE_HZ:
        msg = "sample rate must be 16 kHz for the STT runtime"
        raise ValueError(msg)
    if len(pcm) == 0:
        msg = "pcm must not be empty"
        raise ValueError(msg)
    if len(pcm) % _BYTES_PER_SAMPLE != 0:
        msg = "pcm must contain whole 16-bit samples"
        raise ValueError(msg)


def pcm_duration_ms(*, pcm: bytes, sample_rate_hz: int = STT_SAMPLE_RATE_HZ) -> int:
    validate_pcm16_mono(pcm, sample_rate_hz=sample_rate_hz)
    sample_count = len(pcm) // _BYTES_PER_SAMPLE
    return sample_count * 1000 // sample_rate_hz


def normalize_transcript(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text.strip().lower())
    return collapsed


def transcript_within_tolerance(
    actual: str,
    expected: str,
    tolerance: SttTolerance,
) -> bool:
    if tolerance.mode == "normalized":
        return normalize_transcript(actual) == normalize_transcript(expected)
    if tolerance.mode == "exact":
        return actual == expected
    msg = f"unsupported STT tolerance mode: {tolerance.mode}"
    raise ValueError(msg)


def load_stt_clip_fixture(fixtures_dir: Path | None = None) -> SttClipFixture:
    root = fixtures_dir or Path(__file__).resolve().parents[3] / "evals" / "fixtures"
    data = (root / "stt_clip.json").read_text()
    return SttClipFixture.model_validate_json(data)


def load_stt_clip_pcm(fixture: SttClipFixture, fixtures_dir: Path | None = None) -> bytes:
    root = fixtures_dir or Path(__file__).resolve().parents[3] / "evals" / "fixtures"
    return (root / fixture.pcm_file).read_bytes()


__all__ = [
    "STT_CHANNELS",
    "STT_PCM_ENCODING",
    "STT_SAMPLE_RATE_HZ",
    "FakeSpeechToTextRuntime",
    "SpeechToTextRuntime",
    "SttClipFixture",
    "SttMetadata",
    "SttTolerance",
    "TranscriptionResult",
    "load_stt_clip_fixture",
    "load_stt_clip_pcm",
    "normalize_transcript",
    "pcm_duration_ms",
    "transcript_within_tolerance",
    "validate_pcm16_mono",
]
