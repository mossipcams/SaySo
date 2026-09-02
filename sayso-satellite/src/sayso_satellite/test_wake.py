"""Tests for replaceable wake-word capture."""

from __future__ import annotations

import struct
import time

import pytest

from sayso_satellite.capture import BYTES_PER_SAMPLE, expected_pcm_byte_length
from sayso_satellite.microphone import FakeLiveMicSource, capture_wake_pcm
from sayso_satellite.wake import (
    DEFAULT_WAKE_HITS,
    DEFAULT_WAKE_THRESHOLD,
    EnergyThresholdWakeEngine,
    WakeWordSession,
    pcm_rms,
)


class DetectOn:
    def __init__(self, target: bytes) -> None:
        self.target = target
        self.chunks: list[bytes] = []

    def process(self, chunk: bytes) -> bool:
        self.chunks.append(chunk)
        return chunk == self.target


def _pcm_bytes(*samples: int) -> bytes:
    return b"".join(struct.pack("<h", sample) for sample in samples)


def _chunk(*, sample: int, duration_ms: int) -> bytes:
    count = expected_pcm_byte_length(duration_ms=duration_ms) // BYTES_PER_SAMPLE
    return _pcm_bytes(*([sample] * count))


def test_pcm_rms_is_zero_for_silence() -> None:
    assert pcm_rms(_pcm_bytes(0, 0, 0)) == 0.0


def test_pcm_rms_increases_with_amplitude() -> None:
    quiet = pcm_rms(_pcm_bytes(100, -100))
    loud = pcm_rms(_pcm_bytes(10_000, -10_000))

    assert loud > quiet


def test_energy_threshold_engine_requires_configured_hits() -> None:
    engine = EnergyThresholdWakeEngine(threshold=1_000.0, required_hits=2)
    loud = _chunk(sample=10_000, duration_ms=20)

    assert engine.process(loud) is False
    assert engine.process(loud) is True


def test_energy_threshold_engine_resets_after_quiet_gap() -> None:
    engine = EnergyThresholdWakeEngine(threshold=1_000.0, required_hits=2)
    loud = _chunk(sample=10_000, duration_ms=20)
    quiet = _chunk(sample=0, duration_ms=20)

    assert engine.process(loud) is False
    assert engine.process(quiet) is False
    assert engine.process(loud) is False
    assert engine.process(loud) is True


def test_energy_threshold_engine_ignores_quiet_audio() -> None:
    engine = EnergyThresholdWakeEngine(threshold=5_000.0, required_hits=DEFAULT_WAKE_HITS)
    quiet = _chunk(sample=100, duration_ms=20)

    for _ in range(10):
        assert engine.process(quiet) is False


def test_energy_threshold_engine_exposes_defaults() -> None:
    engine = EnergyThresholdWakeEngine(threshold=DEFAULT_WAKE_THRESHOLD)

    assert engine.threshold == DEFAULT_WAKE_THRESHOLD
    assert engine.required_hits == DEFAULT_WAKE_HITS


def test_energy_threshold_engine_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="threshold"):
        EnergyThresholdWakeEngine(threshold=0)
    with pytest.raises(ValueError, match="required_hits"):
        EnergyThresholdWakeEngine(threshold=100.0, required_hits=0)


def test_energy_threshold_wake_preserves_pre_roll() -> None:
    engine = EnergyThresholdWakeEngine(threshold=1_000.0, required_hits=2)
    session = WakeWordSession(engine, pre_roll_ms=60)
    leading = _chunk(sample=100, duration_ms=20)
    wake = _chunk(sample=10_000, duration_ms=20)
    speech = _chunk(sample=500, duration_ms=20)

    session.feed(leading)
    session.feed(wake)
    session.feed(wake)
    session.feed(speech)

    assert session.finish() == leading + wake + wake + speech


def test_capture_wake_pcm_returns_none_without_detection() -> None:
    chunk_bytes = expected_pcm_byte_length(duration_ms=20)
    quiet = _chunk(sample=100, duration_ms=20) * 5
    mic = FakeLiveMicSource(quiet, chunk_bytes=chunk_bytes)
    engine = EnergyThresholdWakeEngine(threshold=5_000.0, required_hits=2)
    start = time.monotonic()

    pcm = capture_wake_pcm(
        mic,
        engine,
        capture_ms=100,
        listen_timeout_ms=50,
        chunk_bytes=chunk_bytes,
        monotonic=time.monotonic,
        sleep=lambda _seconds: None,
    )

    assert pcm is None
    assert mic.closed is True
    assert time.monotonic() - start < 1.0


def test_capture_wake_pcm_captures_after_wake_with_pre_roll() -> None:
    chunk_bytes = expected_pcm_byte_length(duration_ms=20)
    leading = _chunk(sample=100, duration_ms=60)
    wake = _chunk(sample=10_000, duration_ms=20)
    speech = _chunk(sample=500, duration_ms=80)
    mic = FakeLiveMicSource(leading + wake + wake + speech, chunk_bytes=chunk_bytes)
    engine = EnergyThresholdWakeEngine(threshold=1_000.0, required_hits=2)
    timeline = {"now": 0.0}

    def monotonic() -> float:
        return timeline["now"]

    def advance(_seconds: float) -> None:
        timeline["now"] += _seconds

    pcm = capture_wake_pcm(
        mic,
        engine,
        capture_ms=100,
        listen_timeout_ms=500,
        chunk_bytes=chunk_bytes,
        pre_roll_ms=60,
        monotonic=monotonic,
        sleep=advance,
    )

    assert pcm is not None
    assert pcm.startswith(leading[: chunk_bytes * 2])
    assert len(pcm) % BYTES_PER_SAMPLE == 0
    assert mic.closed is True


def test_undetected_audio_does_not_emit_a_turn() -> None:
    engine = DetectOn(b"\x02\x00")
    session = WakeWordSession(engine, pre_roll_ms=20)

    session.feed(b"\x00\x00")

    assert session.finish() is None
    assert engine.chunks == [b"\x00\x00"]


def test_detected_turn_includes_bounded_pre_roll_and_following_audio() -> None:
    engine = DetectOn(b"\x02\x00")
    session = WakeWordSession(engine, pre_roll_ms=10)
    leading = b"\x01\x00" * (expected_pcm_byte_length(duration_ms=10) // 2)
    wake = b"\x02\x00"
    speech = b"\x03\x00"

    session.feed(leading)
    session.feed(wake)
    session.feed(speech)

    assert session.finish() == leading + wake + speech


def test_manual_start_captures_without_calling_wake_engine() -> None:
    session = WakeWordSession(None, pre_roll_ms=20)
    leading = b"\x01\x00" * (expected_pcm_byte_length(duration_ms=10) // 2)
    speech = b"\x03\x00"

    session.feed(leading)
    session.start_manual()
    session.feed(speech)

    assert session.finish() == leading + speech
