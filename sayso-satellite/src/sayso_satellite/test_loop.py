"""Tests for the continuous wake-listen-assist-playback satellite loop."""

from __future__ import annotations

import struct

import pytest

from sayso_satellite.capture import BYTES_PER_SAMPLE, expected_pcm_byte_length
from sayso_satellite.loop import run_continuous_loop, run_one_wake_turn
from sayso_satellite.microphone import FakeLiveMicSource, MicInputError
from sayso_satellite.wake import EnergyThresholdWakeEngine


def _pcm_bytes(*samples: int) -> bytes:
    return b"".join(struct.pack("<h", sample) for sample in samples)


def _chunk(*, sample: int, duration_ms: int) -> bytes:
    count = expected_pcm_byte_length(duration_ms=duration_ms) // BYTES_PER_SAMPLE
    return _pcm_bytes(*([sample] * count))


def _wake_sequence(*, hits: int = 2, speech_ms: int = 80, leading_ms: int = 60) -> bytes:
    leading = _chunk(sample=100, duration_ms=leading_ms)
    wake = _chunk(sample=10_000, duration_ms=20)
    speech = _chunk(sample=500, duration_ms=speech_ms)
    quiet = _chunk(sample=0, duration_ms=500)
    return leading + wake * hits + speech + quiet


class RecordingTurnHandler:
    def __init__(self, *, fail_with: Exception | None = None) -> None:
        self.calls: list[bytes] = []
        self._fail_with = fail_with

    def __call__(self, pcm: bytes) -> None:
        self.calls.append(pcm)
        if self._fail_with is not None:
            raise self._fail_with


def test_run_one_wake_turn_returns_none_without_closing_source() -> None:
    chunk_bytes = expected_pcm_byte_length(duration_ms=20)
    quiet = _chunk(sample=100, duration_ms=20) * 5
    mic = FakeLiveMicSource(quiet, chunk_bytes=chunk_bytes)
    engine = EnergyThresholdWakeEngine(threshold=5_000.0, required_hits=2)
    timeline = {"now": 0.0}

    def monotonic() -> float:
        return timeline["now"]

    def advance(seconds: float) -> None:
        timeline["now"] += seconds

    pcm = run_one_wake_turn(
        mic,
        engine,
        capture_ms=100,
        listen_timeout_ms=50,
        chunk_bytes=chunk_bytes,
        monotonic=monotonic,
        sleep=advance,
    )

    assert pcm is None
    assert mic.closed is False


def test_run_one_wake_turn_captures_without_closing_source() -> None:
    chunk_bytes = expected_pcm_byte_length(duration_ms=20)
    sequence = _wake_sequence(hits=2, speech_ms=80)
    mic = FakeLiveMicSource(sequence, chunk_bytes=chunk_bytes)
    engine = EnergyThresholdWakeEngine(threshold=1_000.0, required_hits=2)
    timeline = {"now": 0.0}

    def monotonic() -> float:
        return timeline["now"]

    def advance(seconds: float) -> None:
        timeline["now"] += seconds

    pcm = run_one_wake_turn(
        mic,
        engine,
        capture_ms=100,
        listen_timeout_ms=500,
        chunk_bytes=chunk_bytes,
        monotonic=monotonic,
        sleep=advance,
    )

    assert pcm is not None
    assert len(pcm) > 0
    assert mic.closed is False


def test_run_continuous_loop_completes_turn_and_returns_to_listening() -> None:
    chunk_bytes = expected_pcm_byte_length(duration_ms=20)
    sequence = _wake_sequence(hits=2, speech_ms=80) + _wake_sequence(hits=2, speech_ms=80)
    mic = FakeLiveMicSource(sequence, chunk_bytes=chunk_bytes)
    engine = EnergyThresholdWakeEngine(threshold=1_000.0, required_hits=2)
    handler = RecordingTurnHandler()
    timeline = {"now": 0.0}
    turns_seen = {"count": 0}

    def monotonic() -> float:
        return timeline["now"]

    def advance(seconds: float) -> None:
        timeline["now"] += seconds

    def should_stop() -> bool:
        turns_seen["count"] = len(handler.calls)
        return len(handler.calls) >= 2

    run_continuous_loop(
        mic,
        engine,
        capture_ms=100,
        listen_timeout_ms=500,
        chunk_bytes=chunk_bytes,
        on_turn=handler,
        should_stop=should_stop,
        monotonic=monotonic,
        sleep=advance,
    )

    assert len(handler.calls) == 2
    assert all(len(pcm) > 0 for pcm in handler.calls)
    assert mic.closed is True


def test_run_continuous_loop_skips_turn_when_no_wake() -> None:
    chunk_bytes = expected_pcm_byte_length(duration_ms=20)
    quiet = _chunk(sample=100, duration_ms=20) * 20
    mic = FakeLiveMicSource(quiet, chunk_bytes=chunk_bytes)
    engine = EnergyThresholdWakeEngine(threshold=5_000.0, required_hits=2)
    handler = RecordingTurnHandler()
    timeline = {"now": 0.0}
    waits = {"count": 0}

    def monotonic() -> float:
        return timeline["now"]

    def advance(seconds: float) -> None:
        timeline["now"] += seconds

    def should_stop() -> bool:
        waits["count"] += 1
        return waits["count"] >= 2

    run_continuous_loop(
        mic,
        engine,
        capture_ms=100,
        listen_timeout_ms=50,
        chunk_bytes=chunk_bytes,
        on_turn=handler,
        should_stop=should_stop,
        monotonic=monotonic,
        sleep=advance,
    )

    assert handler.calls == []
    assert mic.closed is True


def test_run_continuous_loop_does_not_call_turn_after_capture_failure() -> None:
    chunk_bytes = expected_pcm_byte_length(duration_ms=20)
    sequence = _wake_sequence(hits=2, speech_ms=80)
    mic = FakeLiveMicSource(sequence, chunk_bytes=chunk_bytes, fail_after_reads=3)
    engine = EnergyThresholdWakeEngine(threshold=1_000.0, required_hits=2)
    handler = RecordingTurnHandler()
    timeline = {"now": 0.0}

    def monotonic() -> float:
        return timeline["now"]

    def advance(seconds: float) -> None:
        timeline["now"] += seconds

    with pytest.raises(MicInputError):
        run_continuous_loop(
            mic,
            engine,
            capture_ms=100,
            listen_timeout_ms=500,
            chunk_bytes=chunk_bytes,
            on_turn=handler,
            monotonic=monotonic,
            sleep=advance,
        )

    assert handler.calls == []
    assert mic.closed is True


def test_run_continuous_loop_propagates_turn_failure_without_second_turn() -> None:
    from sayso_satellite.assist import AssistError

    chunk_bytes = expected_pcm_byte_length(duration_ms=20)
    sequence = _wake_sequence(hits=2, speech_ms=80) + _wake_sequence(hits=2, speech_ms=80)
    mic = FakeLiveMicSource(sequence, chunk_bytes=chunk_bytes)
    engine = EnergyThresholdWakeEngine(threshold=1_000.0, required_hits=2)
    handler = RecordingTurnHandler(fail_with=AssistError("assist failed"))
    timeline = {"now": 0.0}

    def monotonic() -> float:
        return timeline["now"]

    def advance(seconds: float) -> None:
        timeline["now"] += seconds

    with pytest.raises(AssistError, match="assist failed"):
        run_continuous_loop(
            mic,
            engine,
            capture_ms=100,
            listen_timeout_ms=500,
            chunk_bytes=chunk_bytes,
            on_turn=handler,
            monotonic=monotonic,
            sleep=advance,
        )

    assert len(handler.calls) == 1
    assert mic.closed is True


def test_run_continuous_loop_respects_should_stop_before_turn() -> None:
    chunk_bytes = expected_pcm_byte_length(duration_ms=20)
    sequence = _wake_sequence(hits=2, speech_ms=80)
    mic = FakeLiveMicSource(sequence, chunk_bytes=chunk_bytes)
    engine = EnergyThresholdWakeEngine(threshold=1_000.0, required_hits=2)
    handler = RecordingTurnHandler()

    run_continuous_loop(
        mic,
        engine,
        capture_ms=100,
        listen_timeout_ms=500,
        chunk_bytes=chunk_bytes,
        on_turn=handler,
        should_stop=lambda: True,
        monotonic=lambda: 0.0,
        sleep=lambda _seconds: None,
    )

    assert handler.calls == []
    assert mic.closed is True
