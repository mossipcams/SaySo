"""Tests for replaceable wake-word capture."""

from __future__ import annotations

from sayso_satellite.capture import expected_pcm_byte_length
from sayso_satellite.wake import WakeWordSession


class DetectOn:
    def __init__(self, target: bytes) -> None:
        self.target = target
        self.chunks: list[bytes] = []

    def process(self, chunk: bytes) -> bool:
        self.chunks.append(chunk)
        return chunk == self.target


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
