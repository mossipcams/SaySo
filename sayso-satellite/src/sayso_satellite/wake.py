"""Replaceable wake-word control around PCM capture."""

from __future__ import annotations

import struct
from typing import Protocol

from sayso_satellite.capture import DEFAULT_PRE_ROLL_MS, PushToTalkCapture

DEFAULT_WAKE_THRESHOLD = 5_000.0
DEFAULT_WAKE_HITS = 3


class WakeWordEngine(Protocol):
    """Wake-word detector that accepts PCM chunks independently of its engine."""

    def process(self, chunk: bytes) -> bool:
        """Return whether this chunk completes wake-word detection."""


def pcm_rms(chunk: bytes) -> float:
    """Return RMS amplitude for a PCM16 mono chunk."""

    if not chunk:
        return 0.0
    if len(chunk) % 2 != 0:
        raise ValueError("PCM chunk length must be even")
    samples = struct.unpack(f"<{len(chunk) // 2}h", chunk)
    total = sum(sample * sample for sample in samples)
    return (total / len(samples)) ** 0.5


class EnergyThresholdWakeEngine:
    """Prototype wake engine that triggers after consecutive loud PCM chunks."""

    def __init__(self, *, threshold: float, required_hits: int = DEFAULT_WAKE_HITS) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if required_hits <= 0:
            raise ValueError("required_hits must be positive")
        self._threshold = threshold
        self._required_hits = required_hits
        self._consecutive_hits = 0

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def required_hits(self) -> int:
        return self._required_hits

    def process(self, chunk: bytes) -> bool:
        if pcm_rms(chunk) >= self._threshold:
            self._consecutive_hits += 1
        else:
            self._consecutive_hits = 0
        if self._consecutive_hits >= self._required_hits:
            self._consecutive_hits = 0
            return True
        return False


class WakeWordSession:
    """Buffer a voice turn after wake detection or an explicit manual start."""

    def __init__(
        self,
        engine: WakeWordEngine | None,
        *,
        pre_roll_ms: int = DEFAULT_PRE_ROLL_MS,
    ) -> None:
        self._engine = engine
        self._capture = PushToTalkCapture(pre_roll_ms=pre_roll_ms)
        self._manual = False

    @property
    def is_active(self) -> bool:
        return self._capture.is_active

    def feed(self, chunk: bytes) -> None:
        """Feed PCM to pre-roll and, once started, the active voice turn."""

        if not chunk:
            return
        if (
            not self._capture.is_active
            and not self._manual
            and self._engine is not None
            and self._engine.process(chunk)
        ):
            self._capture.begin()
        self._capture.feed(chunk)

    def start_manual(self) -> None:
        """Start push-to-talk capture without invoking or implying wake detection."""

        self._capture.begin()
        self._manual = True

    def finish(self) -> bytes | None:
        """Return the buffered turn, or nothing when no turn was triggered."""

        if not self._capture.is_active:
            return None
        self._manual = False
        return self._capture.end()
