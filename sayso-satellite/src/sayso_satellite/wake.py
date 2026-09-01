"""Replaceable wake-word control around PCM capture."""

from __future__ import annotations

from typing import Protocol

from sayso_satellite.capture import DEFAULT_PRE_ROLL_MS, PushToTalkCapture


class WakeWordEngine(Protocol):
    """Wake-word detector that accepts PCM chunks independently of its engine."""

    def process(self, chunk: bytes) -> bool:
        """Return whether this chunk completes wake-word detection."""


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
