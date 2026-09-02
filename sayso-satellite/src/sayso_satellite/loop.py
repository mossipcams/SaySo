"""Continuous wake-listen-assist-playback loop for the Mac satellite."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

from sayso_satellite.capture import DEFAULT_PRE_ROLL_MS, pcm_duration_ms
from sayso_satellite.microphone import MicSource
from sayso_satellite.wake import WakeWordEngine, WakeWordSession


class TurnHandler(Protocol):
    def __call__(self, pcm: bytes) -> None: ...


def run_one_wake_turn(
    source: MicSource,
    engine: WakeWordEngine,
    *,
    capture_ms: int,
    listen_timeout_ms: int,
    chunk_bytes: int,
    pre_roll_ms: int = DEFAULT_PRE_ROLL_MS,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes | None:
    """Wait for wake and capture one utterance without closing the mic source."""

    if capture_ms <= 0:
        raise ValueError("capture_ms must be positive")
    if listen_timeout_ms <= 0:
        raise ValueError("listen_timeout_ms must be positive")
    session = WakeWordSession(engine, pre_roll_ms=pre_roll_ms)
    listen_deadline = monotonic() + listen_timeout_ms / 1000
    capture_deadline: float | None = None
    captured_bytes = 0
    while not source.closed:
        now = monotonic()
        if capture_deadline is None and now >= listen_deadline:
            return None
        if capture_deadline is not None and now >= capture_deadline:
            break
        chunk = source.read(max_bytes=chunk_bytes)
        if chunk:
            was_active = session.is_active
            session.feed(chunk)
            if not was_active and session.is_active:
                capture_deadline = now + capture_ms / 1000
                captured_bytes = 0
            elif session.is_active:
                captured_bytes += len(chunk)
                if pcm_duration_ms(byte_length=captured_bytes) >= capture_ms:
                    break
            continue
        if source.closed:
            break
        sleep(0.01)
    return session.finish()


def run_continuous_loop(
    source: MicSource,
    engine: WakeWordEngine,
    *,
    capture_ms: int,
    listen_timeout_ms: int,
    chunk_bytes: int,
    on_turn: TurnHandler,
    should_stop: Callable[[], bool] | None = None,
    pre_roll_ms: int = DEFAULT_PRE_ROLL_MS,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Run wake -> capture -> on_turn repeatedly until ``should_stop`` returns true."""

    stop = should_stop or (lambda: False)
    try:
        while not stop():
            pcm = run_one_wake_turn(
                source,
                engine,
                capture_ms=capture_ms,
                listen_timeout_ms=listen_timeout_ms,
                chunk_bytes=chunk_bytes,
                pre_roll_ms=pre_roll_ms,
                monotonic=monotonic,
                sleep=sleep,
            )
            if stop():
                break
            if pcm is None or not pcm:
                continue
            on_turn(pcm)
    finally:
        if not source.closed:
            source.close()
