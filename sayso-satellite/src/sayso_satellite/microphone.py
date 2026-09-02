"""Live Mac microphone input as 16 kHz mono PCM16 chunks."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Protocol

from sayso_satellite.capture import (
    BYTES_PER_SAMPLE,
    DEFAULT_PRE_ROLL_MS,
    SAMPLE_RATE_HZ,
    expected_pcm_byte_length,
)
from sayso_satellite.wake import WakeWordEngine, WakeWordSession

DEFAULT_CHUNK_MS = 20


class MicInputError(RuntimeError):
    """Raised when live microphone input fails or returns invalid PCM."""


class MicSource(Protocol):
    """Streaming PCM source that yields fixed-size chunks until closed."""

    def read(self, *, max_bytes: int) -> bytes: ...

    def close(self) -> None: ...

    @property
    def closed(self) -> bool: ...


class FakeLiveMicSource:
    """Replay PCM in fixed-size chunks like a live microphone for tests."""

    def __init__(
        self,
        pcm: bytes,
        *,
        chunk_bytes: int,
        fail_after_reads: int | None = None,
    ) -> None:
        if chunk_bytes <= 0 or chunk_bytes % BYTES_PER_SAMPLE != 0:
            raise ValueError("chunk_bytes must be a positive even number")
        if len(pcm) % BYTES_PER_SAMPLE != 0:
            raise ValueError("PCM byte length must be even")
        self._pcm = pcm
        self._chunk_bytes = chunk_bytes
        self._offset = 0
        self._reads = 0
        self._fail_after_reads = fail_after_reads
        self._closed = False

    def read(self, *, max_bytes: int) -> bytes:
        if self._closed:
            return b""
        if self._fail_after_reads is not None and self._reads >= self._fail_after_reads:
            self.close()
            raise MicInputError("simulated input failure")
        self._reads += 1
        limit = min(max_bytes, self._chunk_bytes)
        if self._offset >= len(self._pcm):
            return b""
        chunk = self._pcm[self._offset : self._offset + limit]
        self._offset += len(chunk)
        return chunk

    def close(self) -> None:
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def exhausted(self) -> bool:
        return self._offset >= len(self._pcm)


class MacMicrophoneSource:
    """Read 16 kHz mono PCM16 from the default Mac input via ffmpeg."""

    def __init__(self, process: subprocess.Popen[bytes], *, chunk_bytes: int) -> None:
        if chunk_bytes <= 0 or chunk_bytes % BYTES_PER_SAMPLE != 0:
            raise ValueError("chunk_bytes must be a positive even number")
        self._process = process
        self._chunk_bytes = chunk_bytes
        self._buffer = bytearray()
        self._closed = False

    def read(self, *, max_bytes: int) -> bytes:
        if self._closed:
            return b""
        self._ensure_process_running()
        limit = min(max_bytes, self._chunk_bytes)
        while len(self._buffer) < limit:
            if self._process.stdout is None:
                self.close()
                raise MicInputError("microphone process has no stdout")
            block = self._process.stdout.read(self._chunk_bytes)
            if not block:
                self._ensure_process_running()
                break
            if len(block) % BYTES_PER_SAMPLE != 0:
                self.close()
                raise MicInputError("invalid PCM16 framing from microphone")
            self._buffer.extend(block)
        if not self._buffer:
            if self._process.poll() is not None:
                self.close()
                raise MicInputError(
                    f"microphone process exited with code {self._process.returncode}"
                )
            return b""
        chunk = bytes(self._buffer[:limit])
        del self._buffer[:limit]
        return chunk

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.stdout is not None:
            self._process.stdout.close()
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=1.0)

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_process_running(self) -> None:
        if self._closed:
            return
        returncode = self._process.poll()
        if returncode is not None and returncode != 0:
            self.close()
            raise MicInputError(f"microphone process exited with code {returncode}")


def default_chunk_bytes(*, chunk_ms: int = DEFAULT_CHUNK_MS) -> int:
    return expected_pcm_byte_length(duration_ms=chunk_ms)


def open_mac_microphone(*, chunk_ms: int = DEFAULT_CHUNK_MS) -> MacMicrophoneSource:
    """Open the default Mac microphone as a streaming PCM16 source."""

    if sys.platform != "darwin":
        raise MicInputError("live microphone input is supported only on macOS")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise MicInputError("ffmpeg is required for live microphone input")
    chunk_bytes = default_chunk_bytes(chunk_ms=chunk_ms)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "avfoundation",
        "-i",
        ":0",
        "-ac",
        str(1),
        "-ar",
        str(SAMPLE_RATE_HZ),
        "-f",
        "s16le",
        "-",
    ]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise MicInputError(f"failed to start microphone capture: {exc}") from exc
    return MacMicrophoneSource(process, chunk_bytes=chunk_bytes)


def capture_live_pcm(
    source: MicSource,
    *,
    duration_ms: int,
    chunk_bytes: int,
    pre_roll_ms: int = DEFAULT_PRE_ROLL_MS,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    """Capture one bounded utterance from a live source into pre-roll + turn audio."""

    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    session = WakeWordSession(None, pre_roll_ms=pre_roll_ms)
    deadline = monotonic() + duration_ms / 1000
    session.start_manual()
    try:
        while monotonic() < deadline and not source.closed:
            chunk = source.read(max_bytes=chunk_bytes)
            if chunk:
                session.feed(chunk)
                continue
            if source.closed:
                break
            sleep(0.01)
    finally:
        source.close()
    return session.finish() or b""


def capture_wake_pcm(
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
    """Capture one utterance after wake detection, or nothing when no wake occurs."""

    if capture_ms <= 0:
        raise ValueError("capture_ms must be positive")
    if listen_timeout_ms <= 0:
        raise ValueError("listen_timeout_ms must be positive")
    session = WakeWordSession(engine, pre_roll_ms=pre_roll_ms)
    listen_deadline = monotonic() + listen_timeout_ms / 1000
    capture_deadline: float | None = None
    try:
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
                continue
            if source.closed:
                break
            sleep(0.01)
    finally:
        source.close()
    return session.finish()
