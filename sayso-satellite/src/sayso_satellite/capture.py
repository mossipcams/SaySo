"""Push-to-talk capture with bounded pre-roll for 16 kHz mono PCM16."""

from __future__ import annotations

from collections import deque
from pathlib import Path

SAMPLE_RATE_HZ = 16_000
CHANNELS = 1
BYTES_PER_SAMPLE = 2
DEFAULT_PRE_ROLL_MS = 300


def expected_pcm_byte_length(*, duration_ms: int) -> int:
    """Return byte length for a PCM16 chunk at the satellite sample rate."""

    return duration_ms * SAMPLE_RATE_HZ * CHANNELS * BYTES_PER_SAMPLE // 1000


def pcm_duration_ms(*, byte_length: int) -> int:
    """Return duration in milliseconds for a PCM16 byte buffer."""

    return byte_length * 1000 // (SAMPLE_RATE_HZ * CHANNELS * BYTES_PER_SAMPLE)


def read_pcm16_file(path: Path | str) -> bytes:
    """Read raw 16 kHz mono PCM16 bytes from ``path``."""

    pcm = Path(path).read_bytes()
    if not pcm:
        raise ValueError("PCM file is empty")
    if len(pcm) % BYTES_PER_SAMPLE != 0:
        raise ValueError("PCM byte length must be even")
    return pcm


class FixtureMicSource:
    """Replay PCM from a byte buffer in fixed-size chunks for tests."""

    def __init__(self, pcm: bytes, *, chunk_bytes: int) -> None:
        if chunk_bytes <= 0 or chunk_bytes % BYTES_PER_SAMPLE != 0:
            raise ValueError("chunk_bytes must be a positive even number")
        self._pcm = pcm
        self._chunk_bytes = chunk_bytes
        self._offset = 0

    def read(self, *, max_bytes: int) -> bytes:
        limit = min(max_bytes, self._chunk_bytes)
        if self._offset >= len(self._pcm):
            return b""
        chunk = self._pcm[self._offset : self._offset + limit]
        self._offset += len(chunk)
        return chunk

    @property
    def exhausted(self) -> bool:
        return self._offset >= len(self._pcm)


class PreRollBuffer:
    """Fixed-capacity rolling buffer of recent PCM bytes."""

    def __init__(self, *, capacity_bytes: int) -> None:
        if capacity_bytes < 0:
            raise ValueError("capacity_bytes must be non-negative")
        self._capacity = capacity_bytes
        self._chunks: deque[bytes] = deque()
        self._length = 0

    def extend(self, chunk: bytes) -> None:
        if not chunk:
            return
        if len(chunk) % BYTES_PER_SAMPLE != 0:
            raise ValueError("PCM chunk length must be even")
        self._chunks.append(chunk)
        self._length += len(chunk)
        while self._length > self._capacity and self._chunks:
            dropped = self._chunks.popleft()
            self._length -= len(dropped)

    def snapshot(self) -> bytes:
        return b"".join(self._chunks)

    @property
    def byte_length(self) -> int:
        return self._length


class PushToTalkCapture:
    """Capture pressed-to-released audio with bounded leading pre-roll."""

    def __init__(self, *, pre_roll_ms: int = DEFAULT_PRE_ROLL_MS) -> None:
        if pre_roll_ms < 0:
            raise ValueError("pre_roll_ms must be non-negative")
        self._pre_roll = PreRollBuffer(
            capacity_bytes=expected_pcm_byte_length(duration_ms=pre_roll_ms),
        )
        self._recording = bytearray()
        self._active = False
        self._preroll_snapshot = b""

    @property
    def is_active(self) -> bool:
        return self._active

    def feed(self, chunk: bytes) -> None:
        """Ingest mic PCM; updates pre-roll and the active recording."""

        if not chunk:
            return
        if len(chunk) % BYTES_PER_SAMPLE != 0:
            raise ValueError("PCM chunk length must be even")
        self._pre_roll.extend(chunk)
        if self._active:
            self._recording.extend(chunk)

    def begin(self) -> None:
        """Start push-to-talk recording, prepending the current pre-roll."""

        if self._active:
            raise RuntimeError("capture already active")
        self._preroll_snapshot = self._pre_roll.snapshot()
        self._recording = bytearray()
        self._active = True

    def end(self) -> bytes:
        """Stop recording and return pre-roll plus the pressed segment."""

        if not self._active:
            raise RuntimeError("capture not active")
        self._active = False
        return self._preroll_snapshot + bytes(self._recording)
