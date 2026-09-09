"""Rolling wake-word audio window backed by a fixed int16 ring buffer."""

from __future__ import annotations

import numpy as np

from .ring_buffer import Int16RingBuffer


class WakeAudioBuffer:
    """Rolling window emitted on a fixed sample grid, independent of chunk size.

    Windows must advance by exactly ``hop_samples``, not by however much audio
    happened to arrive. The satellite receives PCM in chunks whose size does not
    divide the hop, and the previous implementation reset its counter to zero on
    emit, so the real advance was the chunk size rounded up to the hop -- 4000
    samples in production, which is 25 mel frames. Embedding reuse needs the
    advance to be a whole number of stride-8 mel frames, and 25 is not; measured
    reuse in production was exactly zero.

    Emitting on an internal grid instead costs one hop of ring slack and up to
    one hop of latency, and makes the inference cadence deterministic regardless
    of the caller's chunk size.
    """

    def __init__(self, window_samples: int, hop_samples: int) -> None:
        # One hop of slack so a window ending on the grid is still fully present
        # after a chunk overshoots it.
        self._ring = Int16RingBuffer(window_samples + hop_samples)
        self._window_samples = window_samples
        self._hop_samples = hop_samples
        self._total = 0
        self._next_emit = window_samples
        self._pending_end: int | None = None

    @property
    def window_samples(self) -> int:
        return self._window_samples

    @property
    def filled(self) -> bool:
        return self._total >= self._window_samples

    def clear(self) -> None:
        self._ring.clear()
        self._total = 0
        self._next_emit = self._window_samples
        self._pending_end = None

    def rearm_with_silence(self) -> None:
        """Prefill the window with silence for faster post-TTS re-arm."""
        self._ring.fill_silence()
        self._total = self._ring.capacity
        self._next_emit = self._total + self._hop_samples
        self._pending_end = None

    def feed(self, pcm_s16le: bytes) -> bool:
        """Append PCM and return True when a new inference window is due."""
        samples = np.frombuffer(pcm_s16le, dtype="<i2")
        if samples.size == 0:
            return False
        self._ring.extend(samples)
        self._total += int(samples.size)
        if self._total < self._next_emit:
            return False
        # A chunk can overshoot several grid points. Serve the newest one the
        # ring still covers and drop the rest: a stale window is worth less than
        # a current one, and the gap stays a whole number of hops, so embedding
        # reuse survives it.
        skipped = (self._total - self._next_emit) // self._hop_samples
        self._pending_end = self._next_emit + skipped * self._hop_samples
        self._next_emit = self._pending_end + self._hop_samples
        return True

    def window(self) -> np.ndarray:
        end = self._total if self._pending_end is None else self._pending_end
        newest_offset = self._total - end
        buf = self._ring.view()
        stop = buf.size - newest_offset
        start = stop - self._window_samples
        return buf[start:stop] if start >= 0 else buf[:stop]


class WakePrerollLookback:
    """Rolling PCM lookback for post-wake STT preroll flush."""

    def __init__(self, preroll_ms: int, sample_rate: int = 16000) -> None:
        capacity = max(0, preroll_ms * sample_rate // 1000)
        self._ring = Int16RingBuffer(capacity) if capacity > 0 else None
        self._sample_rate = sample_rate

    def clear(self) -> None:
        if self._ring is not None:
            self._ring.clear()

    def feed(self, pcm_s16le: bytes) -> None:
        if self._ring is None or not pcm_s16le:
            return
        samples = np.frombuffer(pcm_s16le, dtype="<i2")
        self._ring.extend(samples)

    def flush_bytes(self, wake_skip_ms: int) -> bytes:
        if self._ring is None or self._ring.size == 0:
            return b""
        skip_samples = wake_skip_ms * self._sample_rate // 1000
        window = self._ring.view()
        if skip_samples >= window.size:
            trail_samples = min(
                250 * self._sample_rate // 1000,
                window.size,
            )
            return window[-trail_samples:].astype("<i2", copy=False).tobytes()
        return window[skip_samples:].astype("<i2", copy=False).tobytes()
