
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ring_buffer import Int16RingBuffer


class WakeAudioBuffer:

    def __init__(self, window_samples: int, hop_samples: int) -> None:
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

    @property
    def pending_lag(self) -> int:
        if self._pending_end is None:
            return 0
        return self._total - self._pending_end

    def clear(self) -> None:
        self._ring.clear()
        self._total = 0
        self._next_emit = self._window_samples
        self._pending_end = None

    def rearm_with_silence(self) -> None:
        self._ring.fill_silence()
        self._total = self._ring.capacity
        self._next_emit = self._total + self._hop_samples
        self._pending_end = None

    def feed(self, pcm_s16le: bytes) -> bool:
        samples = np.frombuffer(pcm_s16le, dtype="<i2")
        if samples.size == 0:
            return False
        self._ring.extend(samples)
        self._total += int(samples.size)
        if self._total < self._next_emit:
            return False
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


@dataclass(frozen=True)
class PrerollFlush:

    pcm: bytes
    start_index: int
    end_index: int
    underflow: bool = False


class WakePrerollLookback:

    def __init__(self, preroll_ms: int, sample_rate: int = 16000) -> None:
        capacity = max(0, preroll_ms * sample_rate // 1000)
        self._ring = Int16RingBuffer(capacity) if capacity > 0 else None
        self._sample_rate = sample_rate
        self._end_index = 0

    @property
    def end_index(self) -> int:
        return self._end_index

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def available_span(self) -> tuple[int, int]:
        if self._ring is None:
            return (self._end_index, self._end_index)
        held = min(self._end_index, self._ring.size)
        return (self._end_index - held, self._end_index)

    def clear(self, *, keep_index: bool = False) -> None:
        if self._ring is not None:
            self._ring.clear()
        if not keep_index:
            self._end_index = 0

    def feed(self, pcm_s16le: bytes) -> None:
        if not pcm_s16le:
            return
        samples = np.frombuffer(pcm_s16le, dtype="<i2")
        if samples.size == 0:
            return
        self._end_index += int(samples.size)
        if self._ring is not None:
            self._ring.extend(samples)

    def flush_until(self, detection_index: int, skip_ms: int) -> PrerollFlush:
        if self._ring is None or self._ring.size == 0:
            return PrerollFlush(b"", self._end_index, self._end_index, underflow=False)

        skip_samples = max(0, skip_ms) * self._sample_rate // 1000
        requested_start = int(detection_index) - skip_samples
        span_start, span_end = self.available_span
        if requested_start < span_start:
            return PrerollFlush(b"", span_end, span_end, underflow=True)
        start = requested_start
        if start >= span_end:
            return PrerollFlush(b"", span_end, span_end, underflow=False)

        window = self._ring.view()
        offset = start - span_start
        tail = window[offset:]
        return PrerollFlush(
            tail.astype("<i2", copy=False).tobytes(),
            start,
            span_end,
            underflow=False,
        )
