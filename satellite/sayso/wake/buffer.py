"""Rolling wake-word audio window backed by a fixed int16 ring buffer."""

from __future__ import annotations

from dataclasses import dataclass

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

    @property
    def pending_lag(self) -> int:
        """Samples fed after the end of the window :meth:`window` will return.

        Windows land on an internal hop grid, so the newest one generally ends
        *before* the last sample fed -- by however far the arriving chunk
        overshot the grid point. A caller stamping that window with its own
        end-of-stream index would place the detection up to one hop late.
        Subtract this to get the index of the window's final sample.
        """
        if self._pending_end is None:
            return 0
        return self._total - self._pending_end

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


@dataclass(frozen=True)
class PrerollFlush:
    """Result of trimming preroll up to a detection boundary.

    ``start_index``/``end_index`` are absolute capture sample indices, so the
    caller can move its STT cursor to exactly where this flush ended instead of
    guessing from wall-clock time.
    """

    pcm: bytes
    start_index: int
    end_index: int
    underflow: bool = False


class WakePrerollLookback:
    """Absolute-index PCM lookback for post-wake STT preroll flush.

    The trim is anchored to the sample index the classifier actually scored,
    not to the moment the flush happens to run. Detection runs off-thread, so
    the flush call can arrive hundreds of samples after the detection boundary;
    trimming from the end of the ring at flush time silently folded that
    variable latency into the cut. Anchoring to ``detection_index`` makes the
    emitted window a pure function of the audio, not of thread scheduling.

    Superseded on the live path by ``WakeCaptureRing`` (see ``wake/capture.py``),
    which backs both wake windows and the atomic STT handoff. Retained as a
    standalone, independently tested trim primitive.
    """

    def __init__(self, preroll_ms: int, sample_rate: int = 16000) -> None:
        capacity = max(0, preroll_ms * sample_rate // 1000)
        self._ring = Int16RingBuffer(capacity) if capacity > 0 else None
        self._sample_rate = sample_rate
        # Absolute index of the first sample ever appended.
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
        """Emit ``[detection_index - skip, end)`` clamped to what is held.

        Underflow means the requested start predates the oldest sample the
        lookback still holds, so the requested trim cannot be honoured. It is
        reported (and no audio is emitted) instead of silently substituting a
        trailing window: a wrong trim must be visible in the capture sidecar,
        not guessed at from the transcript.
        """
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
        # Ring view is [span_start, span_end); offset the start into it.
        offset = start - span_start
        tail = window[offset:]
        return PrerollFlush(
            tail.astype("<i2", copy=False).tobytes(),
            start,
            span_end,
            underflow=False,
        )
