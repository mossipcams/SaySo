"""Sample-ordered capture timeline and the one deliberate resample to 16 kHz.

The satellite captures at the device's native rate and resamples exactly once,
here, before any audio is used for wake inference or sent to Home Assistant.
A per-block resample (``np.interp`` inside each block) restarts the filter every
block and leaves a discontinuity at every boundary; :class:`CaptureResampler`
keeps one continuous windowed-sinc history so the output is a seamless 16 kHz
stream regardless of how the audio server chunks it.

Every block appended to :class:`WakeCaptureRing` is stamped with its absolute
end sample index. Wake detection, preroll trimming, and the STT handoff all
reference that index, so nothing depends on when a thread happened to run.
"""

from __future__ import annotations

import math
import threading

import numpy as np

# Half-width, in input samples, of the windowed-sinc low-pass used when
# decimating. 16 taps is ample for speech-band audio and cheap enough to run on
# a Pi-class CPU without disturbing the capture cadence.
_FILTER_HALF_TAPS = 16


def gain_scalar_from_db(gain_db: float, ceiling: float = 8.0) -> float:
    """Convert a decibel gain to a bounded linear scalar.

    The ceiling keeps a misconfigured gain from driving the signal straight into
    saturation; clipping is still counted and reported by the capture tap.
    """
    if not math.isfinite(gain_db):
        return 1.0
    return float(min(ceiling, max(0.0, 10.0 ** (gain_db / 20.0))))


def _sinc_kernel(
    input_rate: int,
    output_rate: int,
    half_taps: int = _FILTER_HALF_TAPS,
) -> np.ndarray:
    """Build the anti-alias low-pass used by the streaming resampler.

    Cutoff is the *output* Nyquist relative to the input rate, so the kernel
    attenuates everything a downsample would otherwise fold back into the speech
    band. A Blackman window keeps the stopband well below speech level.
    """
    cutoff = min(0.5, 0.5 * output_rate / input_rate)
    taps = 2 * half_taps + 1
    n = np.arange(taps, dtype=np.float64) - half_taps
    kernel = 2.0 * cutoff * np.sinc(2.0 * cutoff * n)
    window = np.blackman(taps)
    kernel *= window
    total = float(np.sum(kernel))
    if total != 0.0:
        kernel /= total
    return kernel


class CaptureResampler:
    """Streaming windowed-sinc resampler with continuous filter state.

    Input and output are int16 little-endian PCM. Output sample timing is
    tracked with an exact rational accumulator, so long runs do not drift the
    way a floating-point ratio would.
    """

    def __init__(
        self,
        input_rate: int,
        output_rate: int,
        half_taps: int = _FILTER_HALF_TAPS,
    ) -> None:
        if input_rate <= 0 or output_rate <= 0:
            raise ValueError("input_rate and output_rate must be positive")
        self._input_rate = int(input_rate)
        self._output_rate = int(output_rate)
        self._passthrough = self._input_rate == self._output_rate
        self._half_taps = int(half_taps)
        self._kernel = (
            None
            if self._passthrough
            else _sinc_kernel(self._input_rate, self._output_rate, self._half_taps)
        )
        self._history = np.zeros(0, dtype=np.float64)
        # Absolute input sample index of history[0]. Trimming advances this so
        # indices are always interpreted against the retained window.
        self._history_start = 0
        # Exact rational output position: numerator/denominator in input samples.
        self._position_num = 0
        self._denominator = self._output_rate

    @property
    def input_rate(self) -> int:
        return self._input_rate

    @property
    def output_rate(self) -> int:
        return self._output_rate

    def reset(self) -> None:
        self._history = np.zeros(0, dtype=np.float64)
        self._history_start = 0
        self._position_num = 0

    def process(self, pcm_s16le: bytes) -> bytes:
        """Append input PCM and return every output sample now computable."""
        if not pcm_s16le:
            return b""
        incoming = np.frombuffer(pcm_s16le, dtype="<i2")
        if incoming.size == 0:
            return b""
        if self._passthrough:
            return incoming.astype("<i2", copy=False).tobytes()

        self._history = np.concatenate(
            (self._history, incoming.astype(np.float64))
        )
        # The oldest retained sample sits at absolute index `_history_start`.
        base = self._history_start

        outputs: list[float] = []
        last_needed = self._history.size - 1 + base
        while True:
            # Output sample k sits at input index k * input_rate / output_rate.
            centre = (
                self._position_num * self._input_rate
            ) / self._denominator
            if centre > last_needed:
                break
            outputs.append(self._sample_at(centre, base))
            self._position_num += 1

        if outputs:
            out = np.clip(np.rint(np.asarray(outputs)), -32768, 32767).astype("<i2")
        else:
            out = np.zeros(0, dtype="<i2")

        # Drop history that can no longer contribute to a future output sample.
        self._trim_history()
        return out.tobytes()

    def _sample_at(self, centre: float, base: int) -> float:
        assert self._kernel is not None
        left = int(math.floor(centre)) - self._half_taps
        offsets = np.arange(left, left + self._kernel.size, dtype=np.int64)
        indices = offsets - base
        valid = (indices >= 0) & (indices < self._history.size)
        if not np.any(valid):
            return 0.0
        taps = self._kernel[valid]
        values = self._history[indices[valid]]
        # Normalise by the taps actually available so a partially covered
        # window near the stream edge does not attenuate the signal.
        weight = float(np.sum(taps))
        if weight == 0.0:
            return 0.0
        return float(np.dot(values, taps) / weight)
    def _trim_history(self) -> None:
        # Keep one filter half-width of samples behind the earliest position
        # that can still be evaluated, and drop the rest.
        next_centre = (
            self._position_num * self._input_rate
        ) / self._denominator
        keep_from = int(math.floor(next_centre)) - self._half_taps - 1
        drop = keep_from - self._history_start
        if drop > 0:
            drop = min(drop, self._history.size)
            self._history = self._history[drop:]
            self._history_start += drop


class WakeCaptureRing:
    """Absolute-index int16 ring shared by wake inference and the STT handoff.

    ``end_index`` is the total number of samples ever appended. Reads are
    expressed in absolute sample indices, which is what makes the wake→STT
    handoff independent of thread scheduling: the trim boundary is a sample
    index, not a wall-clock instant.

    The STT cursor records how far the live path has already forwarded, so a
    preroll flush can emit ``[trim_index, end_index)`` and the next live block
    resumes exactly where the flush stopped. No sample is delivered twice.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = int(capacity)
        self._data = np.zeros(self._capacity, dtype=np.int16)
        self._end_index = 0
        self._write_pos = 0
        self._valid_since_reset = 0
        self._stt_cursor = 0
        self._lock = threading.RLock()
    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def end_index(self) -> int:
        with self._lock:
            return self._end_index

    @property
    def stt_cursor(self) -> int:
        with self._lock:
            return self._stt_cursor

    def reset(self, *, end_index: int | None = None) -> None:
        """Clear audio and re-anchor the timeline.

        Re-anchoring keeps absolute indices monotonic across a rearm so a stale
        detection index can never be mistaken for a fresh one.
        """
        with self._lock:
            self._data.fill(0)
            self._write_pos = 0
            anchor = self._end_index if end_index is None else int(end_index)
            self._end_index = anchor
            self._stt_cursor = anchor
            # No audio is held after a re-anchor: the timeline advances but the
            # ring contents are gone, so spanning must not claim the pre-rearm
            # samples are still readable.
            self._valid_since_reset = 0
            # Slot i holds absolute index i % capacity, so the write head must
            # stay consistent with the anchor after a re-anchor.
            self._write_pos = anchor % self._capacity

    def append(self, pcm_s16le: bytes) -> int:
        """Append PCM and return the new absolute end index."""
        if not pcm_s16le:
            return self.end_index
        flat = np.frombuffer(pcm_s16le, dtype="<i2").reshape(-1)
        n = int(flat.size)
        if n == 0:
            return self.end_index
        with self._lock:
            # Always advance the absolute timeline by every sample received, even
            # when the ring is too small to hold them all: the index is the
            # contract the handoff relies on, and a short ring must not silently
            # rewind it.
            self._end_index += n
            # Slot i holds absolute index i % capacity; the write head follows
            # the absolute timeline, not a slot-local counter.
            self._valid_since_reset += n
            offset = 0
            pos = (self._end_index - n) % self._capacity
            while offset < n:
                space = self._capacity - pos
                chunk = min(n - offset, space)
                stop = pos + chunk
                self._data[pos:stop] = flat[offset : offset + chunk]
                pos = stop % self._capacity
                offset += chunk
            self._write_pos = self._end_index % self._capacity
            return self._end_index

    def available_span(self) -> tuple[int, int]:
        """Return the absolute ``(start, end)`` sample span still held."""
        with self._lock:
            held = min(self._valid_since_reset, self._capacity)
            return (self._end_index - held, self._end_index)

    def covers(self, index: int) -> bool:
        start, end = self.available_span()
        return start <= index <= end

    def read(self, start_index: int, end_index: int) -> bytes:
        """Return ``[start_index, end_index)`` clamped to what is held."""
        with self._lock:
            span_start, span_end = self.available_span()
            lo = max(int(start_index), span_start)
            hi = min(int(end_index), span_end)
            if hi <= lo:
                return b""
            return self._slice_locked(lo, hi)

    def _slice_locked(self, lo: int, hi: int) -> bytes:
        """Return absolute ``[lo, hi)`` as bytes; caller clamps to held span."""
        count = hi - lo
        if count <= 0:
            return b""
        # Ring slot of absolute index i is i % capacity; the oldest held index
        # is end_index - capacity, so order is preserved by reading forward.
        slots = (np.arange(lo, hi, dtype=np.int64)) % self._capacity
        return self._data[slots].astype("<i2", copy=False).tobytes()

    def drain_after_cursor(self) -> bytes:
        """Return audio appended after the STT cursor and advance it."""
        with self._lock:
            lo = max(self._stt_cursor, self.available_span()[0])
            hi = self._end_index
            if hi <= lo:
                self._stt_cursor = hi
                return b""
            data = self._slice_locked(lo, hi)
            self._stt_cursor = hi
            return data

    def flush_from(self, trim_index: int) -> bytes:
        """Emit ``[max(trim_index, cursor), end)`` once and resync the cursor.

        This is the atomic handoff: the trim slice and the cursor move together
        under the ring lock, so the live path resumes after the emitted audio
        instead of overlapping it.

        The region below the STT cursor has already been delivered live, so it
        is never re-emitted. That keeps the wake boundary exactly-once even
        when the capture thread streamed part of the preroll window before the
        detection thread published its boundary.
        """
        with self._lock:
            lo = max(int(trim_index), self._stt_cursor, self.available_span()[0])
            hi = self._end_index
            if hi <= lo:
                self._stt_cursor = hi
                return b""
            data = self._slice_locked(lo, hi)
            self._stt_cursor = hi
            return data
