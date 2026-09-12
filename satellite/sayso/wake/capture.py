"""Sample-ordered capture timeline and the one deliberate resample to 16 kHz.

The satellite captures at the device's native rate and resamples exactly once,
here, before any audio is used for wake inference or sent to Home Assistant.
A per-block resample (``np.interp`` inside each block) restarts the filter every
block and leaves a discontinuity at every boundary; :class:`CaptureResampler`
keeps one continuous windowed-sinc history so the output is a seamless 16 kHz
stream regardless of how the audio server chunks it.

Output sample timing is exact. 44100/16000 reduces to 441/160, so an output
sample lands on a whole input sample only once every 160 outputs; the other 159
fall between two of them. Rounding those to the nearest input sample -- which is
what convolving a single integer-spaced kernel centred on ``floor(position)``
does -- is +/-11 us of sample jitter, and it measures as roughly -33 dBc of
in-band spurs on a 1 kHz tone. Each of the 160 fractional phases therefore gets
its own kernel, evaluated at that phase's true offset (:func:`_phase_kernels`).

Every block appended to :class:`WakeCaptureRing` is stamped with its absolute
end sample index. Wake detection, preroll trimming, and the STT handoff all
reference that index, so nothing depends on when a thread happened to run.
"""

from __future__ import annotations

import math
import threading
from math import gcd

import numpy as np

# Half-width, in input samples, of the windowed-sinc low-pass used when
# decimating. Sized by alias rejection in the sibilant band, not by feel: a
# 44.1 kHz mic carries real /s/ and /sh/ energy at 8.5-11 kHz, and whatever the
# filter fails to reject there folds straight back on top of the 5-7.5 kHz that
# distinguishes those consonants. Measured rejection at 9 kHz / 10 kHz:
#
#     half_taps=16   -13.7 dB  -27.0 dB   <- audible sibilant smear
#     half_taps=32   -26.0 dB  -78.0 dB
#     half_taps=48   -47.3 dB  -85.1 dB   <- chosen
#
# 48 also flattens the passband to -0.04 dB at 7 kHz (16 was -2.0 dB). The
# kernels are precomputed per phase, so the per-block cost is one gather plus
# one einsum either way: 10 s of audio costs 41 ms instead of 24 ms.
# ponytail: 48 taps, revisit only if a Pi-class CPU can't hold the capture cadence.
_FILTER_HALF_TAPS = 48


def gain_scalar_from_db(gain_db: float, ceiling: float = 8.0) -> float:
    """Convert a decibel gain to a bounded linear scalar.

    The ceiling keeps a misconfigured gain from driving the signal straight into
    saturation; clipping is still counted and reported by the capture tap.
    """
    if not math.isfinite(gain_db):
        return 1.0
    return float(min(ceiling, max(0.0, 10.0 ** (gain_db / 20.0))))


def _phase_kernels(
    phases: int,
    input_rate: int,
    output_rate: int,
    half_taps: int = _FILTER_HALF_TAPS,
) -> np.ndarray:
    """Build one anti-alias kernel per fractional output phase.

    Row ``p`` resamples an output sample sitting ``p / phases`` of an input
    sample after ``floor(position)``. Cutoff is the *output* Nyquist relative to
    the input rate, so the kernel attenuates everything a downsample would
    otherwise fold back into the speech band. A Blackman window, evaluated
    continuously rather than on the integer grid, keeps the stopband well below
    speech level at every phase.

    Each row is normalised to unit sum so DC gain is 1.0 regardless of phase; an
    unnormalised bank makes the output amplitude wobble at the phase rate.
    """
    cutoff = min(0.5, 0.5 * output_rate / input_rate)
    taps = 2 * half_taps + 1
    # Distance from the output position to each contributing input sample.
    offsets = np.arange(taps, dtype=np.float64) - half_taps
    fractions = np.arange(phases, dtype=np.float64) / float(phases)
    distance = offsets[None, :] - fractions[:, None]
    kernels = 2.0 * cutoff * np.sinc(2.0 * cutoff * distance)
    # Blackman over the support [-(half+1), half+1], so a tap that lands at the
    # very edge is windowed to zero instead of truncated mid-ripple.
    u = distance / float(half_taps + 1)
    kernels *= 0.42 + 0.5 * np.cos(np.pi * u) + 0.08 * np.cos(2.0 * np.pi * u)
    totals = kernels.sum(axis=1, keepdims=True)
    np.divide(kernels, totals, out=kernels, where=totals != 0.0)
    return kernels


class CaptureResampler:
    """Streaming polyphase resampler with continuous filter state.

    Input and output are int16 little-endian PCM. Output position is an exact
    integer ratio (``output k`` sits at input ``k * M / L`` for the reduced
    fraction ``input_rate/output_rate = M/L``), so long runs cannot drift the
    way a floating-point accumulator would, and the phase ``k % L`` selects a
    kernel built for that exact fractional offset.

    An output sample is emitted only once every tap it needs has arrived, so the
    result is identical no matter how the audio server chunks the stream. The
    cost is ``half_taps`` input samples of latency (1.1 ms at 44.1 kHz), paid
    once at the start rather than per block.
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
        # Reduced ratio: output k sits at input k * _step / _phases.
        divisor = gcd(self._input_rate, self._output_rate)
        self._step = self._input_rate // divisor
        self._phases = self._output_rate // divisor
        self._kernels = (
            None
            if self._passthrough
            else _phase_kernels(
                self._phases, self._input_rate, self._output_rate, self._half_taps
            )
        )
        self._taps = 2 * self._half_taps + 1
        self._history = np.zeros(0, dtype=np.float64)
        # Absolute input sample index of history[0]. Trimming advances this so
        # indices are always interpreted against the retained window.
        self._history_start = 0
        # Index of the next output sample to produce.
        self._position = 0
        self._primed = False

    @property
    def input_rate(self) -> int:
        return self._input_rate

    @property
    def output_rate(self) -> int:
        return self._output_rate

    def reset(self) -> None:
        self._history = np.zeros(0, dtype=np.float64)
        self._history_start = 0
        self._position = 0
        self._primed = False

    def process(self, pcm_s16le: bytes) -> bytes:
        """Append input PCM and return every output sample now computable."""
        if not pcm_s16le:
            return b""
        incoming = np.frombuffer(pcm_s16le, dtype="<i2")
        if incoming.size == 0:
            return b""
        if self._passthrough:
            return incoming.astype("<i2", copy=False).tobytes()

        if not self._primed:
            # Prime with a half-window of silence so the very first output
            # samples have full left context. Without it the stream opens with
            # `half_taps` samples filtered by a truncated kernel, which is the
            # same defect as a per-block reset, just confined to the start.
            self._history = np.zeros(self._half_taps, dtype=np.float64)
            self._history_start = -self._half_taps
            self._primed = True

        self._history = np.concatenate((self._history, incoming.astype(np.float64)))

        # Emit only while every tap is present: the newest sample an output at
        # input index c needs is c + half_taps.
        last_full = self._history_start + self._history.size - 1 - self._half_taps
        count = self._outputs_through(last_full) - self._position
        if count <= 0:
            self._trim_history()
            return b""

        indices = np.arange(
            self._position, self._position + count, dtype=np.int64
        )
        numerators = indices * self._step
        # Offset of each output's leftmost tap within the retained history.
        starts = numerators // self._phases - self._half_taps - self._history_start
        # _trim_history never drops below the next output's leftmost tap, so this
        # holds. It is asserted because a negative start would not raise: numpy
        # would wrap to the end of the history and silently resample stale audio.
        assert starts[0] >= 0, "history trimmed past a tap still needed"
        taps = starts[:, None] + np.arange(self._taps, dtype=np.int64)[None, :]
        assert self._kernels is not None
        kernels = self._kernels[numerators % self._phases]
        values = np.einsum("ij,ij->i", self._history[taps], kernels)
        self._position += count

        out = np.clip(np.rint(values), -32768, 32767).astype("<i2")
        # Drop history that can no longer contribute to a future output sample.
        self._trim_history()
        return out.tobytes()

    def _outputs_through(self, last_input_index: int) -> int:
        """Number of output samples whose position is <= ``last_input_index``.

        ``k * step // phases <= n`` iff ``k <= ((n + 1) * phases - 1) // step``,
        so the count is available in closed form instead of by stepping.
        """
        if last_input_index < 0:
            return 0
        return ((last_input_index + 1) * self._phases - 1) // self._step + 1

    def _trim_history(self) -> None:
        # Keep everything the next output sample still needs, and drop the rest.
        keep_from = (
            self._position * self._step
        ) // self._phases - self._half_taps
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
        # Absolute index where the current capture epoch began. Audio before it
        # was never held by this ring, as opposed to held and then overwritten.
        self._origin = 0
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

    @property
    def origin(self) -> int:
        """Absolute index at which the current capture epoch began.

        A read below this never existed in the ring, so failing to satisfy it is
        a cold start rather than a shortfall. A read between here and
        ``available_span()[0]`` is audio the ring did hold and has since
        overwritten -- that one is a real underflow.
        """
        with self._lock:
            return self._origin

    def reset(self, *, end_index: int | None = None) -> None:
        """Clear audio and re-anchor the timeline.

        Re-anchoring keeps absolute indices monotonic across a rearm so a stale
        detection index can never be mistaken for a fresh one.
        """
        with self._lock:
            self._data.fill(0)
            anchor = self._end_index if end_index is None else int(end_index)
            self._end_index = anchor
            self._stt_cursor = anchor
            self._origin = anchor
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
