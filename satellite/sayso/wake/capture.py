
from __future__ import annotations

import math
import threading
from math import gcd

import numpy as np

_FILTER_HALF_TAPS = 48


def gain_scalar_from_db(gain_db: float, ceiling: float = 8.0) -> float:
    if not math.isfinite(gain_db):
        return 1.0
    return float(min(ceiling, max(0.0, 10.0 ** (gain_db / 20.0))))


def _phase_kernels(
    phases: int,
    input_rate: int,
    output_rate: int,
    half_taps: int = _FILTER_HALF_TAPS,
) -> np.ndarray:
    cutoff = min(0.5, 0.5 * output_rate / input_rate)
    taps = 2 * half_taps + 1
    offsets = np.arange(taps, dtype=np.float64) - half_taps
    fractions = np.arange(phases, dtype=np.float64) / float(phases)
    distance = offsets[None, :] - fractions[:, None]
    kernels = 2.0 * cutoff * np.sinc(2.0 * cutoff * distance)
    u = distance / float(half_taps + 1)
    kernels *= 0.42 + 0.5 * np.cos(np.pi * u) + 0.08 * np.cos(2.0 * np.pi * u)
    totals = kernels.sum(axis=1, keepdims=True)
    np.divide(kernels, totals, out=kernels, where=totals != 0.0)
    return kernels


class CaptureResampler:

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
        self._history_start = 0
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
        if not pcm_s16le:
            return b""
        incoming = np.frombuffer(pcm_s16le, dtype="<i2")
        if incoming.size == 0:
            return b""
        if self._passthrough:
            return incoming.astype("<i2", copy=False).tobytes()

        if not self._primed:
            self._history = np.zeros(self._half_taps, dtype=np.float64)
            self._history_start = -self._half_taps
            self._primed = True

        self._history = np.concatenate((self._history, incoming.astype(np.float64)))

        last_full = self._history_start + self._history.size - 1 - self._half_taps
        count = self._outputs_through(last_full) - self._position
        if count <= 0:
            self._trim_history()
            return b""

        indices = np.arange(
            self._position, self._position + count, dtype=np.int64
        )
        numerators = indices * self._step
        starts = numerators // self._phases - self._half_taps - self._history_start
        assert starts[0] >= 0, "history trimmed past a tap still needed"
        taps = starts[:, None] + np.arange(self._taps, dtype=np.int64)[None, :]
        assert self._kernels is not None
        kernels = self._kernels[numerators % self._phases]
        values = np.einsum("ij,ij->i", self._history[taps], kernels)
        self._position += count

        out = np.clip(np.rint(values), -32768, 32767).astype("<i2")
        self._trim_history()
        return out.tobytes()

    def _outputs_through(self, last_input_index: int) -> int:
        if last_input_index < 0:
            return 0
        return ((last_input_index + 1) * self._phases - 1) // self._step + 1

    def _trim_history(self) -> None:
        keep_from = (
            self._position * self._step
        ) // self._phases - self._half_taps
        drop = keep_from - self._history_start
        if drop > 0:
            drop = min(drop, self._history.size)
            self._history = self._history[drop:]
            self._history_start += drop


class WakeCaptureRing:

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = int(capacity)
        self._data = np.zeros(self._capacity, dtype=np.int16)
        self._end_index = 0
        self._write_pos = 0
        self._valid_since_reset = 0
        self._stt_cursor = 0
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
        with self._lock:
            return self._origin

    def reset(self, *, end_index: int | None = None) -> None:
        with self._lock:
            self._data.fill(0)
            anchor = self._end_index if end_index is None else int(end_index)
            self._end_index = anchor
            self._stt_cursor = anchor
            self._origin = anchor
            self._valid_since_reset = 0
            self._write_pos = anchor % self._capacity

    def append(self, pcm_s16le: bytes) -> int:
        if not pcm_s16le:
            return self.end_index
        flat = np.frombuffer(pcm_s16le, dtype="<i2").reshape(-1)
        n = int(flat.size)
        if n == 0:
            return self.end_index
        with self._lock:
            self._end_index += n
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
        with self._lock:
            held = min(self._valid_since_reset, self._capacity)
            return (self._end_index - held, self._end_index)

    def covers(self, index: int) -> bool:
        start, end = self.available_span()
        return start <= index <= end

    def read(self, start_index: int, end_index: int) -> bytes:
        with self._lock:
            span_start, span_end = self.available_span()
            lo = max(int(start_index), span_start)
            hi = min(int(end_index), span_end)
            if hi <= lo:
                return b""
            return self._slice_locked(lo, hi)

    def _slice_locked(self, lo: int, hi: int) -> bytes:
        count = hi - lo
        if count <= 0:
            return b""
        slots = (np.arange(lo, hi, dtype=np.int64)) % self._capacity
        return self._data[slots].astype("<i2", copy=False).tobytes()

    def drain_after_cursor(self) -> bytes:
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
        with self._lock:
            lo = max(int(trim_index), self._stt_cursor, self.available_span()[0])
            hi = self._end_index
            if hi <= lo:
                self._stt_cursor = hi
                return b""
            data = self._slice_locked(lo, hi)
            self._stt_cursor = hi
            return data
