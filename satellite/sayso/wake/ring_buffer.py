"""Preallocated int16 ring buffer for wake-word windows."""

from __future__ import annotations

import numpy as np


class Int16RingBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._data = np.zeros(capacity, dtype=np.int16)
        self._size = 0
        self._write_pos = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return self._size

    def clear(self) -> None:
        self._size = 0
        self._write_pos = 0

    def fill_silence(self) -> None:
        self._data.fill(0)
        self._size = self._capacity
        self._write_pos = 0

    def extend(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        flat = np.asarray(samples, dtype=np.int16).reshape(-1)
        n = int(flat.size)
        if n >= self._capacity:
            flat = flat[-self._capacity :]
            n = self._capacity

        offset = 0
        while offset < n:
            space = self._capacity - self._write_pos
            chunk = min(n - offset, space)
            stop = self._write_pos + chunk
            self._data[self._write_pos:stop] = flat[offset : offset + chunk]
            self._write_pos = stop % self._capacity
            offset += chunk

        self._size = min(self._capacity, self._size + n)

    def view(self) -> np.ndarray:
        if self._size < self._capacity:
            return self._data[: self._size].copy()
        start = self._write_pos
        if start == 0:
            return self._data.copy()
        return np.concatenate((self._data[start:], self._data[:start]))
