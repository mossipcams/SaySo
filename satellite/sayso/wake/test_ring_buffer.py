"""Tests for the int16 ring buffer."""

from __future__ import annotations

import numpy as np

from satellite.sayso.wake.ring_buffer import Int16RingBuffer


def test_ring_buffer_preserves_chronological_order_when_wrapped() -> None:
    ring = Int16RingBuffer(4)
    ring.extend(np.array([1, 2, 3, 4], dtype=np.int16))
    ring.extend(np.array([5], dtype=np.int16))
    assert np.array_equal(ring.view(), np.array([2, 3, 4, 5], dtype=np.int16))


def test_ring_buffer_fill_silence_marks_window_full() -> None:
    ring = Int16RingBuffer(8)
    ring.fill_silence()
    assert ring.size == 8
    assert np.all(ring.view() == 0)


def test_ring_buffer_extend_uses_vectorized_writes(monkeypatch) -> None:
    class NoIterArray(np.ndarray):
        def __iter__(self):
            raise AssertionError("extend must not iterate samples in Python")

    original_asarray = np.asarray

    def asarray_no_iter(a, dtype=None):
        arr = original_asarray(a, dtype=dtype)
        return arr.view(NoIterArray)

    monkeypatch.setattr(
        "satellite.sayso.wake.ring_buffer.np.asarray",
        asarray_no_iter,
    )
    ring = Int16RingBuffer(4)
    ring.extend(np.array([1, 2, 3, 4, 5, 6], dtype=np.int16))
    assert np.array_equal(ring.view(), np.array([3, 4, 5, 6], dtype=np.int16))
