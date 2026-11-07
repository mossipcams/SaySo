from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np

from satellite.sayso.wake.worker import FifoWindowQueue, WakeInferenceWorker


def test_fifo_queue_keeps_order_and_does_not_drop_below_capacity() -> None:
    queue = FifoWindowQueue()
    for i in range(25):
        queue.offer(np.array([i], dtype=np.int16), 100 + i)
    assert queue.dropped == 0
    taken = [queue.take(timeout=0.1) for _ in range(25)]
    assert all(job is not None for job in taken)
    assert [int(job[0][0]) for job in taken] == list(range(25))
    assert [job[1] for job in taken] == [100 + i for i in range(25)]


def test_fifo_queue_stores_independent_copy() -> None:
    queue = FifoWindowQueue()
    source = np.array([1, 2, 3], dtype=np.int16)
    queue.offer(source)
    source[...] = 9
    window, _index = queue.take(timeout=0.1)
    assert list(window) == [1, 2, 3]


def test_fifo_queue_drops_oldest_when_full() -> None:
    queue = FifoWindowQueue(capacity=3)
    for i in range(5):
        queue.offer(np.array([i], dtype=np.int16))
    assert queue.dropped == 2
    taken = [int(queue.take(timeout=0.1)[0][0]) for _ in range(3)]
    assert taken == [2, 3, 4]


def test_fifo_queue_clear_empties() -> None:
    queue = FifoWindowQueue()
    queue.offer(np.array([1], dtype=np.int16))
    queue.clear()
    assert queue.take(timeout=0.1) is None


def test_worker_clear_empties_pending_windows() -> None:
    started = threading.Event()

    def predict(_window: np.ndarray):
        started.set()
        return None

    worker = WakeInferenceWorker(predict, poll_timeout=0.5)
    worker.submit(np.array([1], dtype=np.int16))
    worker.clear()
    worker.start(lambda _detection: None)
    try:
        assert not started.wait(timeout=0.3)
    finally:
        worker.shutdown()


def test_slow_predict_scores_every_window_in_order() -> None:
    seen: list[tuple[int, int | None]] = []
    calls = {"n": 0}

    def predict(window: np.ndarray, sample_index: int | None = None):
        calls["n"] += 1
        time.sleep(0.27 if calls["n"] % 10 == 0 else 0.04)
        seen.append((int(window[0]), sample_index))
        return None

    worker = WakeInferenceWorker(predict, poll_timeout=0.05)
    worker.start(lambda _detection: None)
    try:
        for i in range(100):
            window = np.array([i], dtype=np.int16)
            worker.submit(window, 200 * i)
            time.sleep(0.064)
        deadline = time.monotonic() + 20.0
        while len(seen) < 100 and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        worker.shutdown()
    assert seen == [(i, 200 * i) for i in range(100)]


def test_worker_runs_predict_off_capture_thread() -> None:
    started = threading.Event()
    seen: list[int] = []

    def predict(window: np.ndarray):
        started.set()
        seen.append(int(window[0]))
        return None

    worker = WakeInferenceWorker(predict, poll_timeout=0.05)
    worker.start(lambda _detection: None)
    try:
        worker.submit(np.array([7], dtype=np.int16))
        assert started.wait(timeout=1.0)
        assert seen == [7]
    finally:
        worker.shutdown()


def test_worker_survives_predict_and_callback_exceptions() -> None:
    calls = {"predict": 0, "callback": 0}
    first_predict_failed = threading.Event()
    second_predict_done = threading.Event()
    third_predict_done = threading.Event()

    def predict(_window: np.ndarray):
        calls["predict"] += 1
        if calls["predict"] == 1:
            first_predict_failed.set()
            raise RuntimeError("onnx failed")
        if calls["predict"] == 2:
            second_predict_done.set()
            return SimpleNamespace(confidence=0.9, phrase="SaySo", timestamp=0.0)
        third_predict_done.set()
        return None

    def on_detection(_detection) -> None:
        calls["callback"] += 1
        if calls["callback"] == 1:
            raise RuntimeError("callback failed")

    worker = WakeInferenceWorker(predict, poll_timeout=0.05)
    worker.start(on_detection)
    try:
        worker.submit(np.array([1], dtype=np.int16))
        assert first_predict_failed.wait(timeout=1.0)
        worker.submit(np.array([2], dtype=np.int16))
        assert second_predict_done.wait(timeout=1.0)
        worker.submit(np.array([3], dtype=np.int16))
        assert third_predict_done.wait(timeout=1.0)
        assert calls["predict"] == 3
        assert calls["callback"] == 1
    finally:
        worker.shutdown()


def test_worker_forwards_sample_index_to_predict() -> None:
    seen: list[int | None] = []

    def predict(window: np.ndarray, sample_index: int | None = None):
        seen.append(sample_index)
        return None

    worker = WakeInferenceWorker(predict, poll_timeout=0.05)
    worker.start(lambda _detection: None)
    try:
        worker.submit(np.array([1], dtype=np.int16), 4096)
        deadline = threading.Event()
        deadline.wait(0.5)
    finally:
        worker.shutdown()
    assert seen == [4096]


def test_worker_tolerates_predict_without_sample_index() -> None:
    seen: list[int] = []

    def predict(window: np.ndarray):
        seen.append(int(window[0]))
        return None

    worker = WakeInferenceWorker(predict, poll_timeout=0.05)
    worker.start(lambda _detection: None)
    try:
        worker.submit(np.array([5], dtype=np.int16), 123)
        deadline = threading.Event()
        deadline.wait(0.5)
    finally:
        worker.shutdown()
    assert seen == [5]
