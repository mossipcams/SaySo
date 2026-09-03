"""Tests for the non-blocking wake inference worker."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np

from satellite.sayso.wake.worker import LatestWindowQueue, WakeInferenceWorker


def test_latest_window_queue_replaces_stale_window() -> None:
    queue = LatestWindowQueue()
    queue.offer(np.array([1], dtype=np.int16))
    queue.offer(np.array([2], dtype=np.int16))
    assert queue.take(timeout=0.1)[0] == 2  # type: ignore[index]


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
