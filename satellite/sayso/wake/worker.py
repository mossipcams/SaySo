"""Non-blocking wake inference worker with a single-slot latest window queue."""

from __future__ import annotations

import logging
import queue
import threading
from inspect import Parameter, signature
from typing import Callable, Optional

import numpy as np

from .detection import Detection

_LOGGER = logging.getLogger(__name__)


def _accepts_sample_index(predict: Callable[..., Optional[Detection]]) -> bool:
    """Whether ``predict`` accepts a ``sample_index`` keyword.

    Checked once by signature instead of by catching TypeError around the call,
    which would also swallow a genuine TypeError raised inside predict.
    """
    try:
        params = signature(predict).parameters
    except (TypeError, ValueError):
        return False
    if "sample_index" in params:
        return True
    return any(p.kind is Parameter.VAR_KEYWORD for p in params.values())


class LatestWindowQueue:
    """Drop stale windows; keep only the newest pending inference job.

    Each job carries the absolute capture sample index at which its window
    ends, so a detection can be anchored to the capture timeline even though
    inference runs off-thread and may complete one hop later.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[np.ndarray, int | None]] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()

    def offer(self, window: np.ndarray, sample_index: int | None = None) -> None:
        with self._lock:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._queue.put_nowait((window, sample_index))

    def take(self, timeout: float | None = None) -> Optional[tuple[np.ndarray, int | None]]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


class WakeInferenceWorker:
    def __init__(
        self,
        predict: Callable[..., Optional[Detection]],
        *,
        poll_timeout: float = 0.1,
    ) -> None:
        self._predict = predict
        self._poll_timeout = poll_timeout
        self._queue = LatestWindowQueue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._on_detection: Callable[[Detection], None] | None = None
        self._forwards_index = _accepts_sample_index(predict)

    def start(self, on_detection: Callable[[Detection], None]) -> None:
        self._on_detection = on_detection
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="sayso-wake-worker", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def submit(self, window: np.ndarray, sample_index: int | None = None) -> None:
        self._queue.offer(window, sample_index)

    def _run(self) -> None:
        while not self._stop.is_set():
            job = self._queue.take(timeout=self._poll_timeout)
            if job is None:
                continue
            window, sample_index = job
            try:
                if self._forwards_index:
                    detection = self._predict(window, sample_index=sample_index)
                else:
                    detection = self._predict(window)
            except Exception:
                _LOGGER.exception("wake predict failed")
                continue
            if detection is not None and self._on_detection is not None:
                try:
                    self._on_detection(detection)
                except Exception:
                    _LOGGER.exception("wake on_detection failed")
