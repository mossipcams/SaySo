"""Non-blocking wake inference worker with a single-slot latest window queue."""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Optional

import numpy as np

from .detection import Detection

_LOGGER = logging.getLogger(__name__)


class LatestWindowQueue:
    """Drop stale windows; keep only the newest pending inference job."""

    def __init__(self) -> None:
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()

    def offer(self, window: np.ndarray) -> None:
        with self._lock:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._queue.put_nowait(window)

    def take(self, timeout: float | None = None) -> Optional[np.ndarray]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


class WakeInferenceWorker:
    def __init__(
        self,
        predict: Callable[[np.ndarray], Optional[Detection]],
        *,
        poll_timeout: float = 0.1,
    ) -> None:
        self._predict = predict
        self._poll_timeout = poll_timeout
        self._queue = LatestWindowQueue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._on_detection: Callable[[Detection], None] | None = None

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

    def submit(self, window: np.ndarray) -> None:
        self._queue.offer(window)

    def _run(self) -> None:
        while not self._stop.is_set():
            window = self._queue.take(timeout=self._poll_timeout)
            if window is None:
                continue
            try:
                detection = self._predict(window)
            except Exception:
                _LOGGER.exception("wake predict failed")
                continue
            if detection is not None and self._on_detection is not None:
                try:
                    self._on_detection(detection)
                except Exception:
                    _LOGGER.exception("wake on_detection failed")
