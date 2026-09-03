"""LiveKit ONNX wake-word provider.

Uses livekit.wakeword.WakeWordModel.predict() only. Does not use
WakeWordListener (that would open a second capture stream) and does not
install or call OpenWakeWord.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .buffer import WakeAudioBuffer
from .detection import Detection

_LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WINDOW_SAMPLES = SAMPLE_RATE * 2  # LiveKit predict wants ~2s
HOP_SAMPLES = 2560  # 160 ms


class LiveKitWakeWordProvider:
    def __init__(
        self,
        model_path: Path,
        phrase: str,
        threshold: float = 0.65,
        refractory_seconds: float = 2.0,
    ) -> None:
        self._model_path = Path(model_path)
        self._phrase = phrase
        self._threshold = float(threshold)
        self._refractory = float(refractory_seconds)
        self._enabled = False
        self._suspended = False
        self._available = False
        self._model = None
        self._score_key: Optional[str] = None
        self._last_fire = 0.0
        self._logged_keys = False
        self._last_score_log = 0.0
        self._max_score_window = 0.0
        self._load()

    def _load(self) -> None:
        if not self._model_path.is_file():
            _LOGGER.error(
                "Wake model missing: %s. Wake detection is disabled. "
                "Place a LiveKit-exported ONNX classifier at that path, then run: "
                "sayso-satellite test-wake-word",
                self._model_path,
            )
            return
        try:
            from livekit.wakeword import WakeWordModel

            self._model = WakeWordModel(models=[str(self._model_path)])
            self._score_key = self._model_path.stem
            self._available = True
            _LOGGER.info("Loaded LiveKit wake model %s", self._model_path)
        except Exception:
            _LOGGER.exception("Failed to load LiveKit wake model %s (fail closed)", self._model_path)
            self._model = None
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def start(self) -> None:
        self._enabled = True
        self._suspended = False

    def stop(self) -> None:
        self._enabled = False

    def suspend(self) -> None:
        self._suspended = True

    def resume(self) -> None:
        self._suspended = False

    def reset(self) -> None:
        self._last_fire = 0.0

    def shutdown(self) -> None:
        self.stop()
        self._model = None

    def predict_window(self, window: np.ndarray) -> Optional[Detection]:
        if not self._available or self._model is None:
            return None
        if not self._enabled or self._suspended:
            return None
        if window.size < WINDOW_SAMPLES:
            return None

        scores = self._model.predict(window)
        if not self._logged_keys:
            _LOGGER.info(
                "Wake predict keys=%s score_key=%s thresh=%.3f",
                list(scores.keys()) if scores else None,
                self._score_key,
                self._threshold,
            )
            self._logged_keys = True
        score = float(scores.get(self._score_key, 0.0)) if self._score_key else 0.0
        if not scores:
            _LOGGER.info("Wake predict returned empty scores")
            return None
        if self._score_key not in scores:
            score = float(next(iter(scores.values())))

        now = time.monotonic()
        self._max_score_window = max(self._max_score_window, score)
        if now - self._last_score_log >= 1.0:
            _LOGGER.info(
                "Wake score=%.4f max=%.4f key=%s thresh=%.3f",
                score,
                self._max_score_window,
                self._score_key,
                self._threshold,
            )
            self._last_score_log = now
            self._max_score_window = 0.0
        if score < self._threshold:
            return None
        if self._last_fire and (now - self._last_fire) < self._refractory:
            return None

        self._last_fire = now
        _LOGGER.info("Wake phrase detected phrase=%r confidence=%.3f (no audio retained)", self._phrase, score)
        return Detection(phrase=self._phrase, confidence=score, timestamp=now)

    def process_pcm(self, pcm_s16le: bytes, sample_rate: int = 16000) -> Optional[Detection]:
        """Synchronous helper retained for tests and diagnostics."""
        if sample_rate != SAMPLE_RATE or not pcm_s16le:
            return None
        buffer = WakeAudioBuffer(WINDOW_SAMPLES, HOP_SAMPLES)
        if not buffer.feed(pcm_s16le):
            return None
        return self.predict_window(buffer.window())
