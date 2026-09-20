"""NanoWakeWord ONNX wake-word provider.

Uses nanowakeword.NanoInterpreter on processed PCM only. Does not open a
microphone or add a second capture path.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .buffer import WakeAudioBuffer
from .detection import Detection
from .livekit import HOP_SAMPLES, SAMPLE_RATE, WINDOW_SAMPLES
from .mining import HardNegativeMiner

_LOGGER = logging.getLogger(__name__)

# NanoInterpreter zeros scores until prediction_buffer has 5 entries.
_WARMUP_HOPS = 5


class NanoWakeWordProvider:
    def __init__(
        self,
        model_path: Path,
        phrase: str,
        threshold: float = 0.65,
        refractory_seconds: float = 2.0,
        miner: Optional[HardNegativeMiner] = None,
    ) -> None:
        self._model_path = Path(model_path)
        self._phrase = phrase
        self._threshold = float(threshold)
        self._refractory = float(refractory_seconds)
        self._miner = miner
        self._enabled = False
        self._suspended = False
        self._available = False
        self._interpreter = None
        self._last_fire = 0.0
        self._logged_keys = False
        self._last_score_log = 0.0
        self._max_score_window = 0.0
        self._stream_primed = False
        self._load()

    def _load(self) -> None:
        if not self._model_path.is_file():
            _LOGGER.error(
                "NanoWakeWord model missing: %s. Wake detection is disabled. "
                "Train a prototype model (see satellite/models/README.md), then run: "
                "sayso-satellite test-wake-word",
                self._model_path,
            )
            return
        try:
            from nanowakeword import NanoInterpreter

            self._interpreter = NanoInterpreter.load_model(str(self._model_path))
            self._available = True
            self._warmup_stream()
            _LOGGER.info("Loaded NanoWakeWord model %s", self._model_path)
        except Exception:
            _LOGGER.exception(
                "Failed to load NanoWakeWord model %s (fail closed)", self._model_path
            )
            self._interpreter = None
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
        self._stream_primed = False
        if self._interpreter is not None:
            self._interpreter.reset()
            self._warmup_stream()

    def _warmup_stream(self) -> None:
        if self._interpreter is None:
            return
        silence = np.zeros(HOP_SAMPLES, dtype=np.int16)
        for _ in range(_WARMUP_HOPS):
            self._interpreter.predict(silence)

    def shutdown(self) -> None:
        self.stop()
        self._interpreter = None

    def _window_hop_chunks(self, window: np.ndarray) -> list[np.ndarray]:
        pcm = window.astype(np.int16, copy=False)
        if self._stream_primed:
            return [pcm[-HOP_SAMPLES:]]
        # NanoInterpreter treats each predict() as new audio and ignores
        # chunks shorter than 1280 samples. Keep remainders that meet that floor.
        chunks: list[np.ndarray] = []
        for offset in range(0, pcm.size, HOP_SAMPLES):
            piece = pcm[offset : offset + HOP_SAMPLES]
            if piece.size < 1280:
                break
            chunks.append(piece)
        return chunks

    def feed_window_score(self, window: np.ndarray) -> float:
        """Hop-feed one window and return max chunk score (eval path; no fire/reset)."""
        if not self._available or self._interpreter is None:
            return 0.0
        if window.size < WINDOW_SAMPLES:
            return 0.0

        max_score = 0.0
        priming = not self._stream_primed
        for chunk in self._window_hop_chunks(window):
            result = self._interpreter.predict(chunk)
            max_score = max(max_score, float(getattr(result, "score", 0.0)))
        if priming:
            self._stream_primed = True
        return max_score

    def predict_window(
        self,
        window: np.ndarray,
        sample_index: int | None = None,
    ) -> Optional[Detection]:
        if not self._available or self._interpreter is None:
            return None
        if not self._enabled or self._suspended:
            return None
        if window.size < WINDOW_SAMPLES:
            return None

        now = time.monotonic()
        priming = not self._stream_primed
        max_score = 0.0
        for chunk in self._window_hop_chunks(window):
            result = self._interpreter.predict(chunk)
            score = float(getattr(result, "score", 0.0))
            max_score = max(max_score, score)

            if not self._logged_keys:
                _LOGGER.info(
                    "NanoWakeWord predict score=%.4f thresh=%.3f model=%s",
                    score,
                    self._threshold,
                    self._model_path.stem,
                )
                self._logged_keys = True

            if score >= self._threshold and (
                not self._last_fire or (now - self._last_fire) >= self._refractory
            ):
                if self._miner is not None:
                    self._miner.offer(score, window)
                self._last_fire = now
                self._stream_primed = False
                self._interpreter.reset()
                self._warmup_stream()
                _LOGGER.info(
                    "Wake phrase detected phrase=%r confidence=%.3f (no audio retained)",
                    self._phrase,
                    score,
                )
                return Detection(
                    phrase=self._phrase,
                    confidence=score,
                    timestamp=now,
                    sample_index=sample_index,
                )

        if priming:
            self._stream_primed = True

        self._max_score_window = max(self._max_score_window, max_score)
        if now - self._last_score_log >= 1.0:
            _LOGGER.info(
                "NanoWakeWord score=%.4f max=%.4f thresh=%.3f",
                max_score,
                self._max_score_window,
                self._threshold,
            )
            self._last_score_log = now
            self._max_score_window = 0.0

        if self._miner is not None:
            self._miner.offer(max_score, window)

        return None

    def process_pcm(self, pcm_s16le: bytes, sample_rate: int = 16000) -> Optional[Detection]:
        """Synchronous helper retained for tests and diagnostics."""
        if sample_rate != SAMPLE_RATE or not pcm_s16le:
            return None
        buffer = WakeAudioBuffer(WINDOW_SAMPLES, HOP_SAMPLES)
        if not buffer.feed(pcm_s16le):
            return None
        return self.predict_window(buffer.window())
