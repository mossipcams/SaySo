
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .buffer import WakeAudioBuffer
from .detection import Detection
from .mining import HardNegativeMiner
from .streaming import CachedEmbeddingScorer, single_threaded_ort
from .verifier import WakeVerifier, load_wake_verifier

_LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WINDOW_SAMPLES = SAMPLE_RATE * 2
HOP_SAMPLES = 2560


class LiveKitWakeWordProvider:
    def __init__(
        self,
        model_path: Path,
        phrase: str,
        threshold: float = 0.65,
        refractory_seconds: float = 2.0,
        miner: Optional[HardNegativeMiner] = None,
        verifier_path: Optional[Path] = None,
        verifier_threshold: Optional[float] = None,
        verifier_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        if verifier_factory is not None and verifier_path is not None:
            raise ValueError("verifier_factory and verifier_path are mutually exclusive")
        self._model_path = Path(model_path)
        self._phrase = phrase
        self._threshold = float(threshold)
        self._refractory = float(refractory_seconds)
        self._miner = miner
        self._verifier_path = Path(verifier_path) if verifier_path is not None else None
        self._verifier: Optional[WakeVerifier] = None
        self._enabled = False
        self._suspended = False
        self._available = False
        self._model = None
        self._scorer: Optional[CachedEmbeddingScorer] = None
        self._score_key: Optional[str] = None
        self._last_fire_sample: int | None = None
        self._last_fire_time: float | None = None
        self._logged_keys = False
        self._last_score_log = 0.0
        self._max_score_window = 0.0
        self._load()
        if verifier_factory is not None:
            self._load_verifier_factory(verifier_factory)
        elif self._verifier_path is not None:
            self._load_verifier(verifier_threshold)

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

            with single_threaded_ort():
                self._model = WakeWordModel(models=[str(self._model_path)])
            self._scorer = CachedEmbeddingScorer(self._model)
            self._score_key = self._model_path.stem
            self._available = True
            _LOGGER.info(
                "Loaded LiveKit wake model %s (embedding reuse %s)",
                self._model_path,
                "on" if self._scorer.supported else "OFF - will not keep up with the hop",
            )
        except Exception:
            _LOGGER.exception("Failed to load LiveKit wake model %s (fail closed)", self._model_path)
            self._model = None
            self._available = False

    def _load_verifier_factory(self, factory: Callable[[], Any]) -> None:
        if not self._available:
            return
        try:
            self._verifier = factory()
            _LOGGER.info("Loaded wake verifier %s", type(self._verifier).__name__)
        except Exception:
            _LOGGER.exception("Failed to load wake verifier (fail closed)")
            self._verifier = None
            self._available = False

    def _load_verifier(self, verifier_threshold: Optional[float]) -> None:
        if not self._available:
            return
        if not self._verifier_path.is_file():
            _LOGGER.error(
                "Wake verifier missing: %s. Wake detection is disabled (fail closed).",
                self._verifier_path,
            )
            self._available = False
            return
        try:
            self._verifier = load_wake_verifier(self._verifier_path, threshold=verifier_threshold)
            if self._verifier is not None:
                _LOGGER.info(
                    "Loaded speech-embedding wake verifier %s (threshold=%.3f)",
                    self._verifier_path,
                    self._verifier.threshold,
                )
        except Exception:
            _LOGGER.exception(
                "Failed to load wake verifier %s (fail closed)",
                self._verifier_path,
            )
            self._verifier = None
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
        self._last_fire_sample = None
        self._last_fire_time = None
        if self._scorer is not None:
            self._scorer.reset()

    def shutdown(self) -> None:
        self.stop()
        self._model = None

    def predict_window(
        self,
        window: np.ndarray,
        sample_index: int | None = None,
    ) -> Optional[Detection]:
        if not self._available or self._model is None:
            return None
        if not self._enabled or self._suspended:
            return None
        if window.size < WINDOW_SAMPLES:
            return None

        scores = self._scorer.score(window) if self._scorer else self._model.predict(window)
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
            computed, reused = self._scorer.stats if self._scorer else (0, 0)
            _LOGGER.info(
                "Wake score=%.4f max=%.4f key=%s thresh=%.3f emb_computed=%d emb_reused=%d",
                score,
                self._max_score_window,
                self._score_key,
                self._threshold,
                computed,
                reused,
            )
            self._last_score_log = now
            self._max_score_window = 0.0

        if self._miner is not None:
            self._miner.offer(score, window, sample_index=sample_index)

        if score < self._threshold:
            return None
        if sample_index is not None:
            refractory_samples = int(self._refractory * SAMPLE_RATE)
            if (
                self._last_fire_sample is not None
                and refractory_samples > 0
                and (sample_index - self._last_fire_sample) < refractory_samples
            ):
                return None
        elif (
            self._refractory > 0
            and self._last_fire_time is not None
            and (now - self._last_fire_time) < self._refractory
        ):
            return None

        if self._verifier is not None:
            embeddings = self._scorer.last_embeddings if self._scorer is not None else None
            verifier_score = self._verifier.score(
                window,
                self._model,
                embeddings=embeddings,
            )
            if verifier_score is None or verifier_score < self._verifier.threshold:
                _LOGGER.info(
                    "Wake verifier veto phrase=%r livekit=%.3f verifier=%s thresh=%.3f",
                    self._phrase,
                    score,
                    f"{verifier_score:.3f}" if verifier_score is not None else "n/a",
                    self._verifier.threshold,
                )
                return None

        # Stamp only on a real fire: a vetoed hop must not lock out the next one.
        if sample_index is not None:
            self._last_fire_sample = sample_index
        else:
            self._last_fire_time = now

        _LOGGER.info("Wake phrase detected phrase=%r confidence=%.3f", self._phrase, score)
        return Detection(
            phrase=self._phrase,
            confidence=score,
            timestamp=now,
            sample_index=sample_index,
        )

    def process_pcm(self, pcm_s16le: bytes, sample_rate: int = 16000) -> Optional[Detection]:
        if sample_rate != SAMPLE_RATE or not pcm_s16le:
            return None
        buffer = WakeAudioBuffer(WINDOW_SAMPLES, HOP_SAMPLES)
        if not buffer.feed(pcm_s16le):
            return None
        return self.predict_window(buffer.window())
