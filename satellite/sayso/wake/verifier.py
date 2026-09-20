"""Second-stage mel verifier for LiveKit wake detection.

Scores the union of mel frames covered by the last 16 speech-embedding
windows, then applies a logistic regression trained on the host:

  int16 2s window -> mel frontend -> last-16-window mel union ->
  concat(mean, std) -> (x - mean) / scale -> sigmoid(coef·x + intercept)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .streaming import EMBEDDING_STRIDE, EMBEDDING_WINDOW, MIN_EMBEDDINGS

_LOGGER = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return float(1.0 / (1.0 + np.exp(-x)))
    exp_x = np.exp(x)
    return float(exp_x / (1.0 + exp_x))


def mel_union_features(model: Any, window: np.ndarray) -> Optional[np.ndarray]:
    """Extract 64-d mel mean+std features matching the host verifier fit."""
    mel_frontend = getattr(model, "_mel_frontend", None)
    if mel_frontend is None:
        return None

    raw = window
    if window.dtype == np.int16:
        audio = window.astype(np.float32) / 32768.0
    else:
        audio = window.astype(np.float32, copy=False)
    audio = audio.flatten()

    mel = mel_frontend(audio)
    if mel.ndim == 3:
        mel = mel[0]
    if not isinstance(mel, np.ndarray) or mel.ndim != 2:
        return None
    if mel.shape[0] < EMBEDDING_WINDOW:
        return None

    starts = list(range(0, mel.shape[0] - EMBEDDING_WINDOW + 1, EMBEDDING_STRIDE))
    if len(starts) < MIN_EMBEDDINGS:
        return None
    starts = starts[-MIN_EMBEDDINGS:]

    lo = starts[0]
    hi = starts[-1] + EMBEDDING_WINDOW
    mel_union = mel[lo:hi]
    mu = mel_union.mean(axis=0)
    sigma = mel_union.std(axis=0)
    return np.concatenate([mu, sigma]).astype(np.float32)


class MelVerifier:
    """Logistic mel verifier loaded from an npz artifact."""

    def __init__(
        self,
        path: Path,
        threshold: Optional[float] = None,
    ) -> None:
        self._path = Path(path)
        data = np.load(self._path)
        self._mean = np.asarray(data["mean"], dtype=np.float32)
        self._scale = np.asarray(data["scale"], dtype=np.float32)
        self._coef = np.asarray(data["coef"], dtype=np.float32)
        self._intercept = float(np.asarray(data["intercept"]).reshape(()))
        default_t = float(np.asarray(data["threshold"]).reshape(()))
        self._threshold = float(threshold) if threshold is not None else default_t

    @property
    def threshold(self) -> float:
        return self._threshold

    def score(self, window: np.ndarray, model: Any) -> Optional[float]:
        """Return verifier probability in [0, 1], or None when features are unavailable."""
        features = mel_union_features(model, window)
        if features is None:
            _LOGGER.debug("Mel verifier could not extract features from window")
            return None
        if features.shape != self._mean.shape:
            _LOGGER.warning(
                "Mel verifier feature shape %s != model %s",
                features.shape,
                self._mean.shape,
            )
            return None
        x = (features - self._mean) / self._scale
        logit = float(np.dot(self._coef, x) + self._intercept)
        return _sigmoid(logit)

    def passes(self, window: np.ndarray, model: Any) -> bool:
        score = self.score(window, model)
        if score is None:
            return False
        return score >= self._threshold
