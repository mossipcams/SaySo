"""Optional second-stage wake verifier for LiveKit detection.

The generate-first SaySo model classifies from (16, 96) speech embeddings
built by the same mel frontend + embedding ONNX stack as LiveKit. A compatible
second stage must score those embeddings — not a parallel mel pass.

Legacy ``sayso-verifier.npz`` artifacts (unversioned or ``feature_kind=mel_union``)
are a collapsed-mel mean+std logistic fit on 50 Snowball clips vs 19 live FPs.
That is a this-room recording-envelope check, not a phrase verifier: it blessed
recorded SaySo while the new classifier scored ~0.00, and vetoed live
generate-first fires (LiveKit 0.53–0.84, verifier 0.01–0.10). Those artifacts
are treated as incompatible with the generate-first model and do not AND-gate.

Compatible npz files set ``feature_kind=speech_embedding`` and logistic weights
for concat(mean, std) over the last 16 speech-embedding vectors (192-d).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .streaming import EMBEDDING_STRIDE, EMBEDDING_WINDOW, MIN_EMBEDDINGS

_LOGGER = logging.getLogger(__name__)

FEATURE_KIND_MEL_UNION = "mel_union"
FEATURE_KIND_SPEECH_EMBEDDING = "speech_embedding"

EMBEDDING_DIM = 96
SPEECH_EMBEDDING_FEATURE_DIM = EMBEDDING_DIM * 2  # mean + std


def _sigmoid(x: float) -> float:
    if x >= 0:
        return float(1.0 / (1.0 + np.exp(-x)))
    exp_x = np.exp(x)
    return float(exp_x / (1.0 + exp_x))


def _npz_feature_kind(data: np.lib.npyio.NpzFile) -> str:
    if "feature_kind" not in data:
        return FEATURE_KIND_MEL_UNION
    raw = data["feature_kind"]
    if isinstance(raw, np.ndarray):
        if raw.shape == ():
            return str(raw.item())
        return str(raw.reshape(-1)[0])
    return str(raw)


def mel_union_features(model: Any, window: np.ndarray) -> Optional[np.ndarray]:
    """Extract 64-d mel mean+std features from the last-16 embedding mel union.

    Legacy living2 verifier fit only. Not a phrase check for the generate-first
    model — see module docstring.
    """
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


def speech_embedding_features(embeddings: np.ndarray) -> Optional[np.ndarray]:
    """Collapse (16, 96) speech embeddings to 192-d mean+std features."""
    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        return None
    if embeddings.shape[0] < MIN_EMBEDDINGS or embeddings.shape[1] != EMBEDDING_DIM:
        return None
    emb = embeddings[-MIN_EMBEDDINGS:]
    mu = emb.mean(axis=0)
    sigma = emb.std(axis=0)
    return np.concatenate([mu, sigma]).astype(np.float32)


class WakeVerifier:
    """Logistic verifier loaded from a versioned npz artifact."""

    def __init__(
        self,
        path: Path,
        threshold: Optional[float] = None,
    ) -> None:
        self._path = Path(path)
        data = np.load(self._path)
        self.feature_kind = _npz_feature_kind(data)
        self._mean = np.asarray(data["mean"], dtype=np.float32)
        self._scale = np.asarray(data["scale"], dtype=np.float32)
        self._coef = np.asarray(data["coef"], dtype=np.float32)
        self._intercept = float(np.asarray(data["intercept"]).reshape(()))
        default_t = float(np.asarray(data["threshold"]).reshape(()))
        self._threshold = float(threshold) if threshold is not None else default_t

    @property
    def compatible(self) -> bool:
        """True when this artifact can second-gate the generate-first model."""
        return self.feature_kind == FEATURE_KIND_SPEECH_EMBEDDING

    @property
    def threshold(self) -> float:
        return self._threshold

    def score(
        self,
        window: np.ndarray,
        model: Any,
        embeddings: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """Return verifier probability in [0, 1], or None when features are unavailable."""
        if self.feature_kind == FEATURE_KIND_SPEECH_EMBEDDING:
            features = speech_embedding_features(embeddings) if embeddings is not None else None
            if features is None:
                _LOGGER.debug(
                    "Speech-embedding verifier could not use embeddings (shape=%s)",
                    getattr(embeddings, "shape", None),
                )
                return None
        else:
            features = mel_union_features(model, window)
            if features is None:
                _LOGGER.debug("Mel verifier could not extract features from window")
                return None

        if features.shape != self._mean.shape:
            _LOGGER.warning(
                "Wake verifier feature shape %s != model %s (feature_kind=%s)",
                features.shape,
                self._mean.shape,
                self.feature_kind,
            )
            return None
        x = (features - self._mean) / self._scale
        logit = float(np.dot(self._coef, x) + self._intercept)
        return _sigmoid(logit)

    def passes(
        self,
        window: np.ndarray,
        model: Any,
        embeddings: Optional[np.ndarray] = None,
    ) -> bool:
        score = self.score(window, model, embeddings=embeddings)
        if score is None:
            return False
        return score >= self._threshold


def load_wake_verifier(
    path: Path,
    threshold: Optional[float] = None,
) -> Optional[WakeVerifier]:
    """Load a verifier npz, or None when the artifact is legacy/incompatible."""
    verifier = WakeVerifier(path, threshold=threshold)
    if verifier.compatible:
        return verifier
    _LOGGER.warning(
        "Wake verifier %s is legacy %s (not a phrase check for the generate-first "
        "model); running single-stage LiveKit only",
        path,
        verifier.feature_kind,
    )
    return None


# Backward-compatible alias for tests and callers that still refer to MelVerifier.
MelVerifier = WakeVerifier
