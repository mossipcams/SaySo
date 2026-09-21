"""WakeVerifier feature extraction and scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SATELLITE_ROOT = Path(__file__).resolve().parents[2]
if str(_SATELLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SATELLITE_ROOT))

from sayso.wake.streaming import EMBEDDING_STRIDE, EMBEDDING_WINDOW, MIN_EMBEDDINGS  # noqa: E402
from sayso.wake.verifier import (  # noqa: E402
    FEATURE_KIND_MEL_UNION,
    FEATURE_KIND_SPEECH_EMBEDDING,
    SPEECH_EMBEDDING_FEATURE_DIM,
    WakeVerifier,
    load_wake_verifier,
    mel_union_features,
    speech_embedding_features,
)

_VERIFIER_NPZ = _SATELLITE_ROOT / "models" / "sayso-verifier.npz"


class _FakeMelModel:
    def __init__(self, n_mel_bins: int = 32, n_frames: int = 200) -> None:
        self._n_mel_bins = n_mel_bins
        self._n_frames = n_frames

    def _mel_frontend(self, audio: np.ndarray) -> np.ndarray:
        n_frames = max(self._n_frames, EMBEDDING_WINDOW + (MIN_EMBEDDINGS - 1) * EMBEDDING_STRIDE)
        base = np.arange(n_frames, dtype=np.float32)[:, None]
        return base @ np.ones((1, self._n_mel_bins), dtype=np.float32)


def test_mel_union_features_shape() -> None:
    model = _FakeMelModel()
    window = np.zeros(32000, dtype=np.int16)
    features = mel_union_features(model, window)
    assert features is not None
    assert features.shape == (64,)
    assert features.dtype == np.float32


def test_speech_embedding_features_shape() -> None:
    embeddings = np.ones((MIN_EMBEDDINGS, 96), dtype=np.float32)
    features = speech_embedding_features(embeddings)
    assert features is not None
    assert features.shape == (SPEECH_EMBEDDING_FEATURE_DIM,)
    assert features.dtype == np.float32
    assert np.allclose(features[:96], 1.0)
    assert np.allclose(features[96:], 0.0)


def test_legacy_verifier_npz_is_incompatible(tmp_path: Path) -> None:
    path = tmp_path / "legacy.npz"
    np.savez(
        path,
        mean=np.zeros(64, dtype=np.float32),
        scale=np.ones(64, dtype=np.float32),
        coef=np.zeros(64, dtype=np.float32),
        intercept=0.0,
        threshold=0.5,
    )
    verifier = WakeVerifier(path)
    assert verifier.feature_kind == FEATURE_KIND_MEL_UNION
    assert not verifier.compatible
    assert load_wake_verifier(path) is None


def test_mel_verifier_loads_npz_threshold() -> None:
    if not _VERIFIER_NPZ.is_file():
        pytest.skip("sayso-verifier.npz not in tree")
    verifier = WakeVerifier(_VERIFIER_NPZ)
    assert verifier.feature_kind == FEATURE_KIND_MEL_UNION
    assert not verifier.compatible
    assert 0.0 < verifier.threshold < 1.0
    assert verifier.threshold == pytest.approx(0.44523194, rel=1e-4)


def test_speech_embedding_verifier_score_and_passes(tmp_path: Path) -> None:
    mean = np.zeros(SPEECH_EMBEDDING_FEATURE_DIM, dtype=np.float32)
    scale = np.ones(SPEECH_EMBEDDING_FEATURE_DIM, dtype=np.float32)
    coef = np.zeros(SPEECH_EMBEDDING_FEATURE_DIM, dtype=np.float32)
    coef[0] = 1.0
    path = tmp_path / "verifier.npz"
    np.savez(
        path,
        feature_kind=np.array(FEATURE_KIND_SPEECH_EMBEDDING),
        mean=mean,
        scale=scale,
        coef=coef,
        intercept=0.0,
        threshold=0.5,
    )

    verifier = WakeVerifier(path)
    assert verifier.compatible
    embeddings = np.zeros((MIN_EMBEDDINGS, 96), dtype=np.float32)
    window = np.zeros(32000, dtype=np.int16)

    score = verifier.score(window, None, embeddings=embeddings)
    assert score is not None
    assert 0.0 <= score <= 1.0
    assert verifier.passes(window, None, embeddings=embeddings) == (score >= 0.5)
