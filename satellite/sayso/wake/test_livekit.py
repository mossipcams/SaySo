"""LiveKitWakeWordProvider: hop accumulation and worker-friendly predict."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_SATELLITE_ROOT = Path(__file__).resolve().parents[2]
if str(_SATELLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SATELLITE_ROOT))

from sayso.wake.livekit import HOP_SAMPLES, WINDOW_SAMPLES, LiveKitWakeWordProvider  # noqa: E402
from sayso.wake.streaming import EMBEDDING_STRIDE, EMBEDDING_WINDOW, MIN_EMBEDDINGS  # noqa: E402
from sayso.wake.verifier import FEATURE_KIND_SPEECH_EMBEDDING, SPEECH_EMBEDDING_FEATURE_DIM  # noqa: E402

CHUNK_SAMPLES = 512


def _silence_pcm(n_samples: int) -> bytes:
    return np.zeros(n_samples, dtype="<i2").tobytes()


@pytest.fixture
def mock_wake_model(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    # spec limits MagicMock auto-attrs so CachedEmbeddingScorer stays on predict().
    mock_model = MagicMock(spec=["predict"])
    mock_model.predict.return_value = {"hey_ferra": 0.0}
    mock_model_cls = MagicMock(return_value=mock_model)
    fake_wakeword = MagicMock(WakeWordModel=mock_model_cls)
    monkeypatch.setitem(sys.modules, "livekit", MagicMock(wakeword=fake_wakeword))
    monkeypatch.setitem(sys.modules, "livekit.wakeword", fake_wakeword)
    return mock_model


def test_predict_window_uses_numpy_buffer_without_tolist(
    mock_wake_model: MagicMock,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "hey_ferra.onnx"
    model_path.write_bytes(b"fake-onnx")

    provider = LiveKitWakeWordProvider(
        model_path=model_path,
        phrase="hey ferra",
        threshold=0.99,
    )
    assert provider.available
    provider.start()

    window = np.zeros(WINDOW_SAMPLES, dtype=np.int16)
    provider.predict_window(window)

    mock_wake_model.predict.assert_called_once()
    passed = mock_wake_model.predict.call_args.args[0]
    assert isinstance(passed, np.ndarray)
    assert passed.dtype == np.int16


def test_process_pcm_runs_predict_after_window_and_hop(
    mock_wake_model: MagicMock,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "hey_ferra.onnx"
    model_path.write_bytes(b"fake-onnx")

    provider = LiveKitWakeWordProvider(
        model_path=model_path,
        phrase="hey ferra",
        threshold=0.99,
    )
    provider.start()

    chunk = _silence_pcm(WINDOW_SAMPLES + HOP_SAMPLES)
    provider.process_pcm(chunk)

    mock_wake_model.predict.assert_called()


def _attach_cached_scorer_internals(
    mock_model: MagicMock,
    *,
    livekit_score: float = 0.9,
) -> None:
    n_frames = EMBEDDING_WINDOW + (MIN_EMBEDDINGS - 1) * EMBEDDING_STRIDE

    def _mel_frontend(audio: np.ndarray) -> np.ndarray:
        return np.arange(n_frames * 32, dtype=np.float32).reshape(n_frames, 32)

    def _speech_embedding(window: np.ndarray) -> np.ndarray:
        return np.full((window.shape[0], 96), 0.5, dtype=np.float32)

    session = MagicMock()
    session.run.return_value = [np.array([[livekit_score]], dtype=np.float32)]

    mock_model._mel_frontend = _mel_frontend
    mock_model._speech_embedding = _speech_embedding
    mock_model._classifiers = {"sayso": (session, "embeddings")}


def _make_legacy_verifier_npz(
    tmp_path: Path,
    *,
    intercept: float = 10.0,
    threshold: float = 0.5,
) -> Path:
    path = tmp_path / "legacy-verifier.npz"
    np.savez(
        path,
        mean=np.zeros(64, dtype=np.float32),
        scale=np.ones(64, dtype=np.float32),
        coef=np.zeros(64, dtype=np.float32),
        intercept=np.float32(intercept),
        threshold=np.float32(threshold),
    )
    return path


def _make_embedding_verifier_npz(
    tmp_path: Path,
    *,
    intercept: float = 10.0,
    threshold: float = 0.5,
) -> Path:
    path = tmp_path / "embedding-verifier.npz"
    np.savez(
        path,
        feature_kind=np.array(FEATURE_KIND_SPEECH_EMBEDDING),
        mean=np.zeros(SPEECH_EMBEDDING_FEATURE_DIM, dtype=np.float32),
        scale=np.ones(SPEECH_EMBEDDING_FEATURE_DIM, dtype=np.float32),
        coef=np.zeros(SPEECH_EMBEDDING_FEATURE_DIM, dtype=np.float32),
        intercept=np.float32(intercept),
        threshold=np.float32(threshold),
    )
    return path


def test_fires_when_livekit_and_embedding_verifier_pass(
    mock_wake_model: MagicMock,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "sayso.onnx"
    model_path.write_bytes(b"fake-onnx")
    verifier_path = _make_embedding_verifier_npz(tmp_path, intercept=10.0)

    _attach_cached_scorer_internals(mock_wake_model, livekit_score=0.9)

    provider = LiveKitWakeWordProvider(
        model_path=model_path,
        phrase="SaySo",
        threshold=0.5,
        refractory_seconds=0.0,
        verifier_path=verifier_path,
    )
    assert provider.available
    provider.start()

    detection = provider.predict_window(np.zeros(WINDOW_SAMPLES, dtype=np.int16))
    assert detection is not None
    assert detection.phrase == "SaySo"
    assert detection.confidence == pytest.approx(0.9)


def test_embedding_verifier_vetoes_high_livekit_score(
    mock_wake_model: MagicMock,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "sayso.onnx"
    model_path.write_bytes(b"fake-onnx")
    verifier_path = _make_embedding_verifier_npz(tmp_path, intercept=-10.0)

    _attach_cached_scorer_internals(mock_wake_model, livekit_score=0.99)

    provider = LiveKitWakeWordProvider(
        model_path=model_path,
        phrase="SaySo",
        threshold=0.5,
        refractory_seconds=0.0,
        verifier_path=verifier_path,
    )
    provider.start()

    assert provider.predict_window(np.zeros(WINDOW_SAMPLES, dtype=np.int16)) is None


def test_legacy_mel_verifier_does_not_veto_livekit_fire(
    mock_wake_model: MagicMock,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "sayso.onnx"
    model_path.write_bytes(b"fake-onnx")
    verifier_path = _make_legacy_verifier_npz(tmp_path, intercept=-10.0)

    mock_wake_model.predict.return_value = {"sayso": 0.9}

    provider = LiveKitWakeWordProvider(
        model_path=model_path,
        phrase="SaySo",
        threshold=0.5,
        refractory_seconds=0.0,
        verifier_path=verifier_path,
    )
    assert provider.available
    provider.start()

    detection = provider.predict_window(np.zeros(WINDOW_SAMPLES, dtype=np.int16))
    assert detection is not None
    assert detection.confidence == pytest.approx(0.9)


def test_unconfigured_verifier_keeps_single_stage_fire(
    mock_wake_model: MagicMock,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "sayso.onnx"
    model_path.write_bytes(b"fake-onnx")
    mock_wake_model.predict.return_value = {"sayso": 0.9}

    provider = LiveKitWakeWordProvider(
        model_path=model_path,
        phrase="SaySo",
        threshold=0.5,
        refractory_seconds=0.0,
    )
    provider.start()

    assert provider.predict_window(np.zeros(WINDOW_SAMPLES, dtype=np.int16)) is not None


def test_missing_verifier_file_fails_closed(
    mock_wake_model: MagicMock,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "sayso.onnx"
    model_path.write_bytes(b"fake-onnx")

    provider = LiveKitWakeWordProvider(
        model_path=model_path,
        phrase="SaySo",
        threshold=0.5,
        verifier_path=tmp_path / "missing.npz",
    )
    assert not provider.available


def test_miner_sees_livekit_score_before_embedding_verifier_veto(
    mock_wake_model: MagicMock,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "sayso.onnx"
    model_path.write_bytes(b"fake-onnx")
    verifier_path = _make_embedding_verifier_npz(tmp_path, intercept=-10.0)
    _attach_cached_scorer_internals(mock_wake_model, livekit_score=0.85)

    miner = MagicMock()
    provider = LiveKitWakeWordProvider(
        model_path=model_path,
        phrase="SaySo",
        threshold=0.5,
        refractory_seconds=0.0,
        miner=miner,
        verifier_path=verifier_path,
    )
    provider.start()

    assert provider.predict_window(np.zeros(WINDOW_SAMPLES, dtype=np.int16)) is None
    miner.offer.assert_called_once()
    assert miner.offer.call_args.args[0] == pytest.approx(0.85)


def test_predict_window_wall_clock_refractory_without_sample_index(
    mock_wake_model: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "hey_ferra.onnx"
    model_path.write_bytes(b"fake-onnx")
    mock_wake_model.predict.return_value = {"hey_ferra": 0.99}

    clock = {"t": 100.0}

    def fake_monotonic() -> float:
        return clock["t"]

    monkeypatch.setattr(time, "monotonic", fake_monotonic)

    provider = LiveKitWakeWordProvider(
        model_path=model_path,
        phrase="hey ferra",
        threshold=0.5,
        refractory_seconds=2.0,
    )
    provider.start()

    window = np.zeros(WINDOW_SAMPLES, dtype=np.int16)
    first = provider.predict_window(window)
    clock["t"] += 1.0
    second = provider.predict_window(window)

    assert first is not None
    assert second is None
