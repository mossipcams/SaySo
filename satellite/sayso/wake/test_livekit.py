"""LiveKitWakeWordProvider: hop accumulation and worker-friendly predict."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_SATELLITE_ROOT = Path(__file__).resolve().parents[2]
if str(_SATELLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SATELLITE_ROOT))

from sayso.wake.livekit import HOP_SAMPLES, WINDOW_SAMPLES, LiveKitWakeWordProvider  # noqa: E402

CHUNK_SAMPLES = 512


def _silence_pcm(n_samples: int) -> bytes:
    return np.zeros(n_samples, dtype="<i2").tobytes()


@pytest.fixture
def mock_wake_model(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_model = MagicMock()
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
