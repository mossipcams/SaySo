"""LiveKitWakeWordProvider: hop accumulation must survive resume()."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# ponytail: pytest file-path mode puts wake/ on sys.path; satellite/ is the sayso root
_SATELLITE_ROOT = Path(__file__).resolve().parents[2]
if str(_SATELLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SATELLITE_ROOT))

from sayso.wake.livekit import HOP_SAMPLES, WINDOW_SAMPLES, LiveKitWakeWordProvider  # noqa: E402

CHUNK_SAMPLES = 512  # deliberately below HOP_SAMPLES (2560)


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


def test_predict_runs_across_resume_with_sub_hop_chunks(
    mock_wake_model: MagicMock,
    tmp_path: Path,
) -> None:
    """Idle overlay resume() between mic chunks must not zero the hop counter."""
    model_path = tmp_path / "hey_ferra.onnx"
    model_path.write_bytes(b"fake-onnx")

    provider = LiveKitWakeWordProvider(
        model_path=model_path,
        phrase="hey ferra",
        threshold=0.99,
    )
    assert provider.available
    provider.start()

    chunk = _silence_pcm(CHUNK_SAMPLES)
    samples_fed = 0
    while samples_fed < WINDOW_SAMPLES:
        provider.suspend()
        provider.resume()
        provider.process_pcm(chunk)
        samples_fed += CHUNK_SAMPLES

    mock_wake_model.predict.assert_called()
    assert CHUNK_SAMPLES < HOP_SAMPLES
