"""NanoWakeWordProvider: threshold fire/no-fire, reset, fail-closed."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

_SATELLITE_ROOT = Path(__file__).resolve().parents[2]
if str(_SATELLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SATELLITE_ROOT))

from sayso.wake.livekit import HOP_SAMPLES, WINDOW_SAMPLES  # noqa: E402
from sayso.wake.nanowakeword import NanoWakeWordProvider  # noqa: E402


def _silence_pcm(n_samples: int) -> bytes:
    return np.zeros(n_samples, dtype="<i2").tobytes()


def _install_fake_nanowakeword(
    monkeypatch: pytest.MonkeyPatch,
    mock_interp: MagicMock,
) -> None:
    for key in list(sys.modules):
        if key == "nanowakeword" or key.startswith("nanowakeword."):
            monkeypatch.delitem(sys.modules, key, raising=False)
    mock_cls = MagicMock()
    mock_cls.load_model.return_value = mock_interp
    fake_module = types.ModuleType("nanowakeword")
    fake_module.NanoInterpreter = mock_cls
    monkeypatch.setitem(sys.modules, "nanowakeword", fake_module)


@pytest.fixture
def mock_interpreter(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_interp = MagicMock()
    mock_interp.predict.return_value = SimpleNamespace(score=0.0)
    _install_fake_nanowakeword(monkeypatch, mock_interp)
    return mock_interp


def test_missing_model_is_fail_closed(tmp_path: Path) -> None:
    provider = NanoWakeWordProvider(
        model_path=tmp_path / "missing.onnx",
        phrase="SaySo",
        threshold=0.5,
    )
    assert not provider.available
    assert provider.predict_window(np.zeros(WINDOW_SAMPLES, dtype=np.int16)) is None


def test_missing_import_is_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    model_path = tmp_path / "sayso.onnx"
    model_path.write_bytes(b"fake-onnx")
    monkeypatch.delitem(sys.modules, "nanowakeword", raising=False)
    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "nanowakeword" or name.startswith("nanowakeword."):
            raise ImportError("nanowakeword not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    provider = NanoWakeWordProvider(model_path=model_path, phrase="SaySo", threshold=0.5)
    assert not provider.available


def test_predict_window_fires_above_threshold(
    mock_interpreter: MagicMock,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "sayso.onnx"
    model_path.write_bytes(b"fake-onnx")
    mock_interpreter.predict.return_value = SimpleNamespace(score=0.9)

    provider = NanoWakeWordProvider(model_path=model_path, phrase="SaySo", threshold=0.5)
    assert provider.available
    provider.start()

    window = np.zeros(WINDOW_SAMPLES, dtype=np.int16)
    detection = provider.predict_window(window, sample_index=42)

    assert detection is not None
    assert detection.phrase == "SaySo"
    assert detection.confidence == 0.9
    assert detection.sample_index == 42
    mock_interpreter.reset.assert_not_called()


def test_predict_window_no_fire_below_threshold(
    mock_interpreter: MagicMock,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "sayso.onnx"
    model_path.write_bytes(b"fake-onnx")
    mock_interpreter.predict.return_value = SimpleNamespace(score=0.1)

    provider = NanoWakeWordProvider(model_path=model_path, phrase="SaySo", threshold=0.5)
    provider.start()

    detection = provider.predict_window(np.zeros(WINDOW_SAMPLES, dtype=np.int16))
    assert detection is None
    mock_interpreter.reset.assert_not_called()


def test_reset_clears_interpreter_state(
    mock_interpreter: MagicMock,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "sayso.onnx"
    model_path.write_bytes(b"fake-onnx")

    provider = NanoWakeWordProvider(model_path=model_path, phrase="SaySo", threshold=0.5)
    mock_interpreter.predict.reset_mock()
    provider.reset()

    mock_interpreter.reset.assert_called_once()
    assert mock_interpreter.predict.call_count == 5
    assert all(call.args[0].size == HOP_SAMPLES for call in mock_interpreter.predict.call_args_list)


def test_predict_window_feeds_hop_chunks_then_tail_only(
    mock_interpreter: MagicMock,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "sayso.onnx"
    model_path.write_bytes(b"fake-onnx")
    mock_interpreter.predict.return_value = SimpleNamespace(score=0.0)

    provider = NanoWakeWordProvider(model_path=model_path, phrase="SaySo", threshold=0.99)
    provider.start()
    mock_interpreter.predict.reset_mock()

    window = np.arange(WINDOW_SAMPLES, dtype=np.int16)
    provider.predict_window(window)
    first_calls = mock_interpreter.predict.call_args_list
    sizes = [int(call.args[0].size) for call in first_calls]
    assert sizes
    assert all(size >= 1280 for size in sizes)
    assert sum(sizes) == WINDOW_SAMPLES
    assert sizes[:-1] == [HOP_SAMPLES] * (len(sizes) - 1)

    mock_interpreter.predict.reset_mock()
    provider.predict_window(window)
    mock_interpreter.predict.assert_called_once()
    assert mock_interpreter.predict.call_args.args[0].size == HOP_SAMPLES


def test_first_predict_window_fire_stays_primed_tail_hop_only(
    mock_interpreter: MagicMock,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "sayso.onnx"
    model_path.write_bytes(b"fake-onnx")
    mock_interpreter.predict.return_value = SimpleNamespace(score=0.9)

    provider = NanoWakeWordProvider(model_path=model_path, phrase="SaySo", threshold=0.5)
    provider.start()
    mock_interpreter.predict.reset_mock()

    window = np.arange(WINDOW_SAMPLES, dtype=np.int16)
    detection = provider.predict_window(window)
    assert detection is not None
    assert provider._stream_primed
    mock_interpreter.reset.assert_not_called()

    mock_interpreter.predict.reset_mock()
    mock_interpreter.predict.return_value = SimpleNamespace(score=0.0)
    provider.predict_window(window)
    mock_interpreter.predict.assert_called_once()
    assert mock_interpreter.predict.call_args.args[0].size == HOP_SAMPLES


def test_predict_window_fire_keeps_stream_primed_tail_hop_only(
    mock_interpreter: MagicMock,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "sayso.onnx"
    model_path.write_bytes(b"fake-onnx")
    mock_interpreter.predict.return_value = SimpleNamespace(score=0.0)

    provider = NanoWakeWordProvider(model_path=model_path, phrase="SaySo", threshold=0.5)
    provider.start()
    mock_interpreter.predict.reset_mock()

    window = np.arange(WINDOW_SAMPLES, dtype=np.int16)
    provider.predict_window(window)
    assert provider._stream_primed

    mock_interpreter.predict.reset_mock()
    mock_interpreter.predict.return_value = SimpleNamespace(score=0.9)
    detection = provider.predict_window(window)
    assert detection is not None
    assert provider._stream_primed
    mock_interpreter.reset.assert_not_called()

    mock_interpreter.predict.reset_mock()
    mock_interpreter.predict.return_value = SimpleNamespace(score=0.0)
    provider.predict_window(window)
    mock_interpreter.predict.assert_called_once()
    assert mock_interpreter.predict.call_args.args[0].size == HOP_SAMPLES


def test_reset_clears_stream_primed_flag(
    mock_interpreter: MagicMock,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "sayso.onnx"
    model_path.write_bytes(b"fake-onnx")
    mock_interpreter.predict.return_value = SimpleNamespace(score=0.0)

    provider = NanoWakeWordProvider(model_path=model_path, phrase="SaySo", threshold=0.99)
    provider.start()
    provider.predict_window(np.zeros(WINDOW_SAMPLES, dtype=np.int16))
    assert provider._stream_primed

    provider.reset()
    assert not provider._stream_primed
    assert mock_interpreter.predict.call_count >= 5


def test_process_pcm_does_not_open_microphone(
    mock_interpreter: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "sayso.onnx"
    model_path.write_bytes(b"fake-onnx")
    mock_interpreter.predict.return_value = SimpleNamespace(score=0.0)

    mic_mock = MagicMock()
    monkeypatch.setitem(sys.modules, "pyaudio", mic_mock)
    monkeypatch.setitem(sys.modules, "sounddevice", mic_mock)

    provider = NanoWakeWordProvider(model_path=model_path, phrase="SaySo", threshold=0.99)
    provider.start()
    provider.process_pcm(_silence_pcm(WINDOW_SAMPLES + HOP_SAMPLES))

    mic_mock.assert_not_called()
    mock_interpreter.predict.assert_called()
