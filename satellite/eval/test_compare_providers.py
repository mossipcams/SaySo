"""Tests for satellite/eval/compare_providers.py."""

from __future__ import annotations

import importlib.util
import sys
import wave
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_COMPARE = Path(__file__).resolve().parent / "compare_providers.py"
_spec = importlib.util.spec_from_file_location("satellite_eval_compare_providers", _COMPARE)
compare = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = compare
_spec.loader.exec_module(compare)

from satellite.sayso.wake.livekit import HOP_SAMPLES, WINDOW_SAMPLES  # noqa: E402


def _write_wav(path: Path, samples: np.ndarray, rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.astype("<i2").tobytes())


def test_scan_max_score_tracks_peak() -> None:
    total = WINDOW_SAMPLES + HOP_SAMPLES + compare.CHUNK_SAMPLES
    pcm = np.zeros(total, dtype="<i2").tobytes()
    seen = {"count": 0}

    def score_fn(_window: np.ndarray) -> float:
        seen["count"] += 1
        return float(seen["count"]) * 0.1

    max_score = compare.scan_max_score(pcm, 16000, score_fn)
    assert seen["count"] >= 1
    assert max_score == pytest.approx(seen["count"] * 0.1)


def test_main_empty_audio_dir(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    audio_dir = tmp_path / "clips"
    audio_dir.mkdir()
    rc = compare.main(["--audio-dir", str(audio_dir), "--livekit", "x.onnx", "--nano", "y.onnx"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "clips=0" in out
    assert "no threshold" in out
    assert "lk_max" in out
    assert "nano_max" in out
    assert "lk_det" not in out


def test_main_missing_audio_dir(tmp_path: Path) -> None:
    rc = compare.main(
        [
            "--audio-dir",
            str(tmp_path / "missing"),
            "--livekit",
            str(tmp_path / "livekit.onnx"),
            "--nano",
            str(tmp_path / "nano.onnx"),
        ]
    )
    assert rc == 2


def test_main_missing_model_with_clips(tmp_path: Path) -> None:
    audio_dir = tmp_path / "clips"
    audio_dir.mkdir()
    _write_wav(audio_dir / "one.wav", np.zeros(WINDOW_SAMPLES, dtype="<i2"))
    rc = compare.main(
        [
            "--audio-dir",
            str(audio_dir),
            "--livekit",
            str(tmp_path / "missing-livekit.onnx"),
            "--nano",
            str(tmp_path / "missing-nano.onnx"),
        ]
    )
    assert rc == 2


def test_nano_window_score_uses_hop_feed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import sys
    import types
    from types import SimpleNamespace
    from unittest.mock import MagicMock as MM

    model_path = tmp_path / "nano.onnx"
    model_path.write_bytes(b"fake")
    mock_interp = MM()
    mock_interp.predict.return_value = SimpleNamespace(score=0.2)
    for key in list(sys.modules):
        if key == "nanowakeword" or key.startswith("nanowakeword."):
            monkeypatch.delitem(sys.modules, key, raising=False)
    fake_module = types.ModuleType("nanowakeword")
    fake_module.NanoInterpreter = MM(load_model=MM(return_value=mock_interp))
    monkeypatch.setitem(sys.modules, "nanowakeword", fake_module)

    from satellite.sayso.wake.nanowakeword import NanoWakeWordProvider

    provider = NanoWakeWordProvider(model_path=model_path, phrase="SaySo", threshold=0.99)
    window = np.zeros(WINDOW_SAMPLES, dtype=np.int16)

    compare.nano_window_score(provider, window)
    first_sizes = [int(call.args[0].size) for call in mock_interp.predict.call_args_list]
    # Load/reset warmup prepends silence hops; the clip itself is the tail.
    while first_sizes and sum(first_sizes) > WINDOW_SAMPLES:
        first_sizes.pop(0)
    assert first_sizes
    assert sum(first_sizes) == WINDOW_SAMPLES
    assert first_sizes[:-1] == [HOP_SAMPLES] * (len(first_sizes) - 1)
    assert first_sizes[-1] >= 1280

    mock_interp.predict.reset_mock()
    compare.nano_window_score(provider, window)
    assert mock_interp.predict.call_count == 1
    assert mock_interp.predict.call_args.args[0].size == HOP_SAMPLES


def test_score_clip_resets_nano_per_clip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wav_path = tmp_path / "clip.wav"
    _write_wav(wav_path, np.zeros(WINDOW_SAMPLES, dtype="<i2"))

    resets: list[str] = []

    class FakeNano:
        available = True
        _interpreter = object()
        _stream_primed = True

        def reset(self) -> None:
            resets.append("reset")
            self._stream_primed = False

        def feed_window_score(self, _window: np.ndarray) -> float:
            return 0.1

    fake_nano = FakeNano()
    monkeypatch.setattr(compare, "livekit_window_score", lambda _p, _w: 0.0)
    monkeypatch.setattr(compare, "nano_window_score", lambda p, w: p.feed_window_score(w))

    compare.score_clip(
        wav_path,
        "clip.wav",
        MagicMock(available=True),
        fake_nano,
    )
    assert resets == ["reset"]
    assert not fake_nano._stream_primed


def test_main_scores_clips_with_mock_providers(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    audio_dir = tmp_path / "clips"
    nested = audio_dir / "nested"
    nested.mkdir(parents=True)
    _write_wav(nested / "hit.wav", np.zeros(WINDOW_SAMPLES + HOP_SAMPLES, dtype="<i2"))

    livekit_path = tmp_path / "livekit.onnx"
    nano_path = tmp_path / "nano.onnx"
    livekit_path.write_bytes(b"fake")
    nano_path.write_bytes(b"fake")

    class FakeLiveKit:
        available = True
        _model = object()
        _scorer = None
        _score_key = "livekit"

        def predict(self, _window: np.ndarray) -> dict[str, float]:
            return {"livekit": 0.72}

    class FakeNano:
        available = True
        _interpreter = object()

        def reset(self) -> None:
            pass

        def predict(self, _window: np.ndarray) -> MagicMock:
            return MagicMock(score=0.41)

    fake_livekit = FakeLiveKit()
    fake_nano = FakeNano()

    monkeypatch.setattr(
        compare,
        "LiveKitWakeWordProvider",
        lambda *_args, **_kwargs: fake_livekit,
    )
    monkeypatch.setattr(
        compare,
        "NanoWakeWordProvider",
        lambda *_args, **_kwargs: fake_nano,
    )
    monkeypatch.setattr(compare, "livekit_window_score", lambda _p, _w: 0.72)
    monkeypatch.setattr(compare, "nano_window_score", lambda _p, _w: 0.41)

    rc = compare.main(
        [
            "--audio-dir",
            str(audio_dir),
            "--livekit",
            str(livekit_path),
            "--nano",
            str(nano_path),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "nested/hit.wav" in out
    assert "0.7200" in out
    assert "0.4100" in out
    assert "lk_det" not in out
    assert " yes " not in out
