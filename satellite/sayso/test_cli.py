from __future__ import annotations

import importlib
import inspect
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest

from satellite.sayso import cli


def _config(tmp_path=None):
    wake = tmp_path / "wake.wav" if tmp_path else "wake.wav"
    if tmp_path:
        wake.write_bytes(b"wav")
    return SimpleNamespace(
        audio=SimpleNamespace(
            input_device="pulse/configured-mic",
            output_device="pulse/configured-speaker",
            sample_rate=22050,
            channels=2,
        ),
        sounds=SimpleNamespace(wake=wake),
    )


def test_wait_for_audio_uses_config_without_changing_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "load_config", _config)
    run = Mock(
        side_effect=[
            SimpleNamespace(stdout="1\tconfigured-mic\n"),
            SimpleNamespace(stdout="2\tconfigured-speaker\n"),
        ]
    )
    call_process = Mock()
    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setattr(cli.subprocess, "call", call_process)

    assert cli.wait_for_audio(timeout=1) == 0
    call_process.assert_not_called()


def test_mic_check_uses_config_and_runtime_player(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config()
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    record = Mock(return_value=124)
    play = Mock(return_value=0)
    monkeypatch.setattr(cli.subprocess, "call", record)
    monkeypatch.setattr(cli, "_level_report", Mock())
    monkeypatch.setattr(cli, "_play_sound", play, raising=False)

    assert cli.cmd_test_mic(SimpleNamespace()) == 0
    command = record.call_args.args[0]
    assert command[command.index("--device") + 1] == "configured-mic"
    assert command[command.index("--rate") + 1] == "22050"
    assert command[command.index("--channels") + 1] == "2"
    play.assert_called_once_with(cli.CHECK_DIR / "mic-check.wav", "pulse/configured-speaker")


@pytest.mark.parametrize(("reason", "expected"), [(0, 0), (4, 1)])
def test_play_sound_uses_repaired_mpv_and_reports_errors(
    monkeypatch: pytest.MonkeyPatch,
    reason: int,
    expected: int,
) -> None:
    configured = Mock()
    monkeypatch.setattr("satellite.sayso.playback.configure_pulse_mpv", configured)
    monkeypatch.setattr("satellite.sayso.playback.install_playback_recovery", Mock())

    class FakeLibMpvPlayer:
        def _on_end_file(self, _event) -> None:
            self.done()

    class FakeMpvMediaPlayer:
        def __init__(self, device: str) -> None:
            assert device == "pulse/configured-speaker"
            self._player = FakeLibMpvPlayer()
            self._end_file = type(self._player)._on_end_file.__get__(self._player)

        def play(self, _path: str, done_callback) -> None:
            self._player.done = done_callback
            self._end_file(SimpleNamespace(data=SimpleNamespace(reason=reason)))

        def stop(self) -> None:
            pass

    libmpv = ModuleType("linux_voice_assistant.player.libmpv")
    libmpv.LibMpvPlayer = FakeLibMpvPlayer  # type: ignore[attr-defined]
    mpv_player = ModuleType("linux_voice_assistant.mpv_player")
    mpv_player.MpvMediaPlayer = FakeMpvMediaPlayer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "linux_voice_assistant.player.libmpv", libmpv)
    monkeypatch.setitem(sys.modules, "linux_voice_assistant.mpv_player", mpv_player)

    assert cli._play_sound("tone.wav", "pulse/configured-speaker") == expected
    configured.assert_called_once_with()


def test_play_sound_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("satellite.sayso.playback.configure_pulse_mpv", Mock())
    monkeypatch.setattr("satellite.sayso.playback.install_playback_recovery", Mock())

    class FakeLibMpvPlayer:
        def _on_end_file(self, _event) -> None:
            pass

    class FakeMpvMediaPlayer:
        def __init__(self, device: str) -> None:
            self._player = FakeLibMpvPlayer()
            self.stop = Mock()

        def play(self, _path: str, done_callback) -> None:
            pass

    libmpv = ModuleType("linux_voice_assistant.player.libmpv")
    libmpv.LibMpvPlayer = FakeLibMpvPlayer  # type: ignore[attr-defined]
    mpv_player = ModuleType("linux_voice_assistant.mpv_player")
    mpv_player.MpvMediaPlayer = FakeMpvMediaPlayer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "linux_voice_assistant.player.libmpv", libmpv)
    monkeypatch.setitem(sys.modules, "linux_voice_assistant.mpv_player", mpv_player)

    assert cli._play_sound("tone.wav", "speaker", timeout=0) == 1


def test_speaker_check_and_device_listing_use_runtime_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    cfg = _config(tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    play = Mock(return_value=0)
    monkeypatch.setattr(cli, "_play_sound", play, raising=False)

    assert cli.cmd_test_speaker(SimpleNamespace()) == 0
    play.assert_called_once_with(cfg.sounds.wake, cfg.audio.output_device)

    call_process = Mock(return_value=0)
    monkeypatch.setattr(cli.subprocess, "call", call_process)
    assert cli.cmd_devices(SimpleNamespace()) == 0
    assert call([sys.executable, "-m", "linux_voice_assistant", "--list-input-devices"]) in call_process.call_args_list
    assert call([sys.executable, "-m", "linux_voice_assistant", "--list-output-devices"]) in call_process.call_args_list


def test_cmd_test_wake_runs_recorded_eval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    model_path = tmp_path / "sayso.onnx"
    model_path.write_bytes(b"fake-onnx")
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    (eval_root / "cases.json").write_text(
        json.dumps({"version": 1, "cases": []}),
        encoding="utf-8",
    )

    cfg = _config(tmp_path)
    cfg.wake_word = SimpleNamespace(
        model=model_path,
        phrase="SaySo",
        threshold=0.65,
    )
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(
        "satellite.sayso.wake.eval.satellite_eval_root",
        lambda: eval_root,
    )

    mock_model = MagicMock()
    mock_model.predict.return_value = {"sayso": 0.0}
    fake_wakeword = MagicMock(WakeWordModel=MagicMock(return_value=mock_model))
    monkeypatch.setitem(sys.modules, "livekit", MagicMock(wakeword=fake_wakeword))
    monkeypatch.setitem(sys.modules, "livekit.wakeword", fake_wakeword)

    assert cli.cmd_test_wake(SimpleNamespace()) == 0


def test_cmd_test_wake_does_not_import_satellite_sayso() -> None:
    source = inspect.getsource(cli.cmd_test_wake)
    assert "satellite.sayso" not in source
    assert "from .wake.eval import" in source


def test_cmd_test_wake_importable_with_satellite_on_pythonpath(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    model_path = tmp_path / "sayso.onnx"
    model_path.write_bytes(b"fake-onnx")
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    (eval_root / "cases.json").write_text(
        json.dumps({"version": 1, "cases": []}),
        encoding="utf-8",
    )

    cfg = _config(tmp_path)
    cfg.wake_word = SimpleNamespace(
        model=model_path,
        phrase="SaySo",
        threshold=0.65,
    )

    satellite_root = Path(__file__).resolve().parents[1]
    if str(satellite_root) not in sys.path:
        sys.path.insert(0, str(satellite_root))

    deployed_cli = importlib.import_module("sayso.cli")

    monkeypatch.setattr(deployed_cli, "load_config", lambda: cfg)
    monkeypatch.setattr(
        "sayso.wake.eval.satellite_eval_root",
        lambda: eval_root,
    )

    mock_model = MagicMock()
    mock_model.predict.return_value = {"sayso": 0.0}
    fake_wakeword = MagicMock(WakeWordModel=MagicMock(return_value=mock_model))
    monkeypatch.setitem(sys.modules, "livekit", MagicMock(wakeword=fake_wakeword))
    monkeypatch.setitem(sys.modules, "livekit.wakeword", fake_wakeword)

    assert deployed_cli.cmd_test_wake(SimpleNamespace()) == 0


def test_mic_check_records_at_native_capture_rate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = SimpleNamespace(
        audio=SimpleNamespace(
            input_device="pulse/configured-mic",
            output_device="pulse/configured-speaker",
            sample_rate=16000,
            capture_rate=44100,
            channels=1,
            mic_gain_db=6.0,
        ),
        sounds=SimpleNamespace(wake=tmp_path / "wake.wav"),
    )
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    record = Mock(return_value=124)
    monkeypatch.setattr(cli.subprocess, "call", record)
    monkeypatch.setattr(cli, "_level_report", Mock())
    monkeypatch.setattr(cli, "_play_sound", Mock(return_value=0), raising=False)
    monkeypatch.setattr(cli, "_write_processed_copy", Mock(return_value=True))
    monkeypatch.setattr(cli, "CHECK_DIR", tmp_path)

    assert cli.cmd_test_mic(SimpleNamespace()) == 0
    command = record.call_args.args[0]
    # The sanity check must record the native device rate, not the transport rate.
    assert command[command.index("--rate") + 1] == "44100"


def test_processed_copy_applies_gain_and_resamples_to_16k(tmp_path: Path) -> None:
    import numpy as np
    import wave

    src = tmp_path / "native.wav"
    samples = np.full(4410, 1000, dtype="<i2")
    with wave.open(str(src), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(samples.tobytes())

    cfg = SimpleNamespace(
        audio=SimpleNamespace(sample_rate=16000, mic_gain_db=6.0),
    )
    dst = tmp_path / "processed.wav"
    assert cli._write_processed_copy(src, dst, cfg) is True

    with wave.open(str(dst), "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        out = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
    # ~100 ms of audio, and 6 dB (2x) gain applied.
    assert abs(out.size - 1600) <= 8
    assert int(np.max(out)) > 1000
