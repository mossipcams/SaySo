from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call

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
    monkeypatch.setattr("satellite.sayso.launcher._configure_mpv", configured)

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
    monkeypatch.setattr("satellite.sayso.launcher._configure_mpv", Mock())

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
