"""Explicit mpv playback outcome handling."""

from __future__ import annotations

import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from satellite.sayso.playback import (
    END_FILE_ABORT,
    END_FILE_EOF,
    END_FILE_ERROR,
    END_FILE_STOP,
    PlaybackOutcome,
    classify_end_file_reason,
    configure_pulse_mpv,
    install_playback_recovery,
    play_sound,
)


def test_classify_end_file_reason() -> None:
    assert classify_end_file_reason(END_FILE_EOF) is PlaybackOutcome.COMPLETED
    assert classify_end_file_reason(END_FILE_STOP) is PlaybackOutcome.INTERRUPTED
    assert classify_end_file_reason(END_FILE_ABORT) is PlaybackOutcome.INTERRUPTED
    assert classify_end_file_reason(END_FILE_ERROR) is PlaybackOutcome.FAILED
    assert classify_end_file_reason(99) is PlaybackOutcome.FAILED


def test_configure_pulse_mpv_sets_pulse_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    mpv_instance: dict[str, object] = {}
    mpv_constructor = Mock(return_value=mpv_instance)
    fake_mpv = SimpleNamespace(MPV=mpv_constructor)

    class FakeLibMpvPlayer:
        def __init__(self, device: str | None = None) -> None:
            self._mpv = fake_mpv.MPV(cache="yes")
            if device:
                self._mpv["audio-device"] = device

    package = ModuleType("linux_voice_assistant")
    package.__path__ = []  # type: ignore[attr-defined]
    player_package = ModuleType("linux_voice_assistant.player")
    player_package.__path__ = []  # type: ignore[attr-defined]
    libmpv = ModuleType("linux_voice_assistant.player.libmpv")
    libmpv.mpv = fake_mpv  # type: ignore[attr-defined]
    libmpv.LibMpvPlayer = FakeLibMpvPlayer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "linux_voice_assistant", package)
    monkeypatch.setitem(sys.modules, "linux_voice_assistant.player", player_package)
    monkeypatch.setitem(sys.modules, "linux_voice_assistant.player.libmpv", libmpv)

    configure_pulse_mpv()
    player = FakeLibMpvPlayer(device="pulse/speaker")
    mpv_constructor.assert_called_once_with(cache="yes", ao="pulse")
    assert mpv_instance["audio-device"] == "pulse/speaker"


def test_playback_recovery_invokes_cleanup_on_error_not_on_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = Mock()

    class FakeLibMpvPlayer:
        def __init__(self) -> None:
            self._done_callback = None

        def _on_end_file(self, event) -> None:
            if event.data.reason == 0 and self._done_callback:
                callback = self._done_callback
                self._done_callback = None
                callback()

    package = ModuleType("linux_voice_assistant")
    package.__path__ = []  # type: ignore[attr-defined]
    player_package = ModuleType("linux_voice_assistant.player")
    player_package.__path__ = []  # type: ignore[attr-defined]
    libmpv = ModuleType("linux_voice_assistant.player.libmpv")
    libmpv.mpv = SimpleNamespace(MPV=Mock())  # type: ignore[attr-defined]
    libmpv.LibMpvPlayer = FakeLibMpvPlayer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "linux_voice_assistant", package)
    monkeypatch.setitem(sys.modules, "linux_voice_assistant.player", player_package)
    monkeypatch.setitem(sys.modules, "linux_voice_assistant.player.libmpv", libmpv)

    install_playback_recovery()
    player = FakeLibMpvPlayer()
    player._done_callback = completed

    player._on_end_file(SimpleNamespace(data=SimpleNamespace(reason=END_FILE_ABORT)))
    completed.assert_not_called()

    player._done_callback = completed
    player._on_end_file(SimpleNamespace(data=SimpleNamespace(reason=END_FILE_ERROR)))
    completed.assert_called_once_with()


@pytest.mark.parametrize(("reason", "expected"), [(END_FILE_EOF, 0), (END_FILE_ERROR, 1)])
def test_play_sound_reports_explicit_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    reason: int,
    expected: int,
) -> None:
    monkeypatch.setattr("satellite.sayso.playback.configure_pulse_mpv", Mock())
    monkeypatch.setattr("satellite.sayso.playback.install_playback_recovery", Mock())

    class FakeLibMpvPlayer:
        def _on_end_file(self, _event) -> None:
            self.done()

    class FakeMpvMediaPlayer:
        def __init__(self, device: str) -> None:
            assert device == "pulse/speaker"
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

    assert play_sound("tone.wav", "pulse/speaker") == expected


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

    assert play_sound("tone.wav", "speaker", timeout=0) == 1
