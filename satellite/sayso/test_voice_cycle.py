"""Regression coverage for wake recovery after a terminal TTS error."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from satellite.sayso.config import SoundsCfg
from satellite.sayso.events import install_voice_handlers
from satellite.sayso.wake.hook import SaySoExternalWakeHook


class _EventType:
    VOICE_ASSISTANT_STT_END = 1
    VOICE_ASSISTANT_ERROR = 2


class _LVAEvent:
    WAKE_WORD_DETECTED = "wake_word_detected"
    LISTENING = "listening"


class _FakeLibMpvPlayer:
    def __init__(self) -> None:
        self._on_track_finished = None

    def play(self, path: str, done_callback=None, stop_first: bool = False) -> None:
        self._on_track_finished = done_callback

    def eof(self) -> None:
        if self._on_track_finished is not None:
            callback = self._on_track_finished
            self._on_track_finished = None
            callback()


class _FakeMpvMediaPlayer:
    def __init__(self) -> None:
        self._player = _FakeLibMpvPlayer()
        self._done_callback = None
        self.play_calls: list[tuple[str, object | None]] = []

    def play(self, path: str, done_callback=None) -> None:
        self._done_callback = done_callback
        self.play_calls.append((path, done_callback))
        self._player.play(path, done_callback=self._on_track_finished)

    def _on_track_finished(self) -> None:
        callback = self._done_callback
        self._done_callback = None
        if callback is not None:
            callback()

    def eof(self) -> None:
        self._player.eof()


def _sounds(tmp_path) -> SoundsCfg:
    ack = tmp_path / "ack.wav"
    failure = tmp_path / "failure.wav"
    unavailable = tmp_path / "unavailable.wav"
    for path in (ack, failure, unavailable):
        path.write_bytes(b"wav")
    return SoundsCfg(wake=ack, failure=failure, unavailable=unavailable)


def _install_cycle_handlers(monkeypatch: pytest.MonkeyPatch, sounds: SoundsCfg, wake_hook):
    model = ModuleType("aioesphomeapi.model")
    model.VoiceAssistantEventType = _EventType  # type: ignore[attr-defined]
    events = ModuleType("linux_voice_assistant.events")
    events.LVAEvent = _LVAEvent  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aioesphomeapi.model", model)
    monkeypatch.setitem(sys.modules, "linux_voice_assistant.events", events)

    protocol = type(
        "VoiceSatelliteProtocol",
        (),
        {
            "handle_voice_event": Mock(),
            "_tts_finished": Mock(),
            "stop": Mock(),
        },
    )
    install_voice_handlers(protocol, sounds, wake_hook)
    return protocol


def test_tts_finished_rearms_wake_without_resetting_on_every_capture_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MagicMock(available=True)
    provider.predict_window.return_value = None
    hook = SaySoExternalWakeHook(provider)
    sounds = SimpleNamespace(wake="a.wav", failure="b.wav", unavailable="c.wav")
    protocol = _install_cycle_handlers(monkeypatch, sounds, hook)

    satellite = SimpleNamespace(
        _pipeline_active=False,
        state=SimpleNamespace(muted=False, tts_player=SimpleNamespace(play=Mock())),
        duck=Mock(),
        _emit=Mock(),
        _start_audio_streaming=Mock(),
    )
    wake_word = SimpleNamespace(wake_word="SaySo")

    protocol.wakeup(satellite, wake_word)  # type: ignore[attr-defined]
    assert hook._suspended is True
    pcm = np.zeros(512, dtype="<i2").tobytes()
    hook.feed_pcm(SimpleNamespace(satellite=satellite), pcm)
    provider.predict_window.assert_not_called()

    protocol._tts_finished(satellite)  # type: ignore[attr-defined]
    provider.reset.assert_called_once()
    assert hook._suspended is False

    hook.feed_pcm(SimpleNamespace(satellite=satellite), pcm)
    # Suspended blocks inference; after rearm the capture thread may enqueue again.


def test_empty_stt_rearms_wake_after_failure_chime_eof(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    provider = MagicMock(available=True)
    hook = SaySoExternalWakeHook(provider)
    sounds = _sounds(tmp_path)
    protocol = _install_cycle_handlers(monkeypatch, sounds, hook)
    tts_player = _FakeMpvMediaPlayer()
    satellite = SimpleNamespace(
        state=SimpleNamespace(tts_player=tts_player),
        _chime_rearm_pending=False,
    )

    protocol.handle_voice_event(  # type: ignore[attr-defined]
        satellite,
        _EventType.VOICE_ASSISTANT_STT_END,
        {"text": "   "},
    )
    assert satellite._chime_rearm_pending is True
    provider.reset.assert_not_called()

    protocol._tts_finished(satellite)  # type: ignore[attr-defined]
    provider.reset.assert_not_called()

    tts_player.eof()
    assert satellite._chime_rearm_pending is False
    provider.reset.assert_called_once()


def test_error_during_tts_invokes_tts_callback_then_failure_chime_then_rearm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    provider = MagicMock(available=True)
    hook = SaySoExternalWakeHook(provider)
    sounds = _sounds(tmp_path)
    protocol = _install_cycle_handlers(monkeypatch, sounds, hook)
    tts_player = _FakeMpvMediaPlayer()
    tts_finished = Mock()
    tts_player.play("in-flight-tts.wav", done_callback=tts_finished)
    satellite = SimpleNamespace(
        state=SimpleNamespace(tts_player=tts_player),
        _chime_rearm_pending=False,
    )

    protocol.handle_voice_event(satellite, _EventType.VOICE_ASSISTANT_ERROR, {})  # type: ignore[attr-defined]
    assert satellite._chime_rearm_pending is True
    provider.reset.assert_not_called()

    tts_player.eof()
    tts_finished.assert_called_once_with()
    assert len(tts_player.play_calls) == 2
    assert tts_player.play_calls[1][0] == str(sounds.failure)
    provider.reset.assert_not_called()

    tts_player.eof()
    assert satellite._chime_rearm_pending is False
    provider.reset.assert_called_once()
