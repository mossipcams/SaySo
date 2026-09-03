"""Voice pipeline event hooks for post-STT acknowledgement sounds."""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from satellite.sayso.config import SoundsCfg
from satellite.sayso.wake.hook import SaySoExternalWakeHook


class _EventType:
    VOICE_ASSISTANT_STT_END = 1
    VOICE_ASSISTANT_ERROR = 2
    VOICE_ASSISTANT_RUN_START = 3


class _LVAEvent:
    WAKE_WORD_DETECTED = "wake_word_detected"


def _install_test_handlers(monkeypatch: pytest.MonkeyPatch, sounds: SoundsCfg, wake_hook=None):
    model = ModuleType("aioesphomeapi.model")
    model.VoiceAssistantEventType = _EventType  # type: ignore[attr-defined]
    events = ModuleType("linux_voice_assistant.events")
    events.LVAEvent = _LVAEvent  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aioesphomeapi.model", model)
    monkeypatch.setitem(sys.modules, "linux_voice_assistant.events", events)

    from satellite.sayso.events import install_voice_handlers

    protocol = type(
        "VoiceSatelliteProtocol",
        (),
        {
            "wakeup": Mock(),
            "handle_voice_event": Mock(),
            "_tts_finished": Mock(),
            "stop": Mock(),
        },
    )
    install_voice_handlers(protocol, sounds, wake_hook)
    return protocol


def _sounds(tmp_path) -> SoundsCfg:
    ack = tmp_path / "ack.wav"
    failure = tmp_path / "failure.wav"
    unavailable = tmp_path / "unavailable.wav"
    for path in (ack, failure, unavailable):
        path.write_bytes(b"wav")
    return SoundsCfg(wake=ack, failure=failure, unavailable=unavailable)


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
    """Models LVA MpvMediaPlayer: user callback on wrapper; inner EOF drives _on_track_finished."""

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


def test_wakeup_suspends_external_wake_hook(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    sounds = _sounds(tmp_path)
    wake_hook = MagicMock()
    protocol = _install_test_handlers(monkeypatch, sounds, wake_hook)

    satellite = SimpleNamespace(
        state=SimpleNamespace(muted=False),
        _pipeline_active=False,
        _timer_finished=False,
        _timer_ring_start=None,
        duck=Mock(),
        _emit=Mock(),
        _start_audio_streaming=Mock(),
    )
    wake_word = SimpleNamespace(wake_word="SaySo")

    protocol.wakeup(satellite, wake_word)  # type: ignore[attr-defined]

    wake_hook.suspend.assert_called_once_with()


def test_wakeup_starts_streaming_without_wake_chime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    sounds = _sounds(tmp_path)
    protocol = _install_test_handlers(monkeypatch, sounds)

    satellite = SimpleNamespace(
        state=SimpleNamespace(muted=False),
        _pipeline_active=False,
        _timer_finished=False,
        _timer_ring_start=None,
        duck=Mock(),
        _emit=Mock(),
        _start_audio_streaming=Mock(),
    )
    wake_word = SimpleNamespace(wake_word="SaySo")

    protocol.wakeup(satellite, wake_word)  # type: ignore[attr-defined]

    satellite.duck.assert_called_once_with()
    satellite._start_audio_streaming.assert_called_once_with("SaySo")


def test_wakeup_flushes_preroll_after_streaming_starts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    sounds = _sounds(tmp_path)
    provider = MagicMock(available=True, predict_window=MagicMock(return_value=None))
    hook = SaySoExternalWakeHook(provider, preroll_ms=1000, wake_skip_ms=500)
    protocol = _install_test_handlers(monkeypatch, sounds, hook)

    wake_samples = np.zeros(8000, dtype="<i2")
    command_samples = np.full(8000, 7, dtype="<i2")
    hook.feed_pcm(SimpleNamespace(satellite=None), wake_samples.tobytes())
    hook.feed_pcm(SimpleNamespace(satellite=None), command_samples.tobytes())

    streaming_order: list[str] = []
    handle_audio_calls: list[bytes] = []

    def _start_streaming(_phrase: str) -> None:
        streaming_order.append("start")

    def _handle_audio(chunk: bytes, _chunk2: bytes | None = None) -> None:
        streaming_order.append("audio")
        handle_audio_calls.append(chunk)

    satellite = SimpleNamespace(
        state=SimpleNamespace(muted=False),
        _pipeline_active=False,
        _timer_finished=False,
        _timer_ring_start=None,
        duck=Mock(),
        _emit=Mock(),
        _start_audio_streaming=Mock(side_effect=_start_streaming),
        handle_audio=_handle_audio,
    )
    wake_word = SimpleNamespace(wake_word="SaySo")

    protocol.wakeup(satellite, wake_word)  # type: ignore[attr-defined]

    assert streaming_order == ["start", "audio"]
    assert len(handle_audio_calls) == 1
    flushed = np.frombuffer(handle_audio_calls[0], dtype="<i2")
    assert flushed.size == 8000
    assert np.all(flushed == 7)
    assert not np.any(flushed == 0)


def test_stt_end_defers_chime_until_after_handle_voice_event_returns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    sounds = _sounds(tmp_path)
    tts_player = _FakeMpvMediaPlayer()
    protocol = _install_test_handlers(monkeypatch, sounds)
    satellite = SimpleNamespace(state=SimpleNamespace(tts_player=tts_player))

    played_during_handler: list[bool] = []
    original_play = tts_player.play
    inside_handler = False

    def tracking_play(path: str, done_callback=None) -> None:
        if inside_handler:
            played_during_handler.append(True)
        original_play(path, done_callback)

    tts_player.play = tracking_play  # type: ignore[method-assign]

    async def _invoke_under_running_loop() -> None:
        nonlocal inside_handler
        inside_handler = True
        protocol.handle_voice_event(  # type: ignore[attr-defined]
            satellite,
            _EventType.VOICE_ASSISTANT_STT_END,
            {"text": "turn on the lights"},
        )
        inside_handler = False
        assert played_during_handler == []
        assert len(tts_player.play_calls) == 0

    asyncio.run(_invoke_under_running_loop())
    assert len(tts_player.play_calls) == 1
    assert tts_player.play_calls[0] == (str(sounds.wake), None)


def test_stt_end_with_text_plays_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    sounds = _sounds(tmp_path)
    tts_player = _FakeMpvMediaPlayer()
    protocol = _install_test_handlers(monkeypatch, sounds)

    satellite = SimpleNamespace(state=SimpleNamespace(tts_player=tts_player))

    protocol.handle_voice_event(  # type: ignore[attr-defined]
        satellite,
        _EventType.VOICE_ASSISTANT_STT_END,
        {"text": "turn on the lights"},
    )

    assert len(tts_player.play_calls) == 1
    assert tts_player.play_calls[0] == (str(sounds.wake), None)


def test_stt_end_empty_plays_failure_not_ack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    sounds = _sounds(tmp_path)
    tts_player = _FakeMpvMediaPlayer()
    protocol = _install_test_handlers(monkeypatch, sounds)

    satellite = SimpleNamespace(state=SimpleNamespace(tts_player=tts_player))

    protocol.handle_voice_event(  # type: ignore[attr-defined]
        satellite,
        _EventType.VOICE_ASSISTANT_STT_END,
        {"text": "   "},
    )

    assert len(tts_player.play_calls) == 1
    assert tts_player.play_calls[0][0] == str(sounds.failure)
    assert callable(tts_player.play_calls[0][1])


def test_voice_assistant_error_plays_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    sounds = _sounds(tmp_path)
    tts_player = _FakeMpvMediaPlayer()
    protocol = _install_test_handlers(monkeypatch, sounds)

    satellite = SimpleNamespace(state=SimpleNamespace(tts_player=tts_player))

    protocol.handle_voice_event(satellite, _EventType.VOICE_ASSISTANT_ERROR, {})  # type: ignore[attr-defined]

    assert len(tts_player.play_calls) == 1
    assert tts_player.play_calls[0][0] == str(sounds.failure)
    assert callable(tts_player.play_calls[0][1])


def test_stt_ack_chains_onto_in_flight_tts_callback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    sounds = _sounds(tmp_path)
    protocol = _install_test_handlers(monkeypatch, sounds)
    tts_player = _FakeMpvMediaPlayer()
    tts_finished = Mock()
    tts_player.play("in-flight-tts.wav", done_callback=tts_finished)
    inner_on_track_finished = tts_player._player._on_track_finished
    satellite = SimpleNamespace(state=SimpleNamespace(tts_player=tts_player))

    protocol.handle_voice_event(  # type: ignore[attr-defined]
        satellite,
        _EventType.VOICE_ASSISTANT_STT_END,
        {"text": "turn on the lights"},
    )

    chained = tts_player._done_callback
    assert chained is not tts_finished
    assert tts_player._player._on_track_finished is inner_on_track_finished
    tts_finished.assert_not_called()
    tts_player.eof()
    assert len(tts_player.play_calls) == 2
    assert tts_player.play_calls[1][0] == str(sounds.wake)
    tts_finished.assert_called_once_with()


def test_failure_chime_defers_rearm_until_done_callback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    sounds = _sounds(tmp_path)
    wake_hook = MagicMock()
    protocol = _install_test_handlers(monkeypatch, sounds, wake_hook)
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
    wake_hook.rearm.assert_not_called()

    protocol._tts_finished(satellite)  # type: ignore[attr-defined]
    wake_hook.rearm.assert_not_called()

    tts_player.eof()
    assert satellite._chime_rearm_pending is False
    wake_hook.rearm.assert_called_once_with()


def test_error_chime_does_not_steal_in_flight_tts_callback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    sounds = _sounds(tmp_path)
    wake_hook = MagicMock()
    protocol = _install_test_handlers(monkeypatch, sounds, wake_hook)
    tts_player = _FakeMpvMediaPlayer()
    tts_finished = Mock()
    tts_player.play("in-flight-tts.wav", done_callback=tts_finished)
    inner_on_track_finished = tts_player._player._on_track_finished
    satellite = SimpleNamespace(
        state=SimpleNamespace(tts_player=tts_player),
        _chime_rearm_pending=False,
    )

    protocol.handle_voice_event(satellite, _EventType.VOICE_ASSISTANT_ERROR, {})  # type: ignore[attr-defined]

    chained = tts_player._done_callback
    assert chained is not tts_finished
    assert tts_player._player._on_track_finished is inner_on_track_finished
    tts_player.eof()
    assert len(tts_player.play_calls) == 2
    assert tts_player.play_calls[1][0] == str(sounds.failure)
    tts_finished.assert_called_once_with()
    wake_hook.rearm.assert_not_called()

    tts_player.eof()
    wake_hook.rearm.assert_called_once_with()
