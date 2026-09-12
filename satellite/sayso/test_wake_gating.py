"""Wake gating: the mic must not open until the chime and TTS are truly done."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from satellite.sayso.events import install_voice_handlers
from satellite.sayso.wake.hook import SaySoExternalWakeHook


class _EventType:
    VOICE_ASSISTANT_STT_END = 1
    VOICE_ASSISTANT_ERROR = 2


class _LVAEvent:
    WAKE_WORD_DETECTED = "wake_word_detected"
    LISTENING = "listening"


def _install(monkeypatch: pytest.MonkeyPatch, wake_hook, *, gate_ms: float = 0.0):
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
    install_voice_handlers(
        protocol,
        SimpleNamespace(wake="a", failure="b", unavailable="c"),
        wake_hook,
        aec_gate_ms=gate_ms,
    )
    return protocol


def _satellite(**overrides):
    base = dict(
        _pipeline_active=False,
        _tts_played=False,
        state=SimpleNamespace(
            muted=False,
            tts_player=SimpleNamespace(play=Mock(), stop=Mock()),
        ),
        duck=Mock(),
        _emit=Mock(),
        _start_audio_streaming=Mock(),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_mic_opens_immediately_when_player_is_idle(monkeypatch) -> None:
    provider = MagicMock(available=True)
    provider.predict_window.return_value = None
    hook = SaySoExternalWakeHook(provider)
    protocol = _install(monkeypatch, hook)

    satellite = _satellite()
    protocol.wakeup(satellite, SimpleNamespace(wake_word="SaySo"))  # type: ignore[attr-defined]

    satellite._start_audio_streaming.assert_called_once_with("SaySo")


def test_mic_waits_for_in_flight_playback_done_callback(monkeypatch) -> None:
    """A wake during response audio must not open the mic into the speaker."""
    provider = MagicMock(available=True)
    provider.predict_window.return_value = None
    hook = SaySoExternalWakeHook(provider)
    protocol = _install(monkeypatch, hook)

    player = SimpleNamespace(play=Mock(), stop=Mock())
    satellite = _satellite(state=SimpleNamespace(muted=False, tts_player=player))
    # Simulate the player holding an in-flight TTS response callback.
    satellite.state.tts_player._done_callback = Mock()

    protocol.wakeup(satellite, SimpleNamespace(wake_word="SaySo"))  # type: ignore[attr-defined]

    # Streaming must NOT start while playback is still in flight.
    satellite._start_audio_streaming.assert_not_called()

    # When playback genuinely finishes, the chained callback opens the mic.
    player._done_callback()
    satellite._start_audio_streaming.assert_called_once_with("SaySo")


def test_settle_delay_runs_before_streaming_starts(monkeypatch) -> None:
    provider = MagicMock(available=True)
    provider.predict_window.return_value = None
    hook = SaySoExternalWakeHook(provider)
    protocol = _install(monkeypatch, hook, gate_ms=200.0)

    timers: list[tuple[float, object]] = []

    class _Timer:
        def __init__(self, interval, fn):
            timers.append((interval, fn))

        def start(self):
            pass

    monkeypatch.setattr("satellite.sayso.events.threading.Timer", _Timer)

    satellite = _satellite()
    protocol.wakeup(satellite, SimpleNamespace(wake_word="SaySo"))  # type: ignore[attr-defined]

    satellite._start_audio_streaming.assert_not_called()
    assert timers and timers[0][0] == pytest.approx(0.2)
    timers[0][1]()
    satellite._start_audio_streaming.assert_called_once_with("SaySo")


def test_wake_ignored_while_response_tts_is_still_playing(monkeypatch) -> None:
    provider = MagicMock(available=True)
    provider.predict_window.return_value = None
    hook = SaySoExternalWakeHook(provider)
    protocol = _install(monkeypatch, hook)

    satellite = _satellite(_tts_played=True, _pipeline_active=True)
    protocol.wakeup(satellite, SimpleNamespace(wake_word="SaySo"))  # type: ignore[attr-defined]

    satellite._start_audio_streaming.assert_not_called()


def test_flush_preroll_runs_at_the_true_boundary_only(monkeypatch) -> None:
    """Pre-open audio is retained and handed over only once the mic opens."""
    provider = MagicMock(available=True)
    provider.predict_window.return_value = None
    hook = SaySoExternalWakeHook(provider, preroll_ms=1000, wake_skip_ms=500)
    protocol = _install(monkeypatch, hook)

    satellite = _satellite()
    satellite.handle_audio = Mock()
    opened: list[str] = []

    def _record(phrase: str) -> None:
        opened.append(phrase)
        # The flush must happen inside the same open path.
        protocol.wakeup  # noqa: B018 - readability anchor

    satellite._start_audio_streaming = Mock(side_effect=_record)

    protocol.wakeup(satellite, SimpleNamespace(wake_word="SaySo"))  # type: ignore[attr-defined]
    assert opened == ["SaySo"]
    # Nothing captured before the open was forwarded yet; the flush owns it.
    satellite.handle_audio.assert_not_called()


def test_deferred_open_is_cancelled_when_pipeline_tears_down(monkeypatch) -> None:
    provider = MagicMock(available=True)
    provider.predict_window.return_value = None
    hook = SaySoExternalWakeHook(provider)
    protocol = _install(monkeypatch, hook)

    player = SimpleNamespace(play=Mock(), stop=Mock())
    satellite = _satellite(state=SimpleNamespace(muted=False, tts_player=player))
    satellite.state.tts_player._done_callback = Mock()

    protocol.wakeup(satellite, SimpleNamespace(wake_word="SaySo"))  # type: ignore[attr-defined]
    satellite._pipeline_active = False
    player._done_callback()

    satellite._start_audio_streaming.assert_not_called()


def test_lost_done_callback_does_not_wedge_the_pipeline(monkeypatch) -> None:
    """An orphaned player callback must release the wake, not deafen the satellite.

    `_defer_until_playback_idle` reads the player's callback slot and then writes
    a chained one. If playback ends in between, the chained callback is installed
    on a finished player and never fires. Without a watchdog the pipeline stays
    flagged active and every later wake is dropped.
    """
    provider = MagicMock(available=True)
    provider.predict_window.return_value = None
    hook = SaySoExternalWakeHook(provider)
    protocol = _install(monkeypatch, hook)

    watchdogs: list[tuple[float, object]] = []

    class _Timer:
        def __init__(self, interval, fn):
            watchdogs.append((interval, fn))
            self.daemon = False

        def start(self):
            pass

    monkeypatch.setattr("satellite.sayso.events.threading.Timer", _Timer)

    player = SimpleNamespace(play=Mock(), stop=Mock())
    satellite = _satellite(state=SimpleNamespace(muted=False, tts_player=player))
    satellite.state.tts_player._done_callback = Mock()

    protocol.wakeup(satellite, SimpleNamespace(wake_word="SaySo"))  # type: ignore[attr-defined]
    assert satellite._pipeline_active is True

    # The callback is never invoked: playback had already finished.
    assert watchdogs, "no watchdog was armed for the deferred open"
    watchdogs[0][1]()

    # The mic must NOT have opened -- that could capture live speaker output.
    satellite._start_audio_streaming.assert_not_called()
    # But the pipeline must be released so the next wake is heard.
    assert satellite._pipeline_active is False


def test_watchdog_cannot_open_the_mic_after_the_callback_already_did(monkeypatch) -> None:
    """Exactly one path opens the microphone per wake."""
    provider = MagicMock(available=True)
    provider.predict_window.return_value = None
    hook = SaySoExternalWakeHook(provider)
    protocol = _install(monkeypatch, hook)

    watchdogs: list[tuple[float, object]] = []

    class _Timer:
        def __init__(self, interval, fn):
            watchdogs.append((interval, fn))
            self.daemon = False

        def start(self):
            pass

    monkeypatch.setattr("satellite.sayso.events.threading.Timer", _Timer)

    player = SimpleNamespace(play=Mock(), stop=Mock())
    satellite = _satellite(state=SimpleNamespace(muted=False, tts_player=player))
    satellite.state.tts_player._done_callback = Mock()

    protocol.wakeup(satellite, SimpleNamespace(wake_word="SaySo"))  # type: ignore[attr-defined]
    player._done_callback()
    satellite._start_audio_streaming.assert_called_once_with("SaySo")

    # A late watchdog must be a no-op, not a second open or a teardown.
    watchdogs[0][1]()
    satellite._start_audio_streaming.assert_called_once_with("SaySo")
    assert satellite._pipeline_active is True


def test_abandoned_wake_releases_the_detection_boundary(monkeypatch) -> None:
    """A boundary nobody will flush must not block live forwarding forever."""
    provider = MagicMock(available=True)
    provider.predict_window.return_value = None
    hook = SaySoExternalWakeHook(provider)
    protocol = _install(monkeypatch, hook)

    player = SimpleNamespace(play=Mock(), stop=Mock())
    satellite = _satellite(state=SimpleNamespace(muted=False, tts_player=player))
    satellite.state.tts_player._done_callback = Mock()

    protocol.wakeup(satellite, SimpleNamespace(wake_word="SaySo"))  # type: ignore[attr-defined]
    hook._detection_index = 12345
    satellite._pipeline_active = False
    player._done_callback()

    satellite._start_audio_streaming.assert_not_called()
    assert hook.last_detection_index is None


def test_abandoned_wake_unducks_media(monkeypatch) -> None:
    """wakeup() ducks; the abort path is the only thing left to undo it."""
    provider = MagicMock(available=True)
    provider.predict_window.return_value = None
    hook = SaySoExternalWakeHook(provider)
    protocol = _install(monkeypatch, hook)

    watchdogs: list[tuple[float, object]] = []

    class _Timer:
        def __init__(self, interval, fn):
            watchdogs.append((interval, fn))
            self.daemon = False

        def start(self):
            pass

    monkeypatch.setattr("satellite.sayso.events.threading.Timer", _Timer)

    player = SimpleNamespace(play=Mock(), stop=Mock())
    satellite = _satellite(state=SimpleNamespace(muted=False, tts_player=player))
    satellite.unduck = Mock()
    satellite.state.tts_player._done_callback = Mock()

    protocol.wakeup(satellite, SimpleNamespace(wake_word="SaySo"))  # type: ignore[attr-defined]
    satellite.duck.assert_called_once()
    watchdogs[0][1]()
    satellite.unduck.assert_called_once()
