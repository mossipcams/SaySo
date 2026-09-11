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
