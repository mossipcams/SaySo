from __future__ import annotations

import ast
import importlib
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from satellite.sayso.process_audio import _ResamplingRecorder, install_native_rate_capture


class _EventType:
    VOICE_ASSISTANT_ERROR = 2
    VOICE_ASSISTANT_RUN_END = 4


class _LVAEvent:
    WAKE_WORD_DETECTED = "wake_word_detected"


class _UpstreamProtocol:
    def __init__(self, state) -> None:
        self.state = state
        self._pipeline_active = False
        self.tts_finished_kwargs: list[dict] = []
        self.upstream_events: list[tuple] = []

    def _emit(self, *args) -> None:
        return None

    def duck(self) -> None:
        return None

    def unduck(self) -> None:
        return None

    def _start_audio_streaming(self, phrase: str) -> None:
        return None

    def handle_audio(self, audio_chunk, audio_chunk_2=None) -> None:
        return None

    def handle_voice_event(self, event_type, data) -> None:
        self.upstream_events.append((event_type, data))

    def _tts_finished(self, pipeline_ended: bool = False) -> None:
        self.tts_finished_kwargs.append({"pipeline_ended": pipeline_ended})

    def stop(self) -> None:
        return None


def _install_handlers(monkeypatch: pytest.MonkeyPatch, sounds, wake_hook=None, stt_capture=None):
    model = ModuleType("aioesphomeapi.model")
    model.VoiceAssistantEventType = _EventType
    events = ModuleType("linux_voice_assistant.events")
    events.LVAEvent = _LVAEvent
    monkeypatch.setitem(sys.modules, "aioesphomeapi.model", model)
    monkeypatch.setitem(sys.modules, "linux_voice_assistant.events", events)
    monkeypatch.setattr("satellite.sayso.events.threading.Timer", Mock())

    from satellite.sayso.events import install_voice_handlers

    protocol = type("VoiceSatelliteProtocol", (_UpstreamProtocol,), {})
    install_voice_handlers(protocol, sounds, wake_hook, stt_capture=stt_capture)
    return protocol


def _sounds(tmp_path):
    return SimpleNamespace(
        wake=tmp_path / "ack.wav",
        failure=tmp_path / "failure.wav",
        unavailable=tmp_path / "unavailable.wav",
    )


def _state(**overrides) -> SimpleNamespace:
    values = {"muted": False, "tts_player": Mock(spec=["play"])}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_tts_finished_forwards_pipeline_ended_kwarg(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    protocol = _install_handlers(monkeypatch, _sounds(tmp_path))
    satellite = protocol(_state())

    satellite._tts_finished(pipeline_ended=True)

    assert satellite.tts_finished_kwargs == [{"pipeline_ended": True}]


def test_tts_finished_forwards_positional_arguments(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    protocol = _install_handlers(monkeypatch, _sounds(tmp_path))
    satellite = protocol(_state())

    satellite._tts_finished(True)

    assert satellite.tts_finished_kwargs == [{"pipeline_ended": True}]


@pytest.mark.parametrize(
    ("configured", "expected"),
    [("speaker", "pulse/speaker"), ("pulse/speaker", "pulse/speaker")],
)
def test_launcher_passes_pulse_output_device_and_no_pipeline_ended_sound(
    monkeypatch: pytest.MonkeyPatch, configured: str, expected: str
) -> None:
    package = ModuleType("linux_voice_assistant")
    package.__path__ = []
    upstream = ModuleType("linux_voice_assistant.__main__")
    upstream.run = Mock()
    upstream.process_audio = Mock()
    satellite_module = ModuleType("linux_voice_assistant.satellite")
    satellite_module.VoiceSatelliteProtocol = object
    monkeypatch.setitem(sys.modules, "linux_voice_assistant", package)
    monkeypatch.setitem(sys.modules, "linux_voice_assistant.__main__", upstream)
    monkeypatch.setitem(sys.modules, "linux_voice_assistant.satellite", satellite_module)
    monkeypatch.setattr(sys, "argv", list(sys.argv))

    launcher = importlib.import_module("satellite.sayso.launcher")
    cfg = SimpleNamespace(
        satellite=SimpleNamespace(name="Living Room", device_name="sayso-living-room"),
        home_assistant=SimpleNamespace(port=6053),
        audio=SimpleNamespace(
            input_device="mic",
            output_device=configured,
            channels=1,
            noise_suppression=0,
            auto_gain=0,
        ),
        wake_word=SimpleNamespace(
            provider="livekit",
            model="wake.onnx",
            phrase="SaySo",
            threshold=0.5,
            refractory_seconds=2.0,
            post_tts_cooldown_ms=500,
            preroll_ms=500,
            wake_skip_ms=500,
        ),
        sounds=SimpleNamespace(wake="ack.wav", failure="failure.wav", unavailable="unavailable.wav"),
    )
    monkeypatch.setattr(launcher, "load_config", lambda: cfg)
    monkeypatch.setattr(
        launcher,
        "LiveKitWakeWordProvider",
        lambda **_kwargs: SimpleNamespace(available=True, predict_window=Mock(return_value=None)),
    )
    monkeypatch.setattr(launcher, "install_wake_audio_path", Mock())
    monkeypatch.setattr(launcher, "install_voice_handlers", Mock())
    monkeypatch.setattr(launcher, "_configure_mpv", Mock())

    launcher.main()

    argv = sys.argv
    assert argv[argv.index("--audio-output-device") + 1] == expected
    assert argv[argv.index("--pipeline-ended-sound") + 1] == ""
    upstream.run.assert_called_once_with()


class _FakeRecorder:
    def __init__(self, samplerate: int, channels: int, blocksize: int) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize

    def __enter__(self) -> "_FakeRecorder":
        return self

    def __exit__(self, *exc) -> None:
        return None


class _FakeMic:
    def __init__(self, mic_id: str | None) -> None:
        self.id = mic_id
        self.name = "snowball"

    def recorder(self, samplerate, channels, blocksize):
        return _FakeRecorder(samplerate, channels, blocksize)


class _UpstreamReconnectingRecorder:
    def __init__(self, mic, n_channels: int, block_size: int) -> None:
        self._mic = mic
        self._n_channels = n_channels
        self._block_size = block_size
        self._recorder = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *exc_info) -> None:
        self._recorder = None

    def _open(self) -> None:
        self._recorder = self._mic.recorder(
            samplerate=16000, channels=self._n_channels, blocksize=self._block_size
        )

    def _refresh_mic(self) -> None:
        mic_id = getattr(self._mic, "id", None)
        if mic_id is not None:
            self._mic = _FakeMic(mic_id)


def _install_reconnecting_lva(opened: list) -> ModuleType:
    lva_main = ModuleType("linux_voice_assistant.__main__")

    def process_audio(state, mic, blocksize):
        with _UpstreamReconnectingRecorder(mic, 1, blocksize) as recorder:
            recorder._refresh_mic()
            recorder._open()
            opened.append(recorder._recorder)

    lva_main.process_audio = process_audio
    lva_main._ReconnectingRecorder = _UpstreamReconnectingRecorder
    install_native_rate_capture(lva_main, capture_rate=44100, gain_db=0.0, channels=1)
    return lva_main


def test_recorder_opened_after_mic_refresh_keeps_resampling_wrapper() -> None:
    opened: list = []
    lva_main = _install_reconnecting_lva(opened)

    lva_main.process_audio(SimpleNamespace(), _FakeMic("snowball"), 1024)

    assert isinstance(opened[-1], _ResamplingRecorder)


def test_refresh_without_new_mic_does_not_stack_resampling_wrappers() -> None:
    opened: list = []
    lva_main = _install_reconnecting_lva(opened)

    lva_main.process_audio(SimpleNamespace(), _FakeMic(None), 1024)

    assert isinstance(opened[-1], _ResamplingRecorder)
    assert not isinstance(opened[-1]._recorder, _ResamplingRecorder)


def test_install_tolerates_lva_main_without_reconnecting_recorder() -> None:
    lva_main = ModuleType("linux_voice_assistant.__main__")
    lva_main.process_audio = lambda state, mic, blocksize: None

    install_native_rate_capture(lva_main, capture_rate=44100, gain_db=0.0, channels=1)

    assert callable(lva_main.process_audio)


def test_wakeup_records_wake_word_phrase(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    protocol = _install_handlers(monkeypatch, _sounds(tmp_path))
    satellite = protocol(_state())
    satellite._wake_word_phrase = ""

    satellite.wakeup(SimpleNamespace(wake_word="Koda"))

    assert satellite._wake_word_phrase == "Koda"


def test_duplicate_wake_up_error_plays_no_failure_sound_and_still_finishes(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    sounds = _sounds(tmp_path)
    wake_hook = Mock()
    stt_capture = Mock()
    protocol = _install_handlers(monkeypatch, sounds, wake_hook=wake_hook, stt_capture=stt_capture)
    state = _state()
    satellite = protocol(state)
    satellite._sayso_capture_id = "cmd-1"
    satellite._sayso_wake_capture_id = None

    satellite.handle_voice_event(_EventType.VOICE_ASSISTANT_ERROR, {"code": "duplicate_wake_up_detected"})
    satellite._tts_finished(pipeline_ended=False)

    state.tts_player.play.assert_not_called()
    wake_hook.rearm.assert_called_once_with()
    stt_capture.end_command.assert_called_once()
    assert stt_capture.end_command.call_args.args[0] == "cmd-1"


def test_patched_lva_tree_exports_lva_event() -> None:
    lva_src = os.environ.get("SAYSO_LVA_SRC")
    if not lva_src:
        pytest.skip("SAYSO_LVA_SRC is not set")

    events_path = Path(lva_src) / "linux_voice_assistant" / "events.py"
    assert events_path.is_file(), f"missing {events_path}"
    tree = ast.parse(events_path.read_text(encoding="utf-8"))
    exported = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "peripheral_api"
        for alias in node.names
    ]
    assert "LVAEvent" in exported
