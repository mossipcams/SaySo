import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


def test_launcher_rejects_unavailable_wake_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    package = ModuleType("linux_voice_assistant")
    package.__path__ = []  # type: ignore[attr-defined]
    models = ModuleType("linux_voice_assistant.models")
    models.ServerState = object  # type: ignore[attr-defined]
    webrtc = ModuleType("linux_voice_assistant.webrtc")
    webrtc.WebRTCProcessor = object  # type: ignore[attr-defined]
    upstream = ModuleType("linux_voice_assistant.__main__")
    upstream.run = Mock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "linux_voice_assistant", package)
    monkeypatch.setitem(sys.modules, "linux_voice_assistant.models", models)
    monkeypatch.setitem(sys.modules, "linux_voice_assistant.webrtc", webrtc)
    monkeypatch.setitem(sys.modules, "linux_voice_assistant.__main__", upstream)

    launcher = importlib.import_module("satellite.sayso.launcher")
    cfg = SimpleNamespace(
        satellite=SimpleNamespace(name="sayso-living-room"),
        home_assistant=SimpleNamespace(port=6053),
        audio=SimpleNamespace(
            input_device="mic",
            output_device="pulse/speaker",
            channels=1,
            noise_suppression=0,
            auto_gain=0,
        ),
        wake_word=SimpleNamespace(
            provider="livekit",
            model="invalid.onnx",
            phrase="SaySo",
            threshold=0.5,
            refractory_seconds=2.0,
            post_tts_cooldown_ms=500,
            preroll_ms=500,
        ),
        sounds=SimpleNamespace(wake="wake.wav"),
    )
    monkeypatch.setattr(launcher, "load_config", lambda: cfg)
    monkeypatch.setattr(
        launcher,
        "LiveKitWakeWordProvider",
        lambda **_kwargs: SimpleNamespace(available=False),
    )

    with pytest.raises(SystemExit, match="Wake detection is not operational"):
        launcher.main()

    upstream.run.assert_not_called()  # type: ignore[attr-defined]


def test_launcher_keeps_wakeup_sound_out_of_stt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = ModuleType("linux_voice_assistant")
    package.__path__ = []  # type: ignore[attr-defined]
    upstream = ModuleType("linux_voice_assistant.__main__")
    upstream.process_audio = Mock()  # type: ignore[attr-defined]
    upstream.run = Mock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "linux_voice_assistant", package)
    monkeypatch.setitem(sys.modules, "linux_voice_assistant.__main__", upstream)

    launcher = importlib.import_module("satellite.sayso.launcher")
    cfg = SimpleNamespace(
        satellite=SimpleNamespace(name="sayso-living-room"),
        home_assistant=SimpleNamespace(port=6053),
        audio=SimpleNamespace(
            input_device="mic",
            output_device="pulse/speaker",
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
        ),
        sounds=SimpleNamespace(wake="wake.wav"),
    )
    monkeypatch.setattr(launcher, "load_config", lambda: cfg)
    monkeypatch.setattr(
        launcher,
        "LiveKitWakeWordProvider",
        lambda **_kwargs: SimpleNamespace(available=True),
    )
    wrapped_audio = Mock()
    monkeypatch.setattr(launcher, "make_process_audio", Mock(return_value=wrapped_audio))
    configure_mpv = Mock()
    monkeypatch.setattr(launcher, "_configure_mpv", configure_mpv)
    monkeypatch.setenv("PREFERENCES_FILE", "/tmp/sayso-preferences.json")

    launcher.main()

    assert "--wakeup-sound" in sys.argv
    assert "--listen-during-wake-sound" not in sys.argv
    preferences_index = sys.argv.index("--preferences-file")
    assert sys.argv[preferences_index + 1] == "/tmp/sayso-preferences.json"
    assert upstream.process_audio is wrapped_audio  # type: ignore[attr-defined]
    configure_mpv.assert_called_once_with()
    upstream.run.assert_called_once_with()  # type: ignore[attr-defined]


def test_configure_mpv_uses_pulse_and_completes_error_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mpv_instance: dict[str, object] = {}
    mpv_constructor = Mock(return_value=mpv_instance)
    fake_mpv = SimpleNamespace(MPV=mpv_constructor)

    class FakeLibMpvPlayer:
        def __init__(self, device: str | None = None) -> None:
            self._mpv = fake_mpv.MPV(cache="yes")
            if device:
                self._mpv["audio-device"] = device
            self._done_callback = None

        def _on_end_file(self, event) -> None:
            if event.data.reason != 0:
                return
            callback = self._done_callback
            self._done_callback = None
            if callback:
                callback()

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

    launcher = importlib.import_module("satellite.sayso.launcher")
    launcher._configure_mpv()

    player = FakeLibMpvPlayer(device="pulse/speaker")
    mpv_constructor.assert_called_once_with(cache="yes", ao="pulse")
    assert mpv_instance["audio-device"] == "pulse/speaker"

    pipeline = SimpleNamespace(active=True)
    completed = Mock(side_effect=lambda: setattr(pipeline, "active", False))
    player._done_callback = completed
    player._on_end_file(SimpleNamespace(data=SimpleNamespace(reason=2)))
    assert pipeline.active
    player._on_end_file(SimpleNamespace(data=SimpleNamespace(reason=4)))
    player._on_end_file(SimpleNamespace(data=SimpleNamespace(reason=4)))

    assert not pipeline.active
    completed.assert_called_once_with()
