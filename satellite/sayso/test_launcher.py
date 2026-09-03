import importlib
import os
import sys
import textwrap
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

_PATCH_PATH = Path(__file__).resolve().parents[1] / "patches" / "0001-sayso-stable-device-name.patch"
_PATCH_0002_PATH = Path(__file__).resolve().parents[1] / "patches" / "0002-lva-external-wake-provider.patch"
_PATCH_0002_WAKE_HUNK = "@@ -760,44 +769,46 @@"


def _satellite_cfg(
    *,
    name: str = "Living Room",
    device_name: str = "sayso-living-room",
) -> SimpleNamespace:
    return SimpleNamespace(name=name, device_name=device_name)


def test_stable_device_name_patch_uses_cli_option_not_env() -> None:
    content = _PATCH_PATH.read_text(encoding="utf-8")
    assert "SAYSO_STABLE_NAME" not in content
    assert "--device-name" in content
    assert "args.device_name" in content


def test_patch_0002_disables_builtin_wake_only_not_stop_word() -> None:
    content = _PATCH_0002_PATH.read_text(encoding="utf-8")
    assert "if state.disable_builtin_wake_word:\n                    continue" not in content
    assert "if not state.disable_builtin_wake_word:" in content
    assert "state.satellite.stop()" not in content
    assert "# Always process to keep state correct" in content


def _patch_0002_process_audio_wake_hunk_post_lines() -> list[str]:
    """Post-patch lines for the process_audio wake/stop hunk in patch 0002."""

    text = _PATCH_0002_PATH.read_text(encoding="utf-8")
    start = text.index(_PATCH_0002_WAKE_HUNK) + len(_PATCH_0002_WAKE_HUNK)
    raw: list[str] = []
    for line in text[start:].splitlines():
        if line.startswith("@@") or line.startswith("--- "):
            break
        if not line:
            raw.append("")
            continue
        if line[0] in " +-":
            raw.append(line)

    post: list[str] = []
    for line in raw:
        if line.startswith("-"):
            continue
        if line.startswith((" ", "+")):
            post.append(line[1:])
    return post


def _process_audio_hunk_body_before_stop_tail(hunk_post_lines: list[str]) -> str:
    lines: list[str] = []
    in_try_body = False
    for line in hunk_post_lines:
        if "# Always process to keep state correct" in line:
            break
        stripped = line.strip()
        if not in_try_body:
            if stripped == "try:":
                in_try_body = True
            continue
        lines.append(line)
    return textwrap.dedent("\n".join(lines)).strip("\n")


_LVA_STOP_WORD_TAIL = textwrap.dedent(
    """
    stopped = False
    stop_word.probability_cutoff = state.stop_word_threshold
    for _micro_input in micro_inputs:
        if stop_word.process_streaming(_micro_input):
            stopped = True

    if stopped and (stop_word.id in state.active_wake_words) and not state.muted:
        state.satellite.stop()
    """
).strip("\n")


class _MicroWakeWord:
    pass


class _OpenWakeWord:
    pass


def _run_patched_wake_stop_control_flow(
    *,
    disable_builtin_wake_word: bool,
    wake_activated: bool,
    stop_detected: bool,
    muted: bool = False,
) -> tuple[Mock, Mock]:
    """Execute patch-0002 process_audio hunk + unchanged LVA stop-word tail once."""

    hunk_body = _process_audio_hunk_body_before_stop_tail(
        _patch_0002_process_audio_wake_hunk_post_lines()
    )
    loop_source = (
        "for _sayso_loop_once in (None,):\n"
        + textwrap.indent(hunk_body, "    ")
        + "\n"
        + textwrap.indent(_LVA_STOP_WORD_TAIL, "    ")
    )

    satellite = Mock()
    stop_word = Mock()
    stop_word.id = "stop"
    stop_word.process_streaming.side_effect = [stop_detected]
    state = SimpleNamespace(
        disable_builtin_wake_word=disable_builtin_wake_word,
        muted=muted,
        stop_word=stop_word,
        stop_word_threshold=0.5,
        wake_word_1_threshold=0.5,
        wake_word_2_threshold=0.5,
        active_wake_words=["stop"],
        refractory_seconds=0.0,
        satellite=satellite,
    )
    wake_word = _MicroWakeWord()
    wake_word.process_streaming = Mock(return_value=wake_activated)
    wake_word.debug_probabilities = False
    wake_word.probability_cutoff = 0.0

    micro_features = Mock()
    micro_features.process_streaming.return_value = [object()]
    micro_inputs: list[object] = []
    last_active = None

    namespace = {
        "state": state,
        "channel_chunks": [b"", b""],
        "n_channels": 1,
        "audio_chunk": b"chunk",
        "micro_features": micro_features,
        "micro_inputs": micro_inputs,
        "has_oww": False,
        "oww_features": None,
        "oww_inputs": [],
        "wake_words": [wake_word],
        "last_active": last_active,
        "time": time,
        "MicroWakeWord": _MicroWakeWord,
        "OpenWakeWord": _OpenWakeWord,
        "_LOGGER": Mock(),
        "stop_word": stop_word,
    }
    exec(compile(loop_source, "<patch-0002-process_audio-hunk>", "exec"), namespace)

    return satellite.wakeup, satellite.stop


def test_patch_0002_stop_word_still_stops_when_builtin_wake_disabled() -> None:
    wakeup, stop = _run_patched_wake_stop_control_flow(
        disable_builtin_wake_word=True,
        wake_activated=True,
        stop_detected=True,
    )

    wakeup.assert_not_called()
    stop.assert_called_once_with()


def test_launcher_passes_device_name_separate_from_friendly_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = ModuleType("linux_voice_assistant")
    package.__path__ = []  # type: ignore[attr-defined]
    upstream = ModuleType("linux_voice_assistant.__main__")
    upstream.run = Mock()  # type: ignore[attr-defined]
    satellite_module = ModuleType("linux_voice_assistant.satellite")
    satellite_module.VoiceSatelliteProtocol = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "linux_voice_assistant", package)
    monkeypatch.setitem(sys.modules, "linux_voice_assistant.__main__", upstream)
    monkeypatch.setitem(sys.modules, "linux_voice_assistant.satellite", satellite_module)
    monkeypatch.delenv("SAYSO_STABLE_NAME", raising=False)

    launcher = importlib.import_module("satellite.sayso.launcher")
    cfg = SimpleNamespace(
        satellite=_satellite_cfg(),
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

    device_index = sys.argv.index("--device-name")
    name_index = sys.argv.index("--name")
    assert sys.argv[device_index + 1] == "sayso-living-room"
    assert sys.argv[name_index + 1] == "Living Room"
    assert "SAYSO_STABLE_NAME" not in os.environ
    upstream.run.assert_called_once_with()  # type: ignore[attr-defined]


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
        satellite=_satellite_cfg(),
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
    satellite_module = ModuleType("linux_voice_assistant.satellite")
    satellite_module.VoiceSatelliteProtocol = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "linux_voice_assistant", package)
    monkeypatch.setitem(sys.modules, "linux_voice_assistant.__main__", upstream)
    monkeypatch.setitem(sys.modules, "linux_voice_assistant.satellite", satellite_module)

    launcher = importlib.import_module("satellite.sayso.launcher")
    cfg = SimpleNamespace(
        satellite=_satellite_cfg(),
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
        sounds=SimpleNamespace(wake="ack.wav", failure="failure.wav", unavailable="unavailable.wav"),
    )
    monkeypatch.setattr(launcher, "load_config", lambda: cfg)
    monkeypatch.setattr(
        launcher,
        "LiveKitWakeWordProvider",
        lambda **_kwargs: SimpleNamespace(available=True, predict_window=Mock(return_value=None)),
    )
    install_wake = Mock()
    monkeypatch.setattr(launcher, "install_wake_audio_path", install_wake)
    install_handlers = Mock()
    monkeypatch.setattr(launcher, "install_voice_handlers", install_handlers)
    configure_mpv = Mock()
    monkeypatch.setattr(launcher, "_configure_mpv", configure_mpv)
    monkeypatch.setenv("PREFERENCES_FILE", "/tmp/sayso-preferences.json")

    launcher.main()

    assert "--disable-built-in-wake-word" in sys.argv
    assert "--wakeup-sound" not in sys.argv
    assert "--listen-during-wake-sound" not in sys.argv
    preferences_index = sys.argv.index("--preferences-file")
    assert sys.argv[preferences_index + 1] == "/tmp/sayso-preferences.json"
    launcher.install_wake_audio_path.assert_called_once()
    install_handlers.assert_called_once()
    install_wake.assert_called_once()
    call_args = install_handlers.call_args.args
    assert call_args[0] is satellite_module.VoiceSatelliteProtocol
    assert call_args[1] is cfg.sounds
    assert isinstance(call_args[2], launcher.SaySoExternalWakeHook)
    configure_mpv.assert_called_once_with()
    upstream.run.assert_called_once_with()  # type: ignore[attr-defined]


def test_configure_mpv_uses_pulse_and_recovers_from_playback_errors(
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

    from satellite.sayso.playback import END_FILE_ABORT, END_FILE_ERROR

    launcher = importlib.import_module("satellite.sayso.launcher")
    launcher._configure_mpv()

    player = FakeLibMpvPlayer(device="pulse/speaker")
    mpv_constructor.assert_called_once_with(cache="yes", ao="pulse")
    assert mpv_instance["audio-device"] == "pulse/speaker"

    pipeline = SimpleNamespace(active=True)
    completed = Mock(side_effect=lambda: setattr(pipeline, "active", False))
    player._done_callback = completed
    player._on_end_file(SimpleNamespace(data=SimpleNamespace(reason=END_FILE_ABORT)))
    assert pipeline.active
    player._on_end_file(SimpleNamespace(data=SimpleNamespace(reason=END_FILE_ERROR)))

    assert not pipeline.active
    completed.assert_called_once_with()
