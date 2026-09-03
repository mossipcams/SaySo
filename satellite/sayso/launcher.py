"""Launch upstream Linux Voice Assistant with the SaySo overlay."""

from __future__ import annotations

import logging
import os
import sys

from .config import load_config
from .events import install_voice_handlers
from .playback import configure_pulse_mpv, install_playback_recovery
from .process_audio import install_wake_audio_path
from .wake.hook import SaySoExternalWakeHook
from .wake.livekit import LiveKitWakeWordProvider

_LOGGER = logging.getLogger(__name__)


def _configure_mpv() -> None:
    """Configure PulseAudio output and explicit mpv playback recovery."""
    configure_pulse_mpv()
    install_playback_recovery()


def main() -> None:
    cfg = load_config()
    if cfg.wake_word.provider != "livekit":
        raise SystemExit("SaySo satellite is configured to use only the livekit wake provider")

    argv = [
        "linux-voice-assistant",
        "--name",
        cfg.satellite.name,
        "--device-name",
        cfg.satellite.device_name,
        "--port",
        str(cfg.home_assistant.port),
        "--audio-input-device",
        cfg.audio.input_device,
        "--audio-output-device",
        cfg.audio.output_device,
        "--audio-input-channels",
        str(cfg.audio.channels),
        "--mic-noise-suppression",
        str(cfg.audio.noise_suppression),
        "--mic-auto-gain",
        str(cfg.audio.auto_gain),
        "--refractory-seconds",
        str(cfg.wake_word.refractory_seconds),
        "--continue-conversation-delay",
        str(cfg.wake_word.post_tts_cooldown_ms / 1000.0),
        "--disable-peripheral-api",
        "--disable-built-in-wake-word",
    ]
    if os.environ.get("SAYSO_DEBUG") == "1":
        argv.append("--debug")
    if preferences_file := os.environ.get("PREFERENCES_FILE"):
        argv.extend(("--preferences-file", preferences_file))

    sys.argv = argv

    provider = LiveKitWakeWordProvider(
        model_path=cfg.wake_word.model,
        phrase=cfg.wake_word.phrase,
        threshold=cfg.wake_word.threshold,
        refractory_seconds=cfg.wake_word.refractory_seconds,
    )
    if not provider.available:
        raise SystemExit(
            "Wake detection is not operational. "
            f"Model path: {cfg.wake_word.model}. "
            "Run: sayso-satellite test-wake-word"
        )

    import linux_voice_assistant.__main__ as lva_main
    from linux_voice_assistant.satellite import VoiceSatelliteProtocol

    wake_hook = SaySoExternalWakeHook(
        provider,
        preroll_ms=cfg.wake_word.preroll_ms,
        wake_skip_ms=cfg.wake_word.wake_skip_ms,
    )
    install_wake_audio_path(lva_main, wake_hook)
    install_voice_handlers(VoiceSatelliteProtocol, cfg.sounds, wake_hook)
    _configure_mpv()
    lva_main.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
