"""Launch upstream Linux Voice Assistant with the SaySo overlay."""

from __future__ import annotations

import logging
import os
import sys
from types import SimpleNamespace

from .config import load_config
from .process_audio import make_process_audio
from .wake.livekit import LiveKitWakeWordProvider

_LOGGER = logging.getLogger(__name__)


def _configure_mpv() -> None:
    """Use PulseAudio and recover the pinned player from playback errors."""
    from linux_voice_assistant.player import libmpv

    original_mpv = libmpv.mpv.MPV
    original_end_file = libmpv.LibMpvPlayer._on_end_file

    def pulse_mpv(*args, **kwargs):
        kwargs.setdefault("ao", "pulse")
        return original_mpv(*args, **kwargs)

    def end_file(player, event) -> None:
        reason = getattr(getattr(event, "data", None), "reason", -1)
        if reason == 4:
            event = SimpleNamespace(data=SimpleNamespace(reason=0))
        original_end_file(player, event)

    libmpv.mpv.MPV = pulse_mpv
    libmpv.LibMpvPlayer._on_end_file = end_file


def main() -> None:
    cfg = load_config()
    if cfg.wake_word.provider != "livekit":
        raise SystemExit("SaySo satellite is configured to use only the livekit wake provider")

    argv = [
        "linux-voice-assistant",
        "--name",
        cfg.satellite.name,
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
        "--wakeup-sound",
        str(cfg.sounds.wake),
        "--continue-conversation-delay",
        str(cfg.wake_word.post_tts_cooldown_ms / 1000.0),
        "--disable-peripheral-api",
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

    os.environ["SAYSO_STABLE_NAME"] = cfg.satellite.name
    import linux_voice_assistant.__main__ as lva_main

    lva_main.process_audio = make_process_audio(lva_main.process_audio, provider)
    _configure_mpv()
    lva_main.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
