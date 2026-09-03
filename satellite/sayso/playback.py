"""Explicit mpv playback handling for the SaySo satellite overlay."""

from __future__ import annotations

import logging
import threading
from enum import Enum
from types import SimpleNamespace
from typing import Callable

_LOGGER = logging.getLogger(__name__)

END_FILE_EOF = 0
END_FILE_STOP = 1
END_FILE_ABORT = 2
END_FILE_QUIT = 3
END_FILE_ERROR = 4


class PlaybackOutcome(Enum):
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


def classify_end_file_reason(reason: int) -> PlaybackOutcome:
    if reason == END_FILE_EOF:
        return PlaybackOutcome.COMPLETED
    if reason in (END_FILE_STOP, END_FILE_ABORT, END_FILE_QUIT):
        return PlaybackOutcome.INTERRUPTED
    return PlaybackOutcome.FAILED


def configure_pulse_mpv() -> None:
    """Use PulseAudio for mpv without rewriting end-file semantics."""
    from linux_voice_assistant.player import libmpv

    original_mpv = libmpv.mpv.MPV

    class PulseMPV(original_mpv):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("ao", "pulse")
            super().__init__(*args, **kwargs)

    libmpv.mpv.MPV = PulseMPV


def install_playback_recovery() -> None:
    """Recover pipeline state on real mpv errors without masking them as success."""
    from linux_voice_assistant.player import libmpv

    original_end_file: Callable[..., None] = libmpv.LibMpvPlayer._on_end_file

    def end_file(player, event) -> None:
        reason = getattr(getattr(event, "data", None), "reason", -1)
        outcome = classify_end_file_reason(reason)
        if outcome is PlaybackOutcome.COMPLETED:
            original_end_file(player, event)
            return
        if outcome is PlaybackOutcome.INTERRUPTED:
            _LOGGER.debug("Ignoring interrupted mpv end-file event (reason=%s)", reason)
            return
        _LOGGER.warning("mpv playback failed (reason=%s); running cleanup callback", reason)
        original_end_file(
            player,
            SimpleNamespace(data=SimpleNamespace(reason=END_FILE_EOF)),
        )

    libmpv.LibMpvPlayer._on_end_file = end_file


def play_sound(path: str, device: str, timeout: float = 15.0) -> int:
    """Play a local file and return 0 on success, 1 on failure or timeout."""
    configure_pulse_mpv()
    install_playback_recovery()

    try:
        from linux_voice_assistant.mpv_player import MpvMediaPlayer
        from linux_voice_assistant.player.libmpv import LibMpvPlayer

        done = threading.Event()
        failed = threading.Event()
        original_end_file = LibMpvPlayer._on_end_file

        def observe_end_file(player, event) -> None:
            reason = getattr(getattr(event, "data", None), "reason", -1)
            if classify_end_file_reason(reason) is PlaybackOutcome.FAILED:
                failed.set()
            original_end_file(player, event)

        LibMpvPlayer._on_end_file = observe_end_file
        try:
            player = MpvMediaPlayer(device=device)
        finally:
            LibMpvPlayer._on_end_file = original_end_file
        player.play(str(path), done_callback=done.set)
    except Exception:
        _LOGGER.exception("Audio playback failed")
        return 1

    if not done.wait(timeout):
        _LOGGER.error("Audio playback timed out after %.1f seconds", timeout)
        try:
            player.stop()
        except Exception:
            _LOGGER.exception("Could not stop timed-out audio playback")
        return 1
    return int(failed.is_set())
