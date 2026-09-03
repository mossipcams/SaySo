"""SaySo voice pipeline hooks: silent wake and post-STT acknowledgement sounds."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Type

from .config import SoundsCfg
from .wake.hook import SaySoExternalWakeHook

_LOGGER = logging.getLogger(__name__)


def _chain_chime_play(player: Any, path: str, done_callback: Callable[[], None] | None) -> None:
    """Play on tts_player without overwriting an in-flight MpvMediaPlayer callback."""
    existing = getattr(player, "_done_callback", None)
    if existing is not None:

        def chained() -> None:
            existing()
            player.play(str(path), done_callback=done_callback)

        player._done_callback = chained
        return
    player.play(str(path), done_callback=done_callback)


def install_voice_handlers(
    protocol: Type[Any],
    sounds: SoundsCfg,
    wake_hook: SaySoExternalWakeHook | None = None,
) -> None:
    """Patch LVA satellite hooks for silent wake and post-STT sounds."""
    from aioesphomeapi.model import VoiceAssistantEventType
    from linux_voice_assistant.events import LVAEvent

    original_handle_voice_event: Callable[..., None] = protocol.handle_voice_event
    original_tts_finished: Callable[..., None] = protocol._tts_finished
    original_stop: Callable[..., None] = protocol.stop

    def wakeup(self, wake_word) -> None:
        if self.state.muted:
            return
        if self._pipeline_active:
            _LOGGER.debug("Ignoring wake word - pipeline already active")
            return

        wake_word_phrase = wake_word.wake_word  # type: ignore[union-attr]
        _LOGGER.debug("Detected wake word: %s", wake_word_phrase)

        if wake_hook is not None:
            wake_hook.suspend()

        self._timer_finished = False
        self._timer_ring_start = None
        self._pipeline_active = True
        self._emit(LVAEvent.WAKE_WORD_DETECTED)
        self.duck()
        self._start_audio_streaming(wake_word_phrase)

    def handle_voice_event(
        self,
        event_type: VoiceAssistantEventType,
        data: Dict[str, str],
    ) -> None:
        original_handle_voice_event(self, event_type, data)

        if event_type == VoiceAssistantEventType.VOICE_ASSISTANT_STT_END:
            if data.get("text", "").strip():
                _LOGGER.debug("Playing acknowledgement sound after successful STT")
                _chain_chime_play(self.state.tts_player, str(sounds.wake), None)
            else:
                _LOGGER.debug("Playing failure sound after empty STT transcript")
                self._chime_rearm_pending = True

                def _failure_chime_done() -> None:
                    self._chime_rearm_pending = False
                    if wake_hook is not None:
                        wake_hook.rearm()

                _chain_chime_play(
                    self.state.tts_player,
                    str(sounds.failure),
                    _failure_chime_done,
                )
        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_ERROR:
            _LOGGER.debug("Playing failure sound after voice pipeline error")
            self._chime_rearm_pending = True

            def _error_chime_done() -> None:
                self._chime_rearm_pending = False
                if wake_hook is not None:
                    wake_hook.rearm()

            _chain_chime_play(
                self.state.tts_player,
                str(sounds.failure),
                _error_chime_done,
            )

    def _tts_finished(self) -> None:
        original_tts_finished(self)
        if getattr(self, "_chime_rearm_pending", False):
            return
        if wake_hook is not None:
            wake_hook.rearm()

    def stop(self) -> None:
        original_stop(self)
        if wake_hook is not None:
            wake_hook.rearm()

    protocol.wakeup = wakeup  # type: ignore[method-assign]
    protocol.handle_voice_event = handle_voice_event  # type: ignore[method-assign]
    protocol._tts_finished = _tts_finished  # type: ignore[method-assign]
    protocol.stop = stop  # type: ignore[method-assign]
