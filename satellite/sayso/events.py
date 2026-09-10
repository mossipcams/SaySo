"""SaySo voice pipeline hooks: silent wake and post-STT acknowledgement sounds."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, Type

from .config import SoundsCfg
from .tracing import SatelliteTracer
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


def _schedule_chime_play(
    player: Any,
    path: str,
    done_callback: Callable[[], None] | None,
) -> None:
    """Defer chime play until after the ESPHome packet handler returns."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _chain_chime_play(player, path, done_callback)
        return
    loop.call_soon(_chain_chime_play, player, path, done_callback)


def _event_type(source: Any, name: str) -> Any:
    """Return one voice event member, or a value that can never match it.

    Older aioesphomeapi builds do not define every member SaySo observes; a
    unique sentinel keeps the comparison false instead of raising.
    """
    return getattr(source, name, object())


def install_voice_handlers(
    protocol: Type[Any],
    sounds: SoundsCfg,
    wake_hook: SaySoExternalWakeHook | None = None,
    tracer: SatelliteTracer | None = None,
) -> None:
    """Patch LVA satellite hooks for silent wake, post-STT sounds and tracing."""
    from aioesphomeapi.model import VoiceAssistantEventType
    from linux_voice_assistant.events import LVAEvent

    event_stt_vad_end = _event_type(
        VoiceAssistantEventType, "VOICE_ASSISTANT_STT_VAD_END"
    )
    event_stt_end = _event_type(VoiceAssistantEventType, "VOICE_ASSISTANT_STT_END")
    event_intent_end = _event_type(
        VoiceAssistantEventType, "VOICE_ASSISTANT_INTENT_END"
    )
    event_intent_progress = _event_type(
        VoiceAssistantEventType, "VOICE_ASSISTANT_INTENT_PROGRESS"
    )
    event_tts_end = _event_type(VoiceAssistantEventType, "VOICE_ASSISTANT_TTS_END")
    event_error = _event_type(VoiceAssistantEventType, "VOICE_ASSISTANT_ERROR")

    original_handle_voice_event: Callable[..., None] = protocol.handle_voice_event
    original_tts_finished: Callable[..., None] = protocol._tts_finished
    original_stop: Callable[..., None] = protocol.stop
    trace = tracer if tracer is not None else SatelliteTracer()

    def wakeup(self, wake_word) -> None:
        if self.state.muted:
            return
        if self._pipeline_active:
            _LOGGER.debug("Ignoring wake word - pipeline already active")
            return

        wake_word_phrase = wake_word.wake_word  # type: ignore[union-attr]
        _LOGGER.debug("Detected wake word: %s", wake_word_phrase)
        trace.wake(wake_word_phrase)

        if wake_hook is not None:
            wake_hook.suspend()

        self._timer_finished = False
        self._timer_ring_start = None
        self._pipeline_active = True
        self._emit(LVAEvent.WAKE_WORD_DETECTED)
        self.duck()
        trace.upload_started()
        self._start_audio_streaming(wake_word_phrase)
        if wake_hook is not None:
            wake_hook.flush_preroll(self)

    def handle_voice_event(
        self,
        event_type: VoiceAssistantEventType,
        data: Dict[str, str],
    ) -> None:
        original_handle_voice_event(self, event_type, data)

        if event_type in (event_stt_vad_end, event_stt_end):
            # Home Assistant stopped consuming command audio; the transcription
            # itself happens there and is timed there.
            trace.upload_completed()

        if event_type == event_intent_end:
            trace.set_conversation_id(data.get("conversation_id"))
        elif event_type in (event_tts_end, event_intent_progress):
            if getattr(self, "_tts_played", False):
                trace.playback_started()

        if event_type == event_stt_end:
            if data.get("text", "").strip():
                _LOGGER.debug("Playing acknowledgement sound after successful STT")
                _schedule_chime_play(self.state.tts_player, str(sounds.wake), None)
            else:
                _LOGGER.debug("Playing failure sound after empty STT transcript")
                trace.fail("stt", "stt_failed")
                self._chime_rearm_pending = True

                def _failure_chime_done() -> None:
                    self._chime_rearm_pending = False
                    if wake_hook is not None:
                        wake_hook.rearm()

                _schedule_chime_play(
                    self.state.tts_player,
                    str(sounds.failure),
                    _failure_chime_done,
                )
        elif event_type == event_error:
            _LOGGER.debug("Playing failure sound after voice pipeline error")
            trace.fail("pipeline", "pipeline_error")
            self._chime_rearm_pending = True

            def _error_chime_done() -> None:
                self._chime_rearm_pending = False
                if wake_hook is not None:
                    wake_hook.rearm()

            _schedule_chime_play(
                self.state.tts_player,
                str(sounds.failure),
                _error_chime_done,
            )

    def _tts_finished(self) -> None:
        original_tts_finished(self)
        trace.playback_completed()
        trace.finish()
        if getattr(self, "_chime_rearm_pending", False):
            return
        if wake_hook is not None:
            wake_hook.rearm()

    def stop(self) -> None:
        original_stop(self)
        trace.fail("playback", "playback_interrupted")
        trace.finish()
        if wake_hook is not None:
            wake_hook.rearm()

    protocol.wakeup = wakeup  # type: ignore[method-assign]
    protocol.handle_voice_event = handle_voice_event  # type: ignore[method-assign]
    protocol._tts_finished = _tts_finished  # type: ignore[method-assign]
    protocol.stop = stop  # type: ignore[method-assign]
