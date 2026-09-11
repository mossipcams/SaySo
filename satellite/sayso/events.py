"""SaySo voice pipeline hooks: silent wake and post-STT acknowledgement sounds."""

from __future__ import annotations

import asyncio
import itertools
import logging
import threading
import time
from typing import Any, Callable, Dict, Type

from .config import SoundsCfg
from .tracing import SatelliteTracer
from .wake.hook import SaySoExternalWakeHook

_LOGGER = logging.getLogger(__name__)

# Monotonic counter so two commands captured in the same millisecond still get
# distinct run ids.
_capture_counter = itertools.count()


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


def _defer_until_playback_idle(player: Any, action: Callable[[], None]) -> None:
    """Run ``action`` once the player is idle, without stealing its callback.

    If the player is mid-track the action is chained behind the callback it
    already holds; if it is idle the action runs immediately. This is what keeps
    the microphone shut until response audio is genuinely finished instead of
    approximately finished.
    """
    if player is None or not getattr(player, "_done_callback", None):
        action()
        return
    existing = player._done_callback

    def _chained() -> None:
        existing()
        action()

    player._done_callback = _chained


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
    aec_gate_ms: float = 0.0,
    stt_capture: Any = None,
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
    original_handle_audio: Callable[..., None] = getattr(
        protocol, "handle_audio", lambda self, data, data2=None: None
    )
    trace = tracer if tracer is not None else SatelliteTracer()

    def _run_id() -> str:
        # A per-command id derived from the monotonic clock: unique enough for a
        # capture directory, and independent of the Home Assistant trace id.
        return f"cmd-{int(time.monotonic() * 1000)}-{next(_capture_counter)}"

    def _start_capture(self: Any, phrase: str) -> None:
        if stt_capture is None:
            return
        run_id = _run_id()
        self._sayso_capture_id = run_id
        stt_capture.begin_command(run_id)

    def _finish_capture(
        self: Any, *, transcript: str, failed_reason: str | None = None
    ) -> None:
        if stt_capture is None:
            return
        run_id = getattr(self, "_sayso_capture_id", None)
        if run_id is None:
            return
        self._sayso_capture_id = None
        stt_capture.end_command(
            run_id,
            transcript=transcript,
            underflow=bool(getattr(self, "_sayso_capture_underflow", False)),
            failed_reason=failed_reason,
        )

    def wakeup(self, wake_word) -> None:
        if self.state.muted:
            return
        if self._pipeline_active:
            _LOGGER.debug("Ignoring wake word - pipeline already active")
            return
        # Never open the microphone into an in-flight response: the speaker feed
        # would be captured as the first command audio and there is no AEC on
        # this path (webrtc-noise-gain exposes AGC/NS only).
        if getattr(self, "_tts_played", False):
            _LOGGER.debug("Ignoring wake word - response playback still in flight")
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

        def _open_microphone() -> None:
            # Guard the deferral: the pipeline can be torn down while the chime
            # and settle delay are still pending.
            if self.state.muted or not self._pipeline_active:
                # The wake was abandoned before the mic opened; release the
                # suspended hook so the next detection is not swallowed.
                if wake_hook is not None:
                    wake_hook.resume()
                return
            # Start the capture tap at the true command boundary so the WAV
            # begins exactly where command audio begins.
            _start_capture(self, wake_word_phrase)
            self._start_audio_streaming(wake_word_phrase)
            if wake_hook is not None:
                # Retained pre-open audio is handed over only now, at the true
                # detection boundary, so nothing captured during playback is
                # misattributed as command speech.
                result = wake_hook.flush_preroll(self)
                if result.underflow:
                    self._sayso_capture_underflow = True

        def _after_settle() -> None:
            # Hold a short settle delay so the speaker tail does not bleed into
            # the first command samples. There is no AEC on this path
            # (webrtc-noise-gain exposes AGC/NS only), so the delay is the
            # fail-safe against capturing our own output.
            if aec_gate_ms > 0:
                threading.Timer(aec_gate_ms / 1000.0, _open_microphone).start()
            else:
                _open_microphone()

        # Any in-flight playback (a previous response, a timer sound, ducked
        # music) must finish before the mic opens. Chaining through the player's
        # own done callback keeps ordering without overwriting a callback the
        # player already holds.
        player = getattr(self.state, "tts_player", None)
        if player is None:
            _after_settle()
        else:
            _defer_until_playback_idle(player, _after_settle)

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
            transcript = data.get("text", "")
            if transcript.strip():
                _LOGGER.debug("Playing acknowledgement sound after successful STT")
                _finish_capture(self, transcript=transcript)
                _schedule_chime_play(self.state.tts_player, str(sounds.wake), None)
            else:
                _LOGGER.debug("Playing failure sound after empty STT transcript")
                trace.fail("stt", "stt_failed")
                _finish_capture(self, transcript="", failed_reason="empty_transcript")
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
            _finish_capture(self, transcript="", failed_reason="pipeline_error")
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
        # Safety net: a command with no stt_end (aborted turn) must still be
        # written rather than left open and overwritten by the next wake.
        _finish_capture(self, transcript="", failed_reason="no_stt_end")
        if getattr(self, "_chime_rearm_pending", False):
            return
        if wake_hook is not None:
            wake_hook.rearm()

    def stop(self) -> None:
        original_stop(self)
        trace.fail("playback", "playback_interrupted")
        trace.finish()
        _finish_capture(self, transcript="", failed_reason="interrupted")
        if wake_hook is not None:
            wake_hook.rearm()

    def handle_audio(self, audio_chunk: bytes, audio_chunk_2: Any = None) -> None:
        # Tap the exact bytes on their way to Home Assistant. This is the only
        # place that can guarantee the captured WAV equals what Faster Whisper
        # receives, so recording happens before the send, not after.
        if stt_capture is not None and getattr(self, "_sayso_capture_id", None):
            stt_capture.tap(audio_chunk)
        original_handle_audio(self, audio_chunk, audio_chunk_2)

    protocol.wakeup = wakeup  # type: ignore[method-assign]
    protocol.handle_voice_event = handle_voice_event  # type: ignore[method-assign]
    protocol._tts_finished = _tts_finished  # type: ignore[method-assign]
    protocol.stop = stop  # type: ignore[method-assign]
    protocol.handle_audio = handle_audio  # type: ignore[method-assign]
