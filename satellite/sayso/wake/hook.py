"""LVA external wake-provider hook for processed PCM."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from .buffer import PrerollFlush, WakeAudioBuffer
from .capture import WakeCaptureRing
from .detection import Detection
from .livekit import HOP_SAMPLES, SAMPLE_RATE, WINDOW_SAMPLES, LiveKitWakeWordProvider
from .worker import WakeInferenceWorker

_LOGGER = logging.getLogger(__name__)

# ponytail: silence-prefill rearm (~HOP_SAMPLES to next predict) beats clean-window
# accumulation (~WINDOW_SAMPLES) after TTS; benchmarked in test_buffer.test_rearm_latency.

# The capture ring also backs the STT handoff, so it must hold at least the
# wake window plus the preroll lookback plus one hop of slack.
_RING_HEADROOM_SAMPLES = SAMPLE_RATE * 5


class _WakePhrase:
    def __init__(self, phrase: str) -> None:
        self.wake_word = phrase
        self.id = "sayso"


class SaySoExternalWakeHook:
    """Receive post-resample 16 kHz PCM from LVA and run wake inference off-thread.

    One capture ring is the single source of truth for sample order. Wake
    inference reads a window from it, the detection carries the window's end
    index, and the wake→STT handoff drains ``[detection_index - skip, end)``
    under the ring lock. The live path then resumes at the ring's STT cursor, so
    each sample reaches Home Assistant exactly once and in order, no matter when
    the detection thread happens to run.
    """

    def __init__(
        self,
        provider: LiveKitWakeWordProvider,
        *,
        preroll_ms: int = 0,
        wake_skip_ms: int = 500,
        capture_ring: WakeCaptureRing | None = None,
    ) -> None:
        self._provider = provider
        self._buffer = WakeAudioBuffer(WINDOW_SAMPLES, HOP_SAMPLES)
        ring_capacity = (
            (preroll_ms * SAMPLE_RATE) // 1000
            + WINDOW_SAMPLES
            + HOP_SAMPLES
            + _RING_HEADROOM_SAMPLES
        )
        self._ring = capture_ring if capture_ring is not None else WakeCaptureRing(ring_capacity)
        self._preroll_ms = int(preroll_ms)
        self._wake_skip_ms = wake_skip_ms
        self._worker = WakeInferenceWorker(provider.predict_window)
        self._suspended = False
        self._get_satellite: Callable[[], Any] | None = None
        # Published by the worker the instant a detection fires, before the
        # satellite wakeup path runs, so the flush can read the exact boundary.
        self._detection_index: int | None = None

    def bind_satellite(self, getter: Callable[[], Any]) -> None:
        self._get_satellite = getter

    @property
    def last_detection_index(self) -> int | None:
        return self._detection_index

    def start(self) -> None:
        self._provider.start()
        self._worker.start(self._on_detection)

    def shutdown(self) -> None:
        self._worker.shutdown()
        self._provider.shutdown()

    def suspend(self) -> None:
        self._suspended = True
        self._provider.suspend()

    def resume(self) -> None:
        self._suspended = False
        self._provider.resume()

    def rearm(self) -> None:
        """One controlled reset after TTS; do not clear on every capture block."""
        self._buffer.rearm_with_silence()
        # Re-anchor rather than zero the index so a detection that arrived just
        # before rearm can never be confused with one from the next cycle.
        self._ring.reset()
        self._detection_index = None
        self._provider.reset()
        self.resume()

    def flush_preroll(self, satellite: Any) -> PrerollFlush:
        """Atomically hand ``[detection-skip, end)`` to the satellite STT path.

        The detection boundary is the sample index the classifier actually
        scored, so the trim does not depend on how long the wakeup path took to
        reach this call. Emitted samples are marked delivered by the ring's STT
        cursor, so the live path resumes after them instead of replaying them.
        """
        detection_index = self._detection_index
        if detection_index is None:
            detection_index = self._ring.end_index
        skip_samples = max(0, self._wake_skip_ms) * SAMPLE_RATE // 1000
        trim_index = detection_index - skip_samples
        span_start = self._ring.available_span()[0]
        # The flush emits from the later of the requested trim and what the live
        # path has already delivered, so report that as the true start.
        emitted_start = max(trim_index, self._ring.stt_cursor, span_start)
        pcm = self._ring.flush_from(trim_index)
        result = PrerollFlush(
            pcm=pcm,
            start_index=emitted_start,
            end_index=self._ring.end_index,
            underflow=not self._ring.covers(trim_index),
        )
        if result.underflow:
            _LOGGER.warning(
                "Preroll underflow: detection_index=%s trim_index=%s ring_span=%s",
                detection_index,
                trim_index,
                self._ring.available_span(),
            )
        if result.pcm and satellite is not None and hasattr(satellite, "handle_audio"):
            satellite.handle_audio(result.pcm, None)
        # The handoff is complete: from here the live path resumes, and the ring's
        # STT cursor guarantees it continues exactly where the flush stopped.
        self._detection_index = None
        return result

    def feed_pcm(self, state: Any, pcm_s16le: bytes) -> None:
        if state is not None and self._get_satellite is None:
            self.bind_satellite(lambda: getattr(state, "satellite", None))
        if not pcm_s16le:
            return
        end_index = self._ring.append(pcm_s16le)
        # Forward live audio to the STT path whenever no handoff is in flight.
        # Once a detection publishes its boundary, delivery from that boundary
        # belongs to flush_preroll; letting the live path also emit it would
        # duplicate the first command samples. After the flush this fires again
        # and the ring cursor continues exactly where the flush stopped.
        #
        # Note this is gated by the pending detection, NOT by _suspended:
        # _suspended only suppresses wake inference, while command audio must
        # keep streaming to Home Assistant for as long as the mic is open.
        if self._detection_index is None:
            self._forward_live(state)
        if self._suspended:
            return
        if self._buffer.feed(pcm_s16le):
            self._worker.submit(self._buffer.window(), end_index)

    def _forward_live(self, state: Any) -> None:
        satellite = state.satellite if state is not None else None
        if satellite is None or not getattr(satellite, "_is_streaming_audio", False):
            return
        pending = self._ring.drain_after_cursor()
        if pending:
            satellite.handle_audio(pending, None)

    def _on_detection(self, detection: Detection) -> None:
        if detection.sample_index is not None:
            self._detection_index = detection.sample_index
        satellite = self._get_satellite() if self._get_satellite else None
        if satellite is None or getattr(satellite, "_pipeline_active", False):
            return
        try:
            satellite.wakeup(_WakePhrase(detection.phrase))
        except Exception:
            _LOGGER.exception("Unexpected error forwarding SaySo wake detection")


def install_external_wake_hook(_lva_main: Any, hook: SaySoExternalWakeHook) -> None:
    """Register the hook with upstream LVA without wrapping microphone record()."""
    from linux_voice_assistant.external_wake import set_provider

    set_provider(hook.feed_pcm)
