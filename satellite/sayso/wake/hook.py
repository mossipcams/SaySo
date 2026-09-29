
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from .buffer import PrerollFlush, WakeAudioBuffer
from .capture import WakeCaptureRing
from .detection import Detection
from .livekit import HOP_SAMPLES, SAMPLE_RATE, WINDOW_SAMPLES
from .mining import HardNegativeMiner
from .provider import WakeWordProvider
from .worker import WakeInferenceWorker

_LOGGER = logging.getLogger(__name__)


_RING_HEADROOM_SAMPLES = SAMPLE_RATE * 5

DEFAULT_WAKE_SKIP_MS = 120


class _WakePhrase:
    def __init__(self, phrase: str) -> None:
        self.wake_word = phrase
        self.id = "sayso"


class SaySoExternalWakeHook:

    def __init__(
        self,
        provider: WakeWordProvider,
        *,
        preroll_ms: int = 0,
        wake_skip_ms: int = DEFAULT_WAKE_SKIP_MS,
        capture_ring: WakeCaptureRing | None = None,
        miner: HardNegativeMiner | None = None,
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
        self._miner = miner
        if self._miner is not None:
            self._miner.bind_ring(self._ring)
        self._preroll_ms = int(preroll_ms)
        self._wake_skip_ms = wake_skip_ms
        self._worker = WakeInferenceWorker(provider.predict_window)
        self._suspended = False
        self._get_satellite: Callable[[], Any] | None = None
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
        self._buffer.rearm_with_silence()
        if self._miner is not None:
            trigger = self._detection_index if self._detection_index is not None else self._ring.end_index
            self._miner.snapshot_pre_trigger(trigger, synthetic_padding=True)
            self._miner.note_rearm()
        self._ring.reset()
        self._detection_index = None
        self._provider.reset()
        self.resume()

    def discard_detection(self) -> None:
        self._detection_index = None

    def flush_preroll(self, satellite: Any) -> PrerollFlush:
        detection_index = self._detection_index
        if detection_index is None:
            detection_index = self._ring.end_index
        skip_samples = max(0, self._wake_skip_ms) * SAMPLE_RATE // 1000
        trim_index = detection_index - skip_samples
        span_start = self._ring.available_span()[0]
        emitted_start = max(trim_index, self._ring.stt_cursor, span_start)
        origin = self._ring.origin
        underflow = origin <= trim_index < span_start
        pcm = self._ring.flush_from(trim_index)
        result = PrerollFlush(
            pcm=pcm,
            start_index=emitted_start,
            end_index=self._ring.end_index,
            underflow=underflow,
        )
        if result.underflow:
            _LOGGER.warning(
                "Preroll underflow: detection_index=%s trim_index=%s ring_span=%s",
                detection_index,
                trim_index,
                self._ring.available_span(),
            )
        elif trim_index < origin:
            _LOGGER.debug(
                "Preroll cold start: trim_index=%s predates epoch origin=%s",
                trim_index,
                origin,
            )
        if result.pcm and satellite is not None and hasattr(satellite, "handle_audio"):
            satellite.handle_audio(result.pcm, None)
        self._detection_index = None
        return result

    def feed_pcm(self, state: Any, pcm_s16le: bytes) -> None:
        if state is not None and self._get_satellite is None:
            self.bind_satellite(lambda: getattr(state, "satellite", None))
        if not pcm_s16le:
            return
        end_index = self._ring.append(pcm_s16le)
        if self._detection_index is None:
            self._forward_live(state)
        if self._suspended:
            return
        if self._buffer.feed(pcm_s16le):
            self._worker.submit(
                self._buffer.window(), end_index - self._buffer.pending_lag
            )

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
            if self._miner is not None:
                self._miner.snapshot_pre_trigger(detection.sample_index)
        capture_id = (
            self._miner.latest_detection_capture_id if self._miner is not None else None
        )
        satellite = self._get_satellite() if self._get_satellite else None
        if satellite is None or getattr(satellite, "_pipeline_active", False):
            if self._miner is not None and capture_id is not None:
                self._miner.publish_wake_outcome(
                    capture_id,
                    accepted=False,
                    suppressed=True,
                    reason="pipeline_active_or_unbound",
                )
            self.discard_detection()
            return
        if capture_id is not None:
            setattr(satellite, "_sayso_wake_capture_id", capture_id)
        try:
            satellite.wakeup(_WakePhrase(detection.phrase))
        except Exception:
            _LOGGER.exception("Unexpected error forwarding SaySo wake detection")


def install_external_wake_hook(_lva_main: Any, hook: SaySoExternalWakeHook) -> None:
    from linux_voice_assistant.external_wake import set_provider

    set_provider(hook.feed_pcm)
