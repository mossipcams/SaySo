"""LVA external wake-provider hook for processed PCM."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from .buffer import WakeAudioBuffer, WakePrerollLookback
from .detection import Detection
from .livekit import HOP_SAMPLES, SAMPLE_RATE, WINDOW_SAMPLES, LiveKitWakeWordProvider
from .worker import WakeInferenceWorker

_LOGGER = logging.getLogger(__name__)

# ponytail: silence-prefill rearm (~HOP_SAMPLES to next predict) beats clean-window
# accumulation (~WINDOW_SAMPLES) after TTS; benchmarked in test_buffer.test_rearm_latency.


class _WakePhrase:
    def __init__(self, phrase: str) -> None:
        self.wake_word = phrase
        self.id = "sayso"


class SaySoExternalWakeHook:
    """Receive post-WebRTC PCM from LVA and run LiveKit inference off-thread."""

    def __init__(
        self,
        provider: LiveKitWakeWordProvider,
        *,
        preroll_ms: int = 0,
        wake_skip_ms: int = 500,
    ) -> None:
        self._provider = provider
        self._buffer = WakeAudioBuffer(WINDOW_SAMPLES, HOP_SAMPLES)
        self._lookback = WakePrerollLookback(preroll_ms, SAMPLE_RATE)
        self._wake_skip_ms = wake_skip_ms
        self._worker = WakeInferenceWorker(provider.predict_window)
        self._suspended = False
        self._get_satellite: Callable[[], Any] | None = None

    def bind_satellite(self, getter: Callable[[], Any]) -> None:
        self._get_satellite = getter

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
        self._lookback.clear()
        self._provider.reset()
        self.resume()

    def flush_preroll(self, satellite: Any) -> bytes:
        """Send post-wake_skip_ms lookback to the satellite STT path."""
        pcm = self._lookback.flush_bytes(self._wake_skip_ms)
        if pcm and satellite is not None and hasattr(satellite, "handle_audio"):
            satellite.handle_audio(pcm, None)
        return pcm

    def feed_pcm(self, state: Any, pcm_s16le: bytes) -> None:
        if state is not None and self._get_satellite is None:
            self.bind_satellite(lambda: getattr(state, "satellite", None))
        if not self._suspended and pcm_s16le:
            self._lookback.feed(pcm_s16le)
        if self._suspended or not pcm_s16le:
            return
        if self._buffer.feed(pcm_s16le):
            self._worker.submit(self._buffer.window())

    def _on_detection(self, detection: Detection) -> None:
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
