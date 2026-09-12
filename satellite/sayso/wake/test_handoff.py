"""Sample-ordered, exactly-once wake→STT handoff across the capture ring."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from sayso.wake.hook import SaySoExternalWakeHook
from sayso.wake.livekit import HOP_SAMPLES, WINDOW_SAMPLES


class _RecordingSatellite:
    """Minimal satellite that records every byte handed to handle_audio."""

    def __init__(self) -> None:
        self._is_streaming_audio = True
        self._pipeline_active = False
        self.chunks: list[bytes] = []
        self.wakeups: list[object] = []

    def handle_audio(self, pcm: bytes, ref: bytes | None = None) -> None:
        self.chunks.append(pcm)

    def wakeup(self, wake_word) -> None:
        self.wakeups.append(wake_word)

    def delivered(self) -> np.ndarray:
        if not self.chunks:
            return np.zeros(0, dtype="<i2")
        return np.frombuffer(b"".join(self.chunks), dtype="<i2")


def _hook_with_detection(phrase: str = "SaySo") -> SaySoExternalWakeHook:
    provider = MagicMock(available=True)
    provider.predict_window.return_value = None
    hook = SaySoExternalWakeHook(provider, preroll_ms=1000, wake_skip_ms=500)
    return hook


def test_every_captured_sample_is_delivered_to_stt_exactly_once() -> None:
    hook = _hook_with_detection()
    satellite = _RecordingSatellite()
    state = SimpleNamespace(satellite=satellite)
    hook.bind_satellite(lambda: satellite)
    hook.start()
    try:
        # Two seconds of a strictly increasing ramp so order is observable.
        total = WINDOW_SAMPLES * 2
        ramp = np.arange(total, dtype="<i2")
        chunk = 512
        for offset in range(0, total, chunk):
            hook.feed_pcm(state, ramp[offset : offset + chunk].tobytes())
    finally:
        hook.shutdown()

    delivered = satellite.delivered()
    assert delivered.size == total
    # Exactly once, in order: the delivered stream is the captured ramp.
    assert np.array_equal(delivered, ramp)


def test_preroll_flush_does_not_duplicate_live_samples() -> None:
    hook = _hook_with_detection()
    satellite = _RecordingSatellite()
    state = SimpleNamespace(satellite=satellite)
    hook.bind_satellite(lambda: satellite)
    hook.start()
    try:
        total = WINDOW_SAMPLES
        ramp = np.arange(total, dtype="<i2")
        chunk = 512
        for offset in range(0, total, chunk):
            hook.feed_pcm(state, ramp[offset : offset + chunk].tobytes())

        # Simulate a detection landing on the last captured sample.
        hook._detection_index = total - HOP_SAMPLES
        before = satellite.delivered().size

        # The live path already delivered the whole stream, so the flush has
        # nothing left to emit from the preroll window: it must not reappear.
        hook._suspended = True
        result = hook.flush_preroll(satellite)
        assert result.pcm == b""
        assert satellite.delivered().size == before

        # And the live path must not replay it on the next block either.
        hook._suspended = False
        hook.feed_pcm(state, np.full(chunk, 1234, dtype="<i2").tobytes())
        delivered = satellite.delivered()
        assert np.all(delivered[before:] == 1234)
    finally:
        hook.shutdown()


def test_flush_preroll_uses_detection_index_not_flush_time() -> None:
    """The emitted trim start is stable regardless of when the flush runs."""

    def _prepared_hook() -> tuple[SaySoExternalWakeHook, _RecordingSatellite]:
        provider = MagicMock(available=True)
        provider.predict_window.return_value = None
        hook = SaySoExternalWakeHook(provider, preroll_ms=2000, wake_skip_ms=500)
        satellite = _RecordingSatellite()
        hook.bind_satellite(lambda: satellite)
        # Append straight to the ring so nothing is delivered live first; this
        # isolates the trim boundary from the exactly-once cursor.
        hook._ring.append(np.zeros(32000, dtype="<i2").tobytes())
        hook._detection_index = 32000
        return hook, satellite

    early_hook, early_sat = _prepared_hook()
    early = early_hook.flush_preroll(early_sat)

    late_hook, late_sat = _prepared_hook()
    # 40 ms of extra capture before the flush runs.
    late_hook._ring.append(np.full(640, 5, dtype="<i2").tobytes())
    late = late_hook.flush_preroll(late_sat)

    # Same detection boundary, same trim start, despite the flush delay.
    assert early.start_index == late.start_index == 32000 - 8000
    assert np.frombuffer(early.pcm, dtype="<i2").size == 8000
    # The extra latency audio is retained, not dropped.
    assert np.frombuffer(late.pcm, dtype="<i2").size == 8640


def test_rearm_reanchors_timeline_without_zeroing_index() -> None:
    hook = _hook_with_detection()
    satellite = _RecordingSatellite()
    hook.bind_satellite(lambda: satellite)
    hook.feed_pcm(SimpleNamespace(satellite=satellite), np.ones(1000, dtype="<i2").tobytes())
    before = hook._ring.end_index
    hook.rearm()
    assert hook._ring.end_index == before
    assert hook.last_detection_index is None


def test_command_audio_keeps_streaming_after_the_flush() -> None:
    """The handoff must not permanently silence live STT forwarding."""
    hook = _hook_with_detection()
    satellite = _RecordingSatellite()
    state = SimpleNamespace(satellite=satellite)
    hook.bind_satellite(lambda: satellite)
    hook.start()
    try:
        total = WINDOW_SAMPLES
        ramp = np.arange(total, dtype="<i2")
        for offset in range(0, total, 512):
            hook.feed_pcm(state, ramp[offset : offset + 512].tobytes())

        # Wake fires; the satellite opens the mic and flushes the boundary.
        hook.suspend()
        hook._detection_index = total - HOP_SAMPLES
        hook.flush_preroll(satellite)
        after_flush = satellite.delivered().size

        # The mic is open now: subsequent command audio must reach STT.
        hook.feed_pcm(state, np.full(512, 42, dtype="<i2").tobytes())
        delivered = satellite.delivered()
        assert delivered.size == after_flush + 512
        assert np.all(delivered[after_flush:] == 42)
    finally:
        hook.shutdown()


def test_cold_start_after_rearm_is_not_reported_as_underflow() -> None:
    """A wake soon after a rearm asks for audio that never existed.

    The ring re-anchors on every rearm, so a 500 ms trim reaches back past the
    start of the epoch. That is not the ring dropping audio, and stamping it as
    underflow would put a false defect marker on a good command's sidecar.
    """
    hook = _hook_with_detection()
    satellite = _RecordingSatellite()
    state = SimpleNamespace(satellite=satellite)

    hook.rearm()
    # Only 100 ms of audio since the rearm, far less than the 500 ms skip.
    hook.feed_pcm(state, np.zeros(1600, dtype="<i2").tobytes())
    hook._detection_index = hook._ring.end_index

    result = hook.flush_preroll(satellite)
    assert result.underflow is False


def test_genuine_ring_overwrite_is_still_reported_as_underflow() -> None:
    """Audio the ring held and lost must stay visible as a real defect."""
    provider = MagicMock(available=True)
    provider.predict_window.return_value = None
    hook = SaySoExternalWakeHook(provider, preroll_ms=1000, wake_skip_ms=500)
    satellite = _RecordingSatellite()
    satellite._is_streaming_audio = False
    state = SimpleNamespace(satellite=satellite)

    # Overrun the ring so the oldest audio of this epoch is genuinely gone.
    block = np.ones(4096, dtype="<i2").tobytes()
    for _ in range((hook._ring.capacity // 4096) + 4):
        hook.feed_pcm(state, block)

    span_start = hook._ring.available_span()[0]
    assert span_start > hook._ring.origin, "ring did not actually wrap"
    # Ask for a trim inside the epoch but below what is still held.
    hook._detection_index = span_start
    hook._wake_skip_ms = 500

    result = hook.flush_preroll(satellite)
    assert result.underflow is True
