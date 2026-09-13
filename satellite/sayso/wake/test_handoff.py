"""Sample-ordered, exactly-once wake→STT handoff across the capture ring."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from satellite.sayso.config import WakeWordCfg
from satellite.sayso.wake.hook import DEFAULT_WAKE_SKIP_MS, SaySoExternalWakeHook
from satellite.sayso.wake.livekit import HOP_SAMPLES, SAMPLE_RATE, WINDOW_SAMPLES


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


# Detection lag of each fired mined clip, measured by re-scoring the clip with
# the wake model under a growing tail cut (scripts/wake_lag_probe.py). Not from
# an energy envelope -- a gate cannot separate the phrase from an inter-syllable
# gap or from room background at this SNR.
MEASURED_DETECTION_LAGS_MS = (0, 40, 80, 80, 120, 120, 240)
# Burst that demonstrably opened HA's VAD and cost the whole command.
OBSERVED_VAD_LATCH_BURST_MS = 280


def test_lookback_bounds_do_not_overlap() -> None:
    """Pin the fact that no fixed lookback is correct, so it is not re-argued.

    Never truncating a pauseless command needs the lookback at or above the
    worst lag. Never prepending a VAD-openable burst needs it at or near the
    smallest lag, which is 0. Those cannot both hold, so DEFAULT_WAKE_SKIP_MS
    is only a choice of which way to fail -- and trimming to the real phrase
    end (issue #49 item 3) is the actual fix.

    If a future measurement makes these overlap, this test fails and the
    fixed-duration approach becomes defensible again.
    """
    assert max(MEASURED_DETECTION_LAGS_MS) > min(MEASURED_DETECTION_LAGS_MS) + 100


def test_default_lookback_fails_toward_truncation_not_vad_latch() -> None:
    """The chosen failure mode is the recoverable one.

    A short lookback clips the command's onset: partial, and Whisper still sees
    the rest. Too long a one prepends wake-word speech, HA's VAD opens on it and
    times out during the speaker's pause, and the command is dropped whole --
    the failure actually seen in production (447/768/766 ms accepted as speech
    while the capture held 2000-3300 ms; two transcripts were just "So").
    So the default sits near the median lag, far below the burst known to latch.
    """
    worst_prepend = DEFAULT_WAKE_SKIP_MS - min(MEASURED_DETECTION_LAGS_MS)
    assert 0 < DEFAULT_WAKE_SKIP_MS
    assert worst_prepend < OBSERVED_VAD_LATCH_BURST_MS, (
        f"worst-case prepend {worst_prepend} ms is at or above the {OBSERVED_VAD_LATCH_BURST_MS} ms "
        "burst that closed HA's STT window"
    )


def test_flush_emits_from_the_lookback_boundary() -> None:
    """Whatever the value, the flush must start exactly one lookback early."""
    provider = MagicMock(available=True)
    provider.predict_window.return_value = None
    hook = SaySoExternalWakeHook(provider, preroll_ms=2000)
    satellite = _RecordingSatellite()

    lookback = DEFAULT_WAKE_SKIP_MS * SAMPLE_RATE // 1000
    detection = 32000
    hook._ring.append(np.arange(1, detection + 1, dtype="<i2").tobytes())
    hook._detection_index = detection

    result = hook.flush_preroll(satellite)

    assert result.start_index == detection - lookback
    emitted = np.frombuffer(result.pcm, dtype="<i2")
    assert emitted.size == lookback
    assert emitted[0] == np.int16(detection - lookback + 1)


def test_config_default_matches_the_hook_default() -> None:
    """config.py keeps the value as a literal to stay yaml-only; pin them."""
    assert WakeWordCfg.wake_skip_ms == DEFAULT_WAKE_SKIP_MS


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
