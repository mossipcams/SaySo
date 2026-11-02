
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from satellite.sayso.wake.buffer import PrerollFlush, WakePrerollLookback
from satellite.sayso.wake.capture import WakeCaptureRing
from satellite.sayso.wake.hook import SaySoExternalWakeHook, wake_ring_capacity


def test_preroll_trim_is_relative_to_detection_index_not_flush_time() -> None:
    lookback = WakePrerollLookback(preroll_ms=2000, sample_rate=16000)
    lookback.feed(np.zeros(32000, dtype="<i2").tobytes())
    lookback.feed(np.full(16000, 7, dtype="<i2").tobytes())

    detection_index = 32000
    flushed = np.frombuffer(
        lookback.flush_until(detection_index, skip_ms=500).pcm, dtype="<i2"
    )

    assert flushed.size == 24000
    assert np.all(flushed[:8000] == 0)
    assert np.all(flushed[8000:] == 7)


def test_preroll_trim_start_is_unchanged_by_extra_audio_after_detection() -> None:
    detection_index = 32000
    skip_ms = 500

    early = WakePrerollLookback(preroll_ms=2000, sample_rate=16000)
    early.feed(np.zeros(32000, dtype="<i2").tobytes())
    early_result = early.flush_until(detection_index, skip_ms=skip_ms)

    late = WakePrerollLookback(preroll_ms=2000, sample_rate=16000)
    late.feed(np.zeros(32000, dtype="<i2").tobytes())
    late.feed(np.full(640, 3, dtype="<i2").tobytes())
    late_result = late.flush_until(detection_index, skip_ms=skip_ms)

    assert late_result.start_index == early_result.start_index == 24000
    late_pcm = np.frombuffer(late_result.pcm, dtype="<i2")
    assert late_pcm.size == 8640
    assert np.all(late_pcm[:8000] == 0)
    assert np.all(late_pcm[8000:] == 3)


def test_preroll_underflow_is_reported_not_hidden() -> None:
    lookback = WakePrerollLookback(preroll_ms=100, sample_rate=16000)
    lookback.feed(np.full(1600, 5, dtype="<i2").tobytes())
    result = lookback.flush_until(1600, skip_ms=60_000)
    assert result.pcm == b""
    assert result.underflow is True


def test_preroll_zero_ms_is_noop() -> None:
    lookback = WakePrerollLookback(preroll_ms=0)
    lookback.feed(np.ones(160, dtype="<i2").tobytes())
    result = lookback.flush_until(160, skip_ms=0)
    assert result.pcm == b""


def test_preroll_reports_emitted_span_for_cursor_sync() -> None:
    lookback = WakePrerollLookback(preroll_ms=1000, sample_rate=16000)
    lookback.feed(np.arange(16000, dtype="<i2").tobytes())
    result = lookback.flush_until(16000, skip_ms=500)
    assert result.start_index == 8000
    assert result.end_index == 16000
    assert result.underflow is False


def test_detection_index_tracks_the_window_not_the_chunk_boundary() -> None:
    from unittest.mock import MagicMock

    from satellite.sayso.wake.hook import SaySoExternalWakeHook
    from satellite.sayso.wake.livekit import HOP_SAMPLES, WINDOW_SAMPLES

    def indices_for(chunk: int) -> list[int]:
        provider = MagicMock(available=True)
        provider.predict_window.return_value = None
        hook = SaySoExternalWakeHook(provider, preroll_ms=1000, wake_skip_ms=500)
        seen: list[int] = []
        hook._worker.submit = lambda window, sample_index=None: seen.append(sample_index)
        total = WINDOW_SAMPLES + 4 * HOP_SAMPLES
        block = np.zeros(chunk, dtype="<i2").tobytes()
        for _ in range(total // chunk):
            hook.feed_pcm(None, block)
        return seen

    reference = indices_for(320)
    assert reference, "no windows were submitted"
    for chunk in (371, 512, 2560):
        common = min(len(reference), len(indices_for(chunk)))
        assert indices_for(chunk)[:common] == reference[:common], f"chunk {chunk}"

    for index in reference:
        assert (index - WINDOW_SAMPLES) % HOP_SAMPLES == 0


def _stt_hook(ring_samples: int) -> SaySoExternalWakeHook:
    provider = MagicMock(available=True)
    provider.predict_window.return_value = None
    return SaySoExternalWakeHook(
        provider,
        preroll_ms=750,
        wake_skip_ms=50,
        capture_ring=WakeCaptureRing(ring_samples),
    )


def test_stt_preroll_is_identical_with_detection_sized_ring() -> None:
    small = _stt_hook(wake_ring_capacity(750))
    large = _stt_hook(wake_ring_capacity(max(750, 10000)))

    audio = np.arange(48800, dtype="<i2")
    for hook in (small, large):
        for offset in range(0, audio.size, 512):
            hook._ring.append(audio[offset : offset + 512].tobytes())
        hook._detection_index = audio.size

    short_result = small.flush_preroll(None)
    large_result = large.flush_preroll(None)

    assert large_result.pcm == short_result.pcm
    assert np.frombuffer(large_result.pcm, dtype="<i2").size == 800
    assert large_result.start_index == short_result.start_index
    assert large_result.end_index == short_result.end_index
    assert large_result.underflow is False


def test_stt_preroll_underflow_is_identical_with_detection_sized_ring() -> None:
    capacities = (wake_ring_capacity(750), wake_ring_capacity(max(750, 10000)))
    results: list[PrerollFlush] = []
    for capacity in capacities:
        hook = _stt_hook(capacity)
        total = capacity + 8192
        audio = (np.arange(total) % 30000).astype("<i2")
        for offset in range(0, audio.size, 8192):
            hook._ring.append(audio[offset : offset + 8192].tobytes())
        hook._detection_index = hook._ring.available_span()[0]
        results.append(hook.flush_preroll(None))

    for capacity, result in zip(capacities, results):
        assert result.underflow is True
        expected = ((np.arange(total - capacity, total) % 30000).astype("<i2")).tobytes()
        assert result.pcm == expected
