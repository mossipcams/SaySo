"""Tests for detection-indexed preroll trimming and the atomic handoff."""

from __future__ import annotations

import numpy as np

from satellite.sayso.wake.buffer import WakePrerollLookback


def test_preroll_trim_is_relative_to_detection_index_not_flush_time() -> None:
    """The trim must ignore how late the flush call arrives."""
    lookback = WakePrerollLookback(preroll_ms=2000, sample_rate=16000)
    # 2 s of zeros (the wake word), then 1 s of 7s (the command).
    lookback.feed(np.zeros(32000, dtype="<i2").tobytes())
    lookback.feed(np.full(16000, 7, dtype="<i2").tobytes())

    # Detection landing on the last sleep-free sample; skip 500 ms.
    detection_index = 32000
    flushed = np.frombuffer(
        lookback.flush_until(detection_index, skip_ms=500).pcm, dtype="<i2"
    )

    # Expect [detection-500ms, now) = 8000 zero samples + 16000 command samples.
    assert flushed.size == 24000
    assert np.all(flushed[:8000] == 0)
    assert np.all(flushed[8000:] == 7)


def test_preroll_trim_start_is_unchanged_by_extra_audio_after_detection() -> None:
    """Audio between detection and flush must be kept, not shift the trim."""
    detection_index = 32000
    skip_ms = 500

    early = WakePrerollLookback(preroll_ms=2000, sample_rate=16000)
    early.feed(np.zeros(32000, dtype="<i2").tobytes())
    early_result = early.flush_until(detection_index, skip_ms=skip_ms)

    late = WakePrerollLookback(preroll_ms=2000, sample_rate=16000)
    late.feed(np.zeros(32000, dtype="<i2").tobytes())
    # 40 ms of extra latency before the flush runs; this is real command audio.
    late.feed(np.full(640, 3, dtype="<i2").tobytes())
    late_result = late.flush_until(detection_index, skip_ms=skip_ms)

    # Same start boundary: the trim did not move with the flush delay.
    assert late_result.start_index == early_result.start_index == 24000
    # The extra latency audio is retained, not dropped or reordered.
    late_pcm = np.frombuffer(late_result.pcm, dtype="<i2")
    assert late_pcm.size == 8640
    assert np.all(late_pcm[:8000] == 0)
    assert np.all(late_pcm[8000:] == 3)


def test_preroll_underflow_is_reported_not_hidden() -> None:
    lookback = WakePrerollLookback(preroll_ms=100, sample_rate=16000)
    lookback.feed(np.full(1600, 5, dtype="<i2").tobytes())
    # A 60 s skip asks for a boundary far earlier than the ring has ever held,
    # so the requested trim is unsatisfiable and must be reported.
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
