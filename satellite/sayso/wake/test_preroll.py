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


def test_detection_index_tracks_the_window_not_the_chunk_boundary() -> None:
    """The scored window's end index must not depend on the caller's chunk size.

    Windows are emitted on a hop grid. Whatever chunk size the audio server
    happens to use, the same sample of audio must produce the same detection
    index, or the preroll trim moves with the chunking.
    """
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

    # 371 is what a 1024-frame 44.1 kHz block resamples to; 512 and 2560 bracket it.
    reference = indices_for(320)
    assert reference, "no windows were submitted"
    for chunk in (371, 512, 2560):
        common = min(len(reference), len(indices_for(chunk)))
        assert indices_for(chunk)[:common] == reference[:common], f"chunk {chunk}"

    # And each index must sit on the hop grid, not wherever a chunk happened to end.
    for index in reference:
        assert (index - WINDOW_SAMPLES) % HOP_SAMPLES == 0
