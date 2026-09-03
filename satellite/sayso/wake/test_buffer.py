"""Tests for wake audio window buffering."""

from __future__ import annotations

import numpy as np

from satellite.sayso.wake.buffer import WakeAudioBuffer, WakePrerollLookback
from satellite.sayso.wake.livekit import HOP_SAMPLES, WINDOW_SAMPLES


def test_rearm_latency_prefers_silence_prefill_over_clean_window() -> None:
    silence_prefill = HOP_SAMPLES
    clean_window = WINDOW_SAMPLES
    assert silence_prefill < clean_window


def test_wake_buffer_does_not_predict_before_window_is_full() -> None:
    buffer = WakeAudioBuffer(WINDOW_SAMPLES, HOP_SAMPLES)
    chunk = np.zeros(256, dtype="<i2").tobytes()
    chunks_to_fill = (WINDOW_SAMPLES // 256) - 1
    for _ in range(chunks_to_fill):
        assert buffer.feed(chunk) is False
    assert buffer.filled is False


def test_preroll_lookback_drops_wake_skip_prefix_on_flush() -> None:
    lookback = WakePrerollLookback(preroll_ms=1000, sample_rate=16000)
    wake_samples = np.zeros(8000, dtype="<i2")
    command_samples = np.full(8000, 7, dtype="<i2")
    lookback.feed(wake_samples.tobytes())
    lookback.feed(command_samples.tobytes())

    flushed = np.frombuffer(lookback.flush_bytes(wake_skip_ms=500), dtype="<i2")

    assert flushed.size == 8000
    assert np.all(flushed == 7)


def test_preroll_lookback_equal_skip_flushes_trailing_audio() -> None:
    lookback = WakePrerollLookback(preroll_ms=500, sample_rate=16000)
    wake_samples = np.zeros(4000, dtype="<i2")
    command_samples = np.full(4000, 7, dtype="<i2")
    lookback.feed(wake_samples.tobytes())
    lookback.feed(command_samples.tobytes())

    flushed = np.frombuffer(lookback.flush_bytes(wake_skip_ms=500), dtype="<i2")

    assert flushed.size == 4000
    assert np.all(flushed == 7)


def test_preroll_lookback_zero_ms_is_noop() -> None:
    lookback = WakePrerollLookback(preroll_ms=0)
    lookback.feed(np.ones(160, dtype="<i2").tobytes())
    assert lookback.flush_bytes(wake_skip_ms=0) == b""
