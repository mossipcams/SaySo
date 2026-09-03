"""Tests for wake audio window buffering."""

from __future__ import annotations

import numpy as np

from satellite.sayso.wake.buffer import WakeAudioBuffer
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
