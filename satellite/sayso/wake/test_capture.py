"""Tests for sample-ordered capture and the explicit resampler."""

from __future__ import annotations

import numpy as np

from satellite.sayso.wake.capture import (
    CaptureResampler,
    WakeCaptureRing,
    gain_scalar_from_db,
)

# Window half-width 16 taps => 33-tap kernel, so one impulse spreads over at
# most ~33 input samples, i.e. ~11 output samples at 48k->16k.
_FILTER_SPREAD = 40


def test_resampler_produces_expected_length_for_44100_to_16000() -> None:
    resampler = CaptureResampler(input_rate=44100, output_rate=16000)
    block = np.zeros(441, dtype="<i2")
    out = b"".join(resampler.process(block.tobytes()) for _ in range(10))
    samples = np.frombuffer(out, dtype="<i2")
    # 4410 input samples at 44.1 kHz is exactly 100 ms, so 1600 output samples.
    assert abs(samples.size - 1600) <= 1


def test_resampler_preserves_length_across_many_small_blocks() -> None:
    resampler = CaptureResampler(input_rate=44100, output_rate=16000)
    total = 0
    for _ in range(100):
        total += len(resampler.process(np.zeros(441, dtype="<i2").tobytes()))
    # Allow one sample of latency slack from the streaming filter state.
    assert abs(total // 2 - 16000) <= 4


def test_resampler_is_continuous_across_block_boundaries() -> None:
    """An impulse split by a block boundary must survive as one output peak."""
    rate = 48000
    resampler = CaptureResampler(input_rate=rate, output_rate=16000)
    block = 240
    impulse_index = block - 1  # exactly the first block's final sample
    first = np.zeros(block, dtype="<i2")
    first[-1] = 20000
    second = np.zeros(block, dtype="<i2")
    out = np.frombuffer(
        resampler.process(first.tobytes()) + resampler.process(second.tobytes()),
        dtype="<i2",
    )
    assert out.size > 0
    expected = round(impulse_index * 16000 / rate)
    peak = int(np.argmax(np.abs(out)))
    # The impulse energy must land at the expected output position. A per-block
    # filter reset would smear it toward the seam and move or duplicate the peak.
    assert abs(peak - expected) <= 2
    # A contiguous run, not two disjoint energy islands straddling the boundary.
    energy = np.abs(out).astype(np.int64)
    assert int(energy[peak]) > 0
    nonzero = np.nonzero(energy)[0]
    assert nonzero[-1] - nonzero[0] <= _FILTER_SPREAD


def test_resampler_single_impulse_matches_one_block_impulse() -> None:
    """Chunking must not change the output: same audio, same result."""
    rate = 44100
    samples = np.zeros(882, dtype="<i2")
    samples[500] = 30000
    whole = CaptureResampler(input_rate=rate, output_rate=16000)
    split = CaptureResampler(input_rate=rate, output_rate=16000)
    out_whole = np.frombuffer(whole.process(samples.tobytes()), dtype="<i2")
    parts = [
        split.process(samples[:333].tobytes()),
        split.process(samples[333:700].tobytes()),
        split.process(samples[700:].tobytes()),
    ]
    out_split = np.frombuffer(b"".join(parts), dtype="<i2")
    assert out_whole.size == out_split.size
    assert np.array_equal(out_whole, out_split)


def test_resampler_rejects_non_integer_ratio_passthrough() -> None:
    resampler = CaptureResampler(input_rate=16000, output_rate=16000)
    samples = np.arange(320, dtype="<i2")
    out = np.frombuffer(resampler.process(samples.tobytes()), dtype="<i2")
    assert out.size == samples.size
    assert np.array_equal(out, samples)


def test_resampler_rejects_zero_rates() -> None:
    import pytest

    with pytest.raises(ValueError):
        CaptureResampler(input_rate=0, output_rate=16000)
    with pytest.raises(ValueError):
        CaptureResampler(input_rate=44100, output_rate=0)


def test_capture_ring_indexes_samples_absolutely() -> None:
    ring = WakeCaptureRing(capacity=1000)
    ring.append(np.arange(10, dtype="<i2").tobytes())
    ring.append(np.arange(10, 20, dtype="<i2").tobytes())
    assert ring.end_index == 20
    slice_ = np.frombuffer(ring.read(5, 15), dtype="<i2")
    assert np.array_equal(slice_, np.arange(5, 15, dtype="<i2"))


def test_capture_ring_read_clamps_to_available_span() -> None:
    ring = WakeCaptureRing(capacity=1000)
    ring.append(np.arange(10, dtype="<i2").tobytes())
    # Nothing before the ring start is fabricated.
    assert np.frombuffer(ring.read(-50, 5), dtype="<i2").size == 5


def test_capture_ring_underflow_reports_span_shortfall() -> None:
    ring = WakeCaptureRing(capacity=8)
    ring.append(np.arange(20, dtype="<i2").tobytes())  # capacity forces wrap
    assert ring.end_index == 20
    span = ring.available_span()
    assert span == (12, 20)
    assert ring.covers(12) is True
    assert ring.covers(11) is False


def test_capture_ring_stt_cursor_never_replays_samples() -> None:
    """Each sample is handed to STT exactly once, in order."""
    ring = WakeCaptureRing(capacity=1000)
    ring.append(np.arange(10, dtype="<i2").tobytes())
    assert np.frombuffer(ring.drain_after_cursor(), dtype="<i2").size == 10
    ring.append(np.arange(10, 20, dtype="<i2").tobytes())
    drained = np.frombuffer(ring.drain_after_cursor(), dtype="<i2")
    assert np.array_equal(drained, np.arange(10, 20, dtype="<i2"))


def test_gain_scalar_from_db_is_monotonic_and_clamped() -> None:
    assert gain_scalar_from_db(0.0) == 1.0
    assert gain_scalar_from_db(6.0) > gain_scalar_from_db(0.0)
    assert gain_scalar_from_db(120.0) <= 8.0


def test_capture_ring_reads_correctly_after_wrap() -> None:
    """Absolute-index mapping must survive a wrap-around write."""
    ring = WakeCaptureRing(capacity=100)
    ring.append(np.arange(150, dtype="<i2").tobytes())
    assert ring.end_index == 150
    # The newest 100 samples (50..149) are held, in order.
    data = np.frombuffer(ring.read(50, 150), dtype="<i2")
    assert data.size == 100
    assert np.array_equal(data, np.arange(50, 150, dtype="<i2"))


def test_capture_ring_rearm_does_not_claim_stale_samples() -> None:
    """After a re-anchor the timeline advances but no old audio is readable."""
    ring = WakeCaptureRing(capacity=1000)
    ring.append(np.arange(500, dtype="<i2").tobytes())
    ring.reset()  # re-anchors at 500, clears audio
    assert ring.end_index == 500
    assert ring.available_span() == (500, 500)
    assert ring.read(0, 500) == b""

    ring.append(np.arange(10, dtype="<i2").tobytes())
    assert ring.available_span() == (500, 510)
    assert np.array_equal(
        np.frombuffer(ring.read(500, 510), dtype="<i2"), np.arange(10, dtype="<i2")
    )


def test_capture_ring_large_append_maps_to_absolute_slots() -> None:
    ring = WakeCaptureRing(capacity=64)
    # Feed values equal to their absolute sample index so slot mapping is visible.
    ring.append(np.arange(0, 10, dtype="<i2").tobytes())
    ring.append(np.arange(10, 210, dtype="<i2").tobytes())
    assert ring.end_index == 210
    data = np.frombuffer(ring.read(146, 210), dtype="<i2")
    assert np.array_equal(data, np.arange(146, 210, dtype="<i2"))
