
from __future__ import annotations

import json
import random
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from satellite.sayso.config import WakeWordCfg
from satellite.sayso.wake.detection import Detection
from satellite.sayso.wake.hook import (
    DEFAULT_WAKE_SKIP_MS,
    SaySoExternalWakeHook,
    wake_ring_capacity,
)
from satellite.sayso.wake.livekit import HOP_SAMPLES, SAMPLE_RATE, WINDOW_SAMPLES
from satellite.sayso.wake.mining import HardNegativeMiner


class _RecordingSatellite:

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
        total = WINDOW_SAMPLES * 2
        ramp = np.arange(total, dtype="<i2")
        chunk = 512
        for offset in range(0, total, chunk):
            hook.feed_pcm(state, ramp[offset : offset + chunk].tobytes())
    finally:
        hook.shutdown()

    delivered = satellite.delivered()
    assert delivered.size == total
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

        hook._detection_index = total - HOP_SAMPLES
        before = satellite.delivered().size

        hook._suspended = True
        result = hook.flush_preroll(satellite)
        assert result.pcm == b""
        assert satellite.delivered().size == before

        hook._suspended = False
        hook.feed_pcm(state, np.full(chunk, 1234, dtype="<i2").tobytes())
        delivered = satellite.delivered()
        assert np.all(delivered[before:] == 1234)
    finally:
        hook.shutdown()


def test_flush_preroll_uses_detection_index_not_flush_time() -> None:

    def _prepared_hook() -> tuple[SaySoExternalWakeHook, _RecordingSatellite]:
        provider = MagicMock(available=True)
        provider.predict_window.return_value = None
        hook = SaySoExternalWakeHook(provider, preroll_ms=2000, wake_skip_ms=500)
        satellite = _RecordingSatellite()
        hook.bind_satellite(lambda: satellite)
        hook._ring.append(np.zeros(32000, dtype="<i2").tobytes())
        hook._detection_index = 32000
        return hook, satellite

    early_hook, early_sat = _prepared_hook()
    early = early_hook.flush_preroll(early_sat)

    late_hook, late_sat = _prepared_hook()
    late_hook._ring.append(np.full(640, 5, dtype="<i2").tobytes())
    late = late_hook.flush_preroll(late_sat)

    assert early.start_index == late.start_index == 32000 - 8000
    assert np.frombuffer(early.pcm, dtype="<i2").size == 8000
    assert np.frombuffer(late.pcm, dtype="<i2").size == 8640


MEASURED_DETECTION_LAGS_MS = (0, 40, 80, 80, 120, 120, 240)
OBSERVED_VAD_LATCH_BURST_MS = 280


def test_lookback_bounds_do_not_overlap() -> None:
    assert max(MEASURED_DETECTION_LAGS_MS) > min(MEASURED_DETECTION_LAGS_MS) + 100


def test_default_lookback_fails_toward_truncation_not_vad_latch() -> None:
    worst_prepend = DEFAULT_WAKE_SKIP_MS - min(MEASURED_DETECTION_LAGS_MS)
    assert 0 < DEFAULT_WAKE_SKIP_MS
    assert worst_prepend < OBSERVED_VAD_LATCH_BURST_MS, (
        f"worst-case prepend {worst_prepend} ms is at or above the {OBSERVED_VAD_LATCH_BURST_MS} ms "
        "burst that closed HA's STT window"
    )


def test_flush_emits_from_the_lookback_boundary() -> None:
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

        hook.suspend()
        hook._detection_index = total - HOP_SAMPLES
        hook.flush_preroll(satellite)
        after_flush = satellite.delivered().size

        hook.feed_pcm(state, np.full(512, 42, dtype="<i2").tobytes())
        delivered = satellite.delivered()
        assert delivered.size == after_flush + 512
        assert np.all(delivered[after_flush:] == 42)
    finally:
        hook.shutdown()


def test_cold_start_after_rearm_is_not_reported_as_underflow() -> None:
    hook = _hook_with_detection()
    satellite = _RecordingSatellite()
    state = SimpleNamespace(satellite=satellite)

    hook.rearm()
    hook.feed_pcm(state, np.zeros(1600, dtype="<i2").tobytes())
    hook._detection_index = hook._ring.end_index

    result = hook.flush_preroll(satellite)
    assert result.underflow is False


def test_genuine_ring_overwrite_is_still_reported_as_underflow() -> None:
    provider = MagicMock(available=True)
    provider.predict_window.return_value = None
    hook = SaySoExternalWakeHook(provider, preroll_ms=1000, wake_skip_ms=500)
    satellite = _RecordingSatellite()
    satellite._is_streaming_audio = False
    state = SimpleNamespace(satellite=satellite)

    block = np.ones(4096, dtype="<i2").tobytes()
    for _ in range((hook._ring.capacity // 4096) + 4):
        hook.feed_pcm(state, block)

    span_start = hook._ring.available_span()[0]
    assert span_start > hook._ring.origin, "ring did not actually wrap"
    hook._detection_index = span_start
    hook._wake_skip_ms = 500

    result = hook.flush_preroll(satellite)
    assert result.underflow is True


def test_feed_pcm_forwards_every_chunk_to_miner() -> None:
    provider = MagicMock(available=True)
    provider.predict_window.return_value = None
    miner = MagicMock()
    hook = SaySoExternalWakeHook(provider, preroll_ms=1000, wake_skip_ms=500, miner=miner)
    state = SimpleNamespace(satellite=_RecordingSatellite())

    chunk_size = 384
    total_samples = 1920
    audio = np.arange(total_samples, dtype="<i2")
    for offset in range(0, total_samples, chunk_size):
        hook.feed_pcm(state, audio[offset : offset + chunk_size].tobytes())

    calls = miner.feed_audio.call_args_list
    assert len(calls) == total_samples // chunk_size
    for i, call in enumerate(calls):
        expected_pcm = audio[i * chunk_size : (i + 1) * chunk_size].tobytes()
        assert call.args[0] == expected_pcm
        assert call.args[1] == (i + 1) * chunk_size


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")


def test_detection_post_context_survives_rearm(tmp_path: Path) -> None:
    miner = HardNegativeMiner(
        tmp_path,
        mine_threshold=0.05,
        detect_threshold=0.5,
        detection_post_context_ms=2000,
        below_sample_rate=0.0,
        rng=random.Random(0),
    )
    miner.start()
    provider = MagicMock(available=True)
    provider.predict_window.return_value = None
    hook = SaySoExternalWakeHook(provider, preroll_ms=750, wake_skip_ms=500, miner=miner)
    state = SimpleNamespace(satellite=_RecordingSatellite())

    def feed(samples: np.ndarray) -> None:
        for offset in range(0, samples.size, 512):
            hook.feed_pcm(state, samples[offset : offset + 512].tobytes())

    stt_end = pre_samples = WINDOW_SAMPLES + 1
    stt_end += int(1000 * SAMPLE_RATE // 1000)
    total = stt_end + 24000
    audio = np.arange(total, dtype="<i2")
    feed(audio[:pre_samples])
    sample_index = pre_samples - 1
    window_start = sample_index + 1 - WINDOW_SAMPLES
    capture_id = miner.offer(
        0.9, audio[window_start : sample_index + 1], sample_index=sample_index
    )
    hook._detection_index = sample_index

    feed(audio[pre_samples:stt_end])
    hook.rearm()
    feed(audio[stt_end:])

    miner.stop(timeout=5.0)

    assert capture_id is not None
    record = tmp_path / "records" / capture_id
    meta = json.loads((record / "record.json").read_text(encoding="utf-8"))
    assert meta["sampling_reason"] == "detection"
    assert meta["quality_flags"]["post_context_missing"] is False
    sample_end = meta["sample_end"]
    post = _read_wav(record / "post.wav")
    assert post.size == SAMPLE_RATE * 2
    np.testing.assert_array_equal(post, audio[sample_end : sample_end + SAMPLE_RATE * 2])


MAX_SUPPORTED_WAKE_SKIP_MS = OBSERVED_VAD_LATCH_BURST_MS
WORKER_QUEUE_LAG_MS = 25 * HOP_SAMPLES * 1000 // SAMPLE_RATE


def _wake_hook(wake_skip_ms: int = 500) -> tuple[SaySoExternalWakeHook, _RecordingSatellite]:
    provider = MagicMock(available=True)
    provider.predict_window.return_value = None
    hook = SaySoExternalWakeHook(provider, preroll_ms=1000, wake_skip_ms=wake_skip_ms)
    satellite = _RecordingSatellite()
    hook.bind_satellite(lambda: satellite)
    return hook, satellite


def _detection(sample_index: int) -> Detection:
    return Detection(phrase="SaySo", confidence=0.9, timestamp=0.0, sample_index=sample_index)


def test_wake_ring_holds_queue_lag_skip_and_preroll() -> None:
    preroll_ms = 2000
    required = (WORKER_QUEUE_LAG_MS + MAX_SUPPORTED_WAKE_SKIP_MS + preroll_ms) * SAMPLE_RATE // 1000
    assert wake_ring_capacity(preroll_ms) >= required


def test_rearm_clears_pending_windows() -> None:
    hook, _ = _wake_hook()
    hook._worker.submit(np.zeros(WINDOW_SAMPLES, dtype="<i2"), WINDOW_SAMPLES)
    hook.rearm()
    assert hook._worker._queue.take(timeout=0.0) is None


def test_detection_at_or_before_rearm_boundary_does_not_wake() -> None:
    hook, satellite = _wake_hook()
    state = SimpleNamespace(satellite=satellite)
    hook.feed_pcm(state, np.zeros(WINDOW_SAMPLES, dtype="<i2").tobytes())
    boundary = hook._ring.end_index
    hook.rearm()

    hook._on_detection(_detection(boundary))
    hook._on_detection(_detection(boundary - HOP_SAMPLES))

    assert satellite.wakeups == []
    assert hook.last_detection_index is None


def test_detection_after_rearm_boundary_still_wakes() -> None:
    hook, satellite = _wake_hook()
    state = SimpleNamespace(satellite=satellite)
    hook.feed_pcm(state, np.zeros(WINDOW_SAMPLES, dtype="<i2").tobytes())
    hook.rearm()
    hook.feed_pcm(state, np.zeros(HOP_SAMPLES, dtype="<i2").tobytes())

    hook._on_detection(_detection(hook._ring.end_index))

    assert len(satellite.wakeups) == 1


def test_suspend_and_resume_clear_pending_windows_and_bound_detections() -> None:
    hook, satellite = _wake_hook()
    state = SimpleNamespace(satellite=satellite)
    hook.feed_pcm(state, np.zeros(WINDOW_SAMPLES, dtype="<i2").tobytes())
    boundary = hook._ring.end_index
    hook.suspend()
    assert hook._worker._queue.take(timeout=0.0) is None

    hook.resume()
    hook._on_detection(_detection(boundary))

    assert satellite.wakeups == []


def test_detection_delivered_after_queue_lag_keeps_command_onset() -> None:
    hook, satellite = _wake_hook(DEFAULT_WAKE_SKIP_MS)
    satellite._is_streaming_audio = False
    state = SimpleNamespace(satellite=satellite)
    window_end = WINDOW_SAMPLES
    total = window_end + WORKER_QUEUE_LAG_MS * SAMPLE_RATE // 1000
    ramp = (np.arange(total) % 30000).astype("<i2")
    hook.feed_pcm(state, ramp[:window_end].tobytes())
    hook.feed_pcm(state, ramp[window_end:].tobytes())

    hook._on_detection(_detection(window_end))
    result = hook.flush_preroll(satellite)

    lookback = DEFAULT_WAKE_SKIP_MS * SAMPLE_RATE // 1000
    assert len(satellite.wakeups) == 1
    assert result.underflow is False
    assert result.start_index == window_end - lookback
    assert np.array_equal(
        np.frombuffer(result.pcm, dtype="<i2"), ramp[window_end - lookback : total]
    )
