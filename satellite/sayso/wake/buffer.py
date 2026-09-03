"""Rolling wake-word audio window backed by a fixed int16 ring buffer."""

from __future__ import annotations

import numpy as np

from .ring_buffer import Int16RingBuffer


class WakeAudioBuffer:
    def __init__(self, window_samples: int, hop_samples: int) -> None:
        self._ring = Int16RingBuffer(window_samples)
        self._hop_samples = hop_samples
        self._since_predict = 0

    @property
    def window_samples(self) -> int:
        return self._ring.capacity

    @property
    def filled(self) -> bool:
        return self._ring.size >= self._ring.capacity

    def clear(self) -> None:
        self._ring.clear()
        self._since_predict = 0

    def rearm_with_silence(self) -> None:
        """Prefill the window with silence for faster post-TTS re-arm."""
        self._ring.fill_silence()
        self._since_predict = 0

    def feed(self, pcm_s16le: bytes) -> bool:
        """Append PCM and return True when a new inference window is due."""
        samples = np.frombuffer(pcm_s16le, dtype="<i2")
        if samples.size == 0:
            return False
        self._ring.extend(samples)
        self._since_predict += int(samples.size)
        if not self.filled or self._since_predict < self._hop_samples:
            return False
        self._since_predict = 0
        return True

    def window(self) -> np.ndarray:
        return self._ring.view()


class WakePrerollLookback:
    """Rolling PCM lookback for post-wake STT preroll flush."""

    def __init__(self, preroll_ms: int, sample_rate: int = 16000) -> None:
        capacity = max(0, preroll_ms * sample_rate // 1000)
        self._ring = Int16RingBuffer(capacity) if capacity > 0 else None
        self._sample_rate = sample_rate

    def clear(self) -> None:
        if self._ring is not None:
            self._ring.clear()

    def feed(self, pcm_s16le: bytes) -> None:
        if self._ring is None or not pcm_s16le:
            return
        samples = np.frombuffer(pcm_s16le, dtype="<i2")
        self._ring.extend(samples)

    def flush_bytes(self, wake_skip_ms: int) -> bytes:
        if self._ring is None or self._ring.size == 0:
            return b""
        skip_samples = wake_skip_ms * self._sample_rate // 1000
        window = self._ring.view()
        if skip_samples >= window.size:
            trail_samples = min(
                250 * self._sample_rate // 1000,
                window.size,
            )
            return window[-trail_samples:].astype("<i2", copy=False).tobytes()
        return window[skip_samples:].astype("<i2", copy=False).tobytes()
