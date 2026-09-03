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
