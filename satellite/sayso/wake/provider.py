from __future__ import annotations

from typing import Optional, Protocol

from .detection import Detection


class WakeWordProvider(Protocol):
    """Pluggable on-device wake-word detector.

    Implementations must consume PCM from the existing capture loop and
    must not open the microphone themselves.
    """

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def suspend(self) -> None:
        """Pause detection (TTS / half-duplex)."""

    def resume(self) -> None:
        ...

    def reset(self) -> None:
        """Clear detector state after playback or errors."""

    def process_pcm(self, pcm_s16le: bytes, sample_rate: int = 16000) -> Optional[Detection]:
        """Feed 16-bit little-endian mono PCM. Return a detection or None."""

    def shutdown(self) -> None:
        ...

    @property
    def available(self) -> bool:
        """False if the model is missing/invalid (fail closed)."""
