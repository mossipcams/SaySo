from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """One wake-word firing.

    ``sample_index`` is the absolute capture sample index of the last audio
    sample the classifier scored. The capture timeline is the only clock shared
    by the wake worker, the capture thread, and the STT handoff, so the preroll
    trim is expressed against it rather than against wall-clock flush time.
    """

    phrase: str
    confidence: float
    timestamp: float
    sample_index: int | None = None
