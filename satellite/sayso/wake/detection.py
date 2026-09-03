from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    phrase: str
    confidence: float
    timestamp: float
