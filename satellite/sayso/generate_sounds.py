#!/usr/bin/env python3
"""Generate short local notification tones (no speech)."""
from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

NOTIFICATION_VOLUME = 0.6
WAKE_FREQS = [880.0, 1320.0]
WAKE_DURATION = 0.75
FAILURE_FREQS = [220.0, 180.0]
FAILURE_DURATION = 0.8
UNAVAILABLE_FREQS = [330.0]
UNAVAILABLE_DURATION = 0.71


def write_tone(path: Path, freqs: list[float], duration: float, volume: float = NOTIFICATION_VOLUME) -> None:
    rate = 22050
    n = int(rate * duration)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        for i in range(n):
            t = i / rate
            env = min(1.0, i / (rate * 0.01), (n - i) / (rate * 0.04))
            sample = sum(math.sin(2 * math.pi * f * t) for f in freqs) / len(freqs)
            val = int(max(-1.0, min(1.0, sample * volume * env)) * 32767)
            wf.writeframes(struct.pack("<h", val))


def main() -> None:
    root = Path("/opt/sayso-satellite/sounds")
    write_tone(root / "wake.wav", WAKE_FREQS, WAKE_DURATION)
    write_tone(root / "failure.wav", FAILURE_FREQS, FAILURE_DURATION)
    write_tone(root / "unavailable.wav", UNAVAILABLE_FREQS, UNAVAILABLE_DURATION)
    print("wrote", root)


if __name__ == "__main__":
    main()
