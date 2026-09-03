#!/usr/bin/env python3
"""Generate short local notification tones (no speech)."""
from __future__ import annotations

import math
import struct
import wave
from pathlib import Path


def write_tone(path: Path, freqs: list[float], duration: float, volume: float = 0.25) -> None:
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
    write_tone(root / "wake.wav", [880, 1320], 0.18)
    write_tone(root / "failure.wav", [220, 180], 0.25)
    write_tone(root / "unavailable.wav", [330], 0.22)
    print("wrote", root)


if __name__ == "__main__":
    main()
