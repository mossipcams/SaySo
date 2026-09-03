"""Unit tests for generated notification tones."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

from satellite.sayso.generate_sounds import (
    FAILURE_DURATION,
    FAILURE_FREQS,
    NOTIFICATION_VOLUME,
    UNAVAILABLE_DURATION,
    UNAVAILABLE_FREQS,
    WAKE_DURATION,
    WAKE_FREQS,
    write_tone,
)

OLD_DEFAULT_VOLUME = 0.25
MIN_AUDIBLE_PEAK = int(0.5 * 32767)


def _read_wav_peak_and_duration(path: Path) -> tuple[float, int]:
    with wave.open(str(path), "r") as wf:
        rate = wf.getframerate()
        n = wf.getnframes()
        frames = wf.readframes(n)
    samples = struct.unpack(f"<{n}h", frames)
    return n / rate, max(abs(sample) for sample in samples)


def test_write_tone_produces_mono_s16le_wav(tmp_path: Path) -> None:
    out = tmp_path / "tone.wav"
    write_tone(out, [440.0], duration=0.1)

    with wave.open(str(out), "r") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 22050
        assert wf.getnframes() > 0


def test_notification_tones_are_long_loud_and_distinct(tmp_path: Path) -> None:
    assert NOTIFICATION_VOLUME >= 0.6

    cases = [
        ("wake.wav", WAKE_FREQS, WAKE_DURATION),
        ("failure.wav", FAILURE_FREQS, FAILURE_DURATION),
        ("unavailable.wav", UNAVAILABLE_FREQS, UNAVAILABLE_DURATION),
    ]
    peaks: dict[str, int] = {}
    for name, freqs, duration in cases:
        path = tmp_path / name
        write_tone(path, freqs, duration)

        actual_duration, peak = _read_wav_peak_and_duration(path)
        assert actual_duration >= 0.7, name
        assert peak >= MIN_AUDIBLE_PEAK, name
        assert peak > int(OLD_DEFAULT_VOLUME * 32767), name
        peaks[name] = peak

    wake_pitch = sum(WAKE_FREQS) / len(WAKE_FREQS)
    failure_pitch = sum(FAILURE_FREQS) / len(FAILURE_FREQS)
    unavailable_pitch = sum(UNAVAILABLE_FREQS) / len(UNAVAILABLE_FREQS)
    assert wake_pitch > unavailable_pitch > failure_pitch
    assert peaks["wake.wav"] > 0
    assert peaks["failure.wav"] > 0
    assert peaks["unavailable.wav"] > 0
