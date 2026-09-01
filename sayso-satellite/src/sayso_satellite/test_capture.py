"""Tests for push-to-talk capture with bounded pre-roll."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from sayso_satellite.capture import (
    BYTES_PER_SAMPLE,
    CHANNELS,
    DEFAULT_PRE_ROLL_MS,
    FixtureMicSource,
    PushToTalkCapture,
    SAMPLE_RATE_HZ,
    expected_pcm_byte_length,
    pcm_duration_ms,
    read_pcm16_file,
)

FIXTURES = Path(__file__).resolve().parents[3] / "evals" / "fixtures"
RECORDED_PCM = FIXTURES / "audio_pcm16_mono_16k.bin"
CORNER_LAMP_PCM = FIXTURES / "turn_off_the_corner_lamp.pcm"
RECORDED_DURATION_MS = 160
LEADING_SAMPLE = 1000
MAIN_SAMPLE = 500
LEADING_DURATION_MS = 80


def _leading_byte_length() -> int:
    return expected_pcm_byte_length(duration_ms=LEADING_DURATION_MS)


def _sample_at(pcm: bytes, index: int) -> int:
    return struct.unpack_from("<h", pcm, index * BYTES_PER_SAMPLE)[0]


def test_fixture_is_pcm16_mono_16k() -> None:
    pcm = RECORDED_PCM.read_bytes()
    assert len(pcm) == expected_pcm_byte_length(duration_ms=RECORDED_DURATION_MS)
    assert len(pcm) % BYTES_PER_SAMPLE == 0
    assert pcm_duration_ms(byte_length=len(pcm)) == RECORDED_DURATION_MS
    leading_bytes = _leading_byte_length()
    assert _sample_at(pcm, 0) == LEADING_SAMPLE
    assert _sample_at(pcm, leading_bytes // BYTES_PER_SAMPLE) == MAIN_SAMPLE


def test_pre_roll_retains_leading_phoneme() -> None:
    pcm = RECORDED_PCM.read_bytes()
    chunk_bytes = expected_pcm_byte_length(duration_ms=20)
    pre_roll_ms = 100
    capture = PushToTalkCapture(pre_roll_ms=pre_roll_ms)
    mic = FixtureMicSource(pcm, chunk_bytes=chunk_bytes)

    for _ in range(6):
        capture.feed(mic.read(max_bytes=chunk_bytes))

    capture.begin()
    for _ in range(4):
        capture.feed(mic.read(max_bytes=chunk_bytes))
    result = capture.end()

    assert len(result) % BYTES_PER_SAMPLE == 0
    assert _sample_at(result, 0) == LEADING_SAMPLE

    preroll_only = PushToTalkCapture(pre_roll_ms=0)
    mic_no_preroll = FixtureMicSource(pcm, chunk_bytes=chunk_bytes)
    for _ in range(6):
        preroll_only.feed(mic_no_preroll.read(max_bytes=chunk_bytes))
    preroll_only.begin()
    for _ in range(4):
        preroll_only.feed(mic_no_preroll.read(max_bytes=chunk_bytes))
    clipped = preroll_only.end()

    assert _sample_at(clipped, 0) == MAIN_SAMPLE
    assert result[: len(clipped)] != clipped[: min(len(result), len(clipped))]


def test_capture_output_framing_for_audio_api() -> None:
    pcm = RECORDED_PCM.read_bytes()
    capture = PushToTalkCapture(pre_roll_ms=DEFAULT_PRE_ROLL_MS)
    mic = FixtureMicSource(pcm, chunk_bytes=expected_pcm_byte_length(duration_ms=10))

    while not mic.exhausted:
        chunk = mic.read(max_bytes=expected_pcm_byte_length(duration_ms=10))
        if not chunk:
            break
        capture.feed(chunk)

    capture.begin()
    while not mic.exhausted:
        chunk = mic.read(max_bytes=expected_pcm_byte_length(duration_ms=10))
        if not chunk:
            break
        capture.feed(chunk)
    result = capture.end()

    assert len(result) > 0
    assert len(result) % BYTES_PER_SAMPLE == 0
    assert pcm_duration_ms(byte_length=len(result)) > 0
    assert SAMPLE_RATE_HZ == 16_000
    assert CHANNELS == 1


def test_begin_while_active_raises() -> None:
    capture = PushToTalkCapture(pre_roll_ms=20)
    capture.feed(b"\x00\x00")
    capture.begin()
    with pytest.raises(RuntimeError, match="already active"):
        capture.begin()


def test_end_without_begin_raises() -> None:
    capture = PushToTalkCapture(pre_roll_ms=20)
    with pytest.raises(RuntimeError, match="not active"):
        capture.end()


def test_read_pcm16_file_reads_fixture() -> None:
    pcm = read_pcm16_file(RECORDED_PCM)
    assert pcm == RECORDED_PCM.read_bytes()
    assert len(pcm) % BYTES_PER_SAMPLE == 0


def test_read_pcm16_file_rejects_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        read_pcm16_file(empty)


def test_read_pcm16_file_rejects_odd_byte_length(tmp_path: Path) -> None:
    odd = tmp_path / "odd.bin"
    odd.write_bytes(b"\x00\x01\x02")
    with pytest.raises(ValueError, match="even"):
        read_pcm16_file(odd)


def test_corner_lamp_fixture_is_valid_pcm16_mono_16k() -> None:
    pcm = read_pcm16_file(CORNER_LAMP_PCM)
    assert len(pcm) % BYTES_PER_SAMPLE == 0
    assert pcm_duration_ms(byte_length=len(pcm)) == 2500
    assert SAMPLE_RATE_HZ == 16_000
    assert CHANNELS == 1
