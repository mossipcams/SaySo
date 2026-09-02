"""Tests for live microphone input sources."""

from __future__ import annotations

import struct
from unittest.mock import MagicMock

import pytest

from sayso_satellite.capture import (
    BYTES_PER_SAMPLE,
    PushToTalkCapture,
    expected_pcm_byte_length,
)
from sayso_satellite.microphone import (
    DEFAULT_CHUNK_MS,
    FakeLiveMicSource,
    MacMicrophoneSource,
    MicInputError,
    capture_live_pcm,
    open_mac_microphone,
)


def _sample_at(pcm: bytes, index: int) -> int:
    return struct.unpack_from("<h", pcm, index * BYTES_PER_SAMPLE)[0]


def _pcm_bytes(*samples: int) -> bytes:
    return b"".join(struct.pack("<h", sample) for sample in samples)


def test_fake_live_source_yields_fixed_chunks_into_preroll_and_turn_capture() -> None:
    chunk_bytes = expected_pcm_byte_length(duration_ms=20)
    leading = _pcm_bytes(1000) * (expected_pcm_byte_length(duration_ms=80) // BYTES_PER_SAMPLE)
    speech = _pcm_bytes(500) * (expected_pcm_byte_length(duration_ms=40) // BYTES_PER_SAMPLE)
    pcm = leading + speech
    mic = FakeLiveMicSource(pcm, chunk_bytes=chunk_bytes)
    capture = PushToTalkCapture(pre_roll_ms=100)

    for _ in range(4):
        capture.feed(mic.read(max_bytes=chunk_bytes))

    capture.begin()
    while not mic.exhausted:
        chunk = mic.read(max_bytes=chunk_bytes)
        if not chunk:
            break
        capture.feed(chunk)
    result = capture.end()

    assert len(result) % BYTES_PER_SAMPLE == 0
    assert _sample_at(result, 0) == 1000
    assert _sample_at(result, len(leading) // BYTES_PER_SAMPLE) == 500


def test_fake_live_source_close_is_idempotent() -> None:
    mic = FakeLiveMicSource(_pcm_bytes(1, 2, 3), chunk_bytes=BYTES_PER_SAMPLE * 2)

    mic.close()
    mic.close()

    assert mic.closed is True
    assert mic.read(max_bytes=BYTES_PER_SAMPLE * 2) == b""


def test_fake_live_source_read_after_close_returns_empty() -> None:
    mic = FakeLiveMicSource(_pcm_bytes(1, 2), chunk_bytes=BYTES_PER_SAMPLE * 2)
    mic.close()

    assert mic.read(max_bytes=BYTES_PER_SAMPLE * 2) == b""


def test_fake_live_source_input_failure_closes_cleanly() -> None:
    mic = FakeLiveMicSource(
        _pcm_bytes(1, 2, 3, 4),
        chunk_bytes=BYTES_PER_SAMPLE * 2,
        fail_after_reads=1,
    )

    assert mic.read(max_bytes=BYTES_PER_SAMPLE * 2)
    with pytest.raises(MicInputError, match="simulated input failure"):
        mic.read(max_bytes=BYTES_PER_SAMPLE * 2)

    assert mic.closed is True
    assert mic.read(max_bytes=BYTES_PER_SAMPLE * 2) == b""


def test_mac_microphone_rejects_odd_byte_reads() -> None:
    process = MagicMock()
    process.stdout.read.side_effect = [b"\x00\x01\x02", b""]
    process.poll.return_value = None
    mic = MacMicrophoneSource(process, chunk_bytes=expected_pcm_byte_length(duration_ms=20))

    with pytest.raises(MicInputError, match="invalid PCM16 framing"):
        mic.read(max_bytes=expected_pcm_byte_length(duration_ms=20))

    assert mic.closed is True
    process.terminate.assert_called_once()


def test_mac_microphone_closes_process_on_shutdown() -> None:
    process = MagicMock()
    process.stdout.read.return_value = b""
    process.poll.return_value = None
    mic = MacMicrophoneSource(process, chunk_bytes=expected_pcm_byte_length(duration_ms=20))

    mic.close()

    assert mic.closed is True
    process.terminate.assert_called_once()
    process.wait.assert_called_once()


def test_mac_microphone_process_failure_closes_cleanly() -> None:
    process = MagicMock()
    process.stdout.read.return_value = b""
    process.poll.return_value = 1
    process.returncode = 1
    mic = MacMicrophoneSource(process, chunk_bytes=expected_pcm_byte_length(duration_ms=20))

    with pytest.raises(MicInputError, match="microphone process exited"):
        mic.read(max_bytes=expected_pcm_byte_length(duration_ms=20))

    assert mic.closed is True


def test_capture_live_pcm_uses_manual_turn_with_fake_source() -> None:
    chunk_bytes = expected_pcm_byte_length(duration_ms=20)
    leading = _pcm_bytes(42) * (expected_pcm_byte_length(duration_ms=60) // BYTES_PER_SAMPLE)
    speech = _pcm_bytes(7) * (expected_pcm_byte_length(duration_ms=40) // BYTES_PER_SAMPLE)
    mic = FakeLiveMicSource(leading + speech, chunk_bytes=chunk_bytes)

    pcm = capture_live_pcm(
        mic,
        duration_ms=100,
        chunk_bytes=chunk_bytes,
        pre_roll_ms=60,
    )

    assert mic.closed is True
    assert len(pcm) % BYTES_PER_SAMPLE == 0
    assert _sample_at(pcm, 0) == 42


def test_open_mac_microphone_builds_ffmpeg_command(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[list[str]] = []

    class FakeProcess:
        stdout = MagicMock()
        stderr = MagicMock()
        pid = 1234

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            return 0

    def fake_popen(cmd: list[str], **kwargs: object) -> FakeProcess:
        created.append(cmd)
        return FakeProcess()

    monkeypatch.setattr("sayso_satellite.microphone.subprocess.Popen", fake_popen)

    mic = open_mac_microphone(chunk_ms=DEFAULT_CHUNK_MS)

    assert "-ar" in created[0]
    assert "16000" in created[0]
    assert "-ac" in created[0]
    assert "1" in created[0]
    assert "-f" in created[0]
    assert "s16le" in created[0]
    mic.close()
