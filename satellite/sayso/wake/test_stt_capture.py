"""Milestone 0: the exact PCM sent to Home Assistant must be retained."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from sayso.wake.stt_capture import SttAudioRecorder


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")


def test_captured_wav_is_byte_exact_to_the_pcm_sent(tmp_path: Path) -> None:
    recorder = SttAudioRecorder(tmp_path, sample_rate=16000)
    recorder.start()
    try:
        recorder.begin_command("run-1")
        pcm = np.arange(16000, dtype="<i2").tobytes()
        recorder.tap(pcm)
        path = recorder.end_command("run-1", transcript="turn on the light")
    finally:
        recorder.stop()

    assert path is not None
    assert path.is_file()
    assert np.array_equal(_read_wav(path), np.arange(16000, dtype="<i2"))


def test_command_wav_is_split_into_arbitrary_taps(tmp_path: Path) -> None:
    recorder = SttAudioRecorder(tmp_path, sample_rate=16000)
    recorder.start()
    try:
        recorder.begin_command("run-2")
        recorder.tap(np.zeros(100, dtype="<i2").tobytes())
        recorder.tap(np.ones(100, dtype="<i2").tobytes())
        path = recorder.end_command("run-2", transcript="x")
    finally:
        recorder.stop()

    data = _read_wav(path)
    assert data.size == 200
    assert np.all(data[:100] == 0)
    assert np.all(data[100:] == 1)


def test_failed_command_writes_a_second_wav(tmp_path: Path) -> None:
    recorder = SttAudioRecorder(tmp_path, sample_rate=16000)
    recorder.start()
    try:
        recorder.begin_command("run-3")
        recorder.tap(np.full(500, 9, dtype="<i2").tobytes())
        recorder.end_command("run-3", transcript="")
        recorder.flush(timeout=2.0)
        failures = list((tmp_path / "failures").glob("run-3*.wav"))
    finally:
        recorder.stop()

    assert len(failures) == 1
    assert np.all(_read_wav(failures[0]) == 9)


def test_sidecar_records_the_processing_chain(tmp_path: Path) -> None:
    recorder = SttAudioRecorder(
        tmp_path,
        sample_rate=16000,
        capture_rate=44100,
        mic_gain_db=6.0,
        noise_suppression=0,
        auto_gain=0,
    )
    recorder.start()
    try:
        recorder.begin_command("run-4")
        recorder.tap(np.full(1600, 1000, dtype="<i2").tobytes())
        path = recorder.end_command("run-4", transcript="hello", underflow=False)
    finally:
        recorder.stop()

    sidecar = path.with_suffix(".json")
    meta = json.loads(sidecar.read_text())
    assert meta["capture_rate"] == 44100
    assert meta["sample_rate"] == 16000
    assert meta["mic_gain_db"] == 6.0
    assert meta["noise_suppression"] == 0
    assert meta["auto_gain"] == 0
    assert meta["transcript"] == "hello"
    assert meta["samples"] == 1600
    # Level metrics are the point of the sidecar: measure, do not guess.
    assert meta["peak"] == 1000
    # -30 dBFS is a healthy speech level for a full-scale digital signal.
    assert -40.0 < meta["rms_dbfs"] < -20.0


def test_recorder_never_blocks_the_audio_thread(tmp_path: Path) -> None:
    """A tiny queue must drop old work, not stall the caller or lose the newest."""
    recorder = SttAudioRecorder(tmp_path, sample_rate=16000, max_pending=1)
    recorder.start()
    try:
        for i in range(5):
            recorder.begin_command(f"run-{i}")
            recorder.tap(np.full(100, i, dtype="<i2").tobytes())
            recorder.end_command(f"run-{i}", transcript="t")
        recorder.flush(timeout=2.0)
    finally:
        recorder.stop()

    # The newest command is always retained; older ones may be dropped under
    # backpressure rather than blocking capture.
    last = tmp_path / "commands" / "run-4.wav"
    assert last.is_file(), "newest command must survive queue backpressure"


def test_retention_prunes_by_count(tmp_path: Path) -> None:
    recorder = SttAudioRecorder(tmp_path, sample_rate=16000, max_commands=3)
    recorder.start()
    try:
        for i in range(6):
            recorder.begin_command(f"retain-{i}")
            recorder.tap(np.zeros(10, dtype="<i2").tobytes())
            recorder.end_command(f"retain-{i}", transcript="t")
        recorder.flush(timeout=2.0)
    finally:
        recorder.stop()

    remaining = sorted(p.stem for p in (tmp_path / "commands").glob("*.wav"))
    assert len(remaining) == 3
    assert remaining == ["retain-3", "retain-4", "retain-5"]


def test_disabled_recorder_is_a_noop(tmp_path: Path) -> None:
    recorder = SttAudioRecorder(tmp_path, sample_rate=16000, enabled=False)
    recorder.start()
    try:
        recorder.begin_command("run-x")
        recorder.tap(np.zeros(10, dtype="<i2").tobytes())
        path = recorder.end_command("run-x", transcript="t")
    finally:
        recorder.stop()
    assert path is None
    assert not (tmp_path / "commands").exists() or not list((tmp_path / "commands").glob("*.wav"))


def test_tap_outside_a_command_is_ignored(tmp_path: Path) -> None:
    recorder = SttAudioRecorder(tmp_path, sample_rate=16000)
    recorder.start()
    try:
        recorder.tap(np.zeros(10, dtype="<i2").tobytes())
        recorder.flush(timeout=1.0)
    finally:
        recorder.stop()
    assert not list((tmp_path / "commands").glob("*.wav"))
