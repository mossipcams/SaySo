"""Colocated tests for wake recording session ingest."""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from satellite.sayso.wake.eval import write_synthetic_wav
from satellite.sayso.wake.livekit import SAMPLE_RATE
from satellite.sayso.wake.sessions import (
    DEFAULT_SHIP_REMOTE,
    DEFAULT_SHIP_REMOTE_CORPUS,
    ShipSessionError,
    derive_session_id,
    ingest_session,
    list_sessions,
    load_session,
    session_dir,
    ship_session,
    verify_session,
)


def _long_form_wav(path: Path, *, seconds: float = 3.0) -> None:
    samples = np.zeros(int(SAMPLE_RATE * seconds), dtype="<i2")
    samples[SAMPLE_RATE:] = 5000
    write_synthetic_wav(path, samples)


def test_ingest_long_form_session(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    wav = tmp_path / "living_room.wav"
    _long_form_wav(wav, seconds=4.0)
    session = ingest_session(wav, corpus, session_id="living_room_test")
    assert session.session_id == "living_room_test"
    assert session.sample_rate == SAMPLE_RATE
    assert session.duration_seconds == pytest.approx(4.0)
    assert session.audio_path.is_file()
    reloaded = load_session(corpus, "living_room_test")
    assert reloaded.sample_count == session.sample_count
    ok, message = verify_session(reloaded)
    assert ok, message


def test_derive_session_id_is_stable(tmp_path: Path) -> None:
    wav = tmp_path / "clip.wav"
    _long_form_wav(wav)
    assert derive_session_id(wav) == derive_session_id(wav)


def test_list_sessions_skips_invalid_entries(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    wav = tmp_path / "a.wav"
    _long_form_wav(wav)
    ingest_session(wav, corpus, session_id="good")
    bad = corpus / "sessions" / "broken"
    bad.mkdir(parents=True)
    (bad / "session.json").write_text("{", encoding="utf-8")
    sessions = list_sessions(corpus)
    assert [session.session_id for session in sessions] == ["good"]


def test_ingest_rejects_too_short_wav(tmp_path: Path) -> None:
    wav = tmp_path / "short.wav"
    write_synthetic_wav(wav, np.zeros(SAMPLE_RATE // 2, dtype="<i2"))
    with pytest.raises(ValueError, match="too short"):
        ingest_session(wav, tmp_path / "corpus")


def test_ingest_writes_canonical_16khz_audio(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    source_rate = 48000
    seconds = 3.0
    source_samples = np.zeros(int(source_rate * seconds), dtype="<i2")
    source_samples[source_rate : source_rate + SAMPLE_RATE] = 6000
    source_wav = tmp_path / "source_48k.wav"
    write_synthetic_wav(source_wav, source_samples, sample_rate=source_rate)

    session = ingest_session(source_wav, corpus, session_id="resample_test")
    assert session.sample_rate == SAMPLE_RATE
    assert session.sample_count == int(seconds * SAMPLE_RATE)
    assert session.duration_seconds == pytest.approx(seconds)
    assert session.source_path == str(source_wav.resolve())

    with wave.open(str(session.audio_path), "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE
        assert wf.getnframes() == session.sample_count

    ok, message = verify_session(session)
    assert ok, message


def _mock_ship_subprocess(session_hash: str):
    def fake_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        if cmd[0] == "rsync":
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[0] == "ssh":
            remote_cmd = cmd[2] if len(cmd) > 2 else ""
            if "sha256sum" in remote_cmd:
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    f"{session_hash}  /remote/audio.wav\n",
                    "",
                )
            return subprocess.CompletedProcess(cmd, 0, "", "")
        raise AssertionError(f"unexpected command: {cmd}")

    return fake_run


def test_ship_session_success_deletes_local(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    wav = tmp_path / "room.wav"
    _long_form_wav(wav)
    session = ingest_session(wav, corpus, session_id="room_a")
    local_dir = session_dir(corpus, "room_a")
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        calls.append(cmd)
        return _mock_ship_subprocess(session.audio_sha256)(cmd, **kwargs)

    result = ship_session(
        corpus,
        "room_a",
        remote_corpus=DEFAULT_SHIP_REMOTE_CORPUS,
        subprocess_run=fake_run,
    )
    assert result.deleted_local is True
    assert result.dry_run is False
    assert not local_dir.exists()
    assert len(calls) == 3
    assert calls[0][:2] == ["ssh", DEFAULT_SHIP_REMOTE]
    assert "mkdir -p" in calls[0][2]
    assert calls[1][0] == "rsync"
    assert calls[1][-1] == (
        f"{DEFAULT_SHIP_REMOTE}:{DEFAULT_SHIP_REMOTE_CORPUS}/sessions/room_a/"
    )
    assert calls[2][:2] == ["ssh", DEFAULT_SHIP_REMOTE]


def test_ship_session_verify_fail_keeps_local(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    wav = tmp_path / "room.wav"
    _long_form_wav(wav)
    ingest_session(wav, corpus, session_id="room_a")
    local_dir = session_dir(corpus, "room_a")
    wrong_hash = "0" * 64

    with pytest.raises(ShipSessionError, match="hash mismatch"):
        ship_session(corpus, "room_a", subprocess_run=_mock_ship_subprocess(wrong_hash))
    assert local_dir.is_dir()


def test_ship_session_dry_run_no_copy_no_delete(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    wav = tmp_path / "room.wav"
    _long_form_wav(wav)
    ingest_session(wav, corpus, session_id="room_a")
    local_dir = session_dir(corpus, "room_a")
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    result = ship_session(corpus, "room_a", dry_run=True, subprocess_run=fake_run)
    assert result.dry_run is True
    assert result.deleted_local is False
    assert local_dir.is_dir()
    assert calls == []
