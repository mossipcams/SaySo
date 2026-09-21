"""Colocated tests for wake recording session ingest."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from satellite.sayso.wake.eval import write_synthetic_wav
from satellite.sayso.wake.livekit import SAMPLE_RATE
from satellite.sayso.wake.sessions import (
    derive_session_id,
    ingest_session,
    list_sessions,
    load_session,
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
