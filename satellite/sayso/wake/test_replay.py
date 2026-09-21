"""Colocated tests for production wake session replay."""

from __future__ import annotations

import json
import random
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from satellite.sayso.wake.eval import write_synthetic_wav
from satellite.sayso.wake.livekit import HOP_SAMPLES, SAMPLE_RATE, WINDOW_SAMPLES, LiveKitWakeWordProvider
from satellite.sayso.wake.mining import HardNegativeMiner, SAMPLING_DETECTION
from satellite.sayso.wake.corpus import load_events
from satellite.sayso.wake.replay import (
    ReplayConfig,
    production_replay_constants,
    replay_and_import_session,
    replay_session,
    replay_session_to_spool,
)
from satellite.sayso.wake.sessions import RecordingSession, ingest_session


def _install_fake_livekit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> MagicMock:
    model_path = tmp_path / "sayso.onnx"
    model_path.write_bytes(b"fake-onnx")
    mock_model = MagicMock()
    mock_model.predict.return_value = {"sayso": 0.0}
    fake_wakeword = MagicMock(WakeWordModel=MagicMock(return_value=mock_model))
    monkeypatch.setitem(sys.modules, "livekit", MagicMock(wakeword=fake_wakeword))
    monkeypatch.setitem(sys.modules, "livekit.wakeword", fake_wakeword)
    return mock_model


def test_production_replay_constants_match_livekit() -> None:
    constants = production_replay_constants()
    assert constants == {
        "sample_rate": SAMPLE_RATE,
        "window_samples": WINDOW_SAMPLES,
        "hop_samples": HOP_SAMPLES,
    }


def test_replay_uses_production_window_and_hop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    wav = tmp_path / "session.wav"
    span = WINDOW_SAMPLES + HOP_SAMPLES * 3
    samples = np.zeros(span, dtype="<i2")
    samples[WINDOW_SAMPLES : WINDOW_SAMPLES + HOP_SAMPLES] = 9000
    write_synthetic_wav(wav, samples)
    session = ingest_session(wav, corpus, session_id="replay_test")
    mock_model = _install_fake_livekit(monkeypatch, tmp_path)
    provider = LiveKitWakeWordProvider(
        model_path=tmp_path / "sayso.onnx",
        phrase="SaySo",
        threshold=0.5,
        refractory_seconds=0.0,
    )

    def predict(window: np.ndarray, sample_index: int | None = None):
        if int(np.max(np.abs(window))) >= 9000:
            mock_model.predict.return_value = {"sayso": 0.99}
            return SimpleNamespace(confidence=0.99, phrase="SaySo", timestamp=0.0, sample_index=sample_index)
        mock_model.predict.return_value = {"sayso": 0.0}
        return None

    stats = replay_session(session, provider, predict=predict)
    assert stats.windows_scored >= 2
    assert stats.detections >= 1


def test_replay_mines_candidate_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    wav = tmp_path / "session.wav"
    span = WINDOW_SAMPLES + HOP_SAMPLES * 2
    samples = np.zeros(span, dtype="<i2")
    samples[WINDOW_SAMPLES : WINDOW_SAMPLES + HOP_SAMPLES] = 9000
    write_synthetic_wav(wav, samples)
    session = ingest_session(wav, corpus, session_id="mine_test")
    mock_model = _install_fake_livekit(monkeypatch, tmp_path)
    provider = LiveKitWakeWordProvider(
        model_path=tmp_path / "sayso.onnx",
        phrase="SaySo",
        threshold=0.5,
        refractory_seconds=0.0,
    )
    spool = tmp_path / "spool"
    miner = HardNegativeMiner(
        spool,
        mine_threshold=0.1,
        detect_threshold=0.5,
        below_sample_rate=0.0,
        session_id=session.session_id,
        rng=random.Random(0),
    )
    provider._miner = miner
    miner.start()

    original_predict = LiveKitWakeWordProvider.predict_window

    def patched_predict(self, window: np.ndarray, sample_index: int | None = None):
        if int(np.max(np.abs(window))) >= 9000:
            mock_model.predict.return_value = {"sayso": 0.99}
        else:
            mock_model.predict.return_value = {"sayso": 0.0}
        return original_predict(self, window, sample_index=sample_index)

    monkeypatch.setattr(LiveKitWakeWordProvider, "predict_window", patched_predict)

    stats = replay_session(session, provider, miner=miner)
    assert stats.mined_records >= 1
    record = next((spool / "records").iterdir())
    meta = json.loads((record / "record.json").read_text())
    assert meta["session_id"] == session.session_id
    assert meta["label"] is None
    assert meta["sampling_reason"] == SAMPLING_DETECTION


def test_replay_session_to_spool_mines_without_manual_miner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    wav = tmp_path / "session.wav"
    span = WINDOW_SAMPLES + HOP_SAMPLES * 2
    samples = np.zeros(span, dtype="<i2")
    samples[WINDOW_SAMPLES : WINDOW_SAMPLES + HOP_SAMPLES] = 9000
    write_synthetic_wav(wav, samples)
    session = ingest_session(wav, corpus, session_id="spool_mine_test")
    mock_model = _install_fake_livekit(monkeypatch, tmp_path)
    provider = LiveKitWakeWordProvider(
        model_path=tmp_path / "sayso.onnx",
        phrase="SaySo",
        threshold=0.5,
        refractory_seconds=0.0,
    )
    assert provider._miner is None

    original_predict = LiveKitWakeWordProvider.predict_window

    def patched_predict(self, window: np.ndarray, sample_index: int | None = None):
        if int(np.max(np.abs(window))) >= 9000:
            mock_model.predict.return_value = {"sayso": 0.99}
        else:
            mock_model.predict.return_value = {"sayso": 0.0}
        return original_predict(self, window, sample_index=sample_index)

    monkeypatch.setattr(LiveKitWakeWordProvider, "predict_window", patched_predict)

    spool = tmp_path / "spool"
    stats = replay_session_to_spool(
        session,
        provider,
        spool,
        mine_threshold=0.1,
        detect_threshold=0.5,
        below_sample_rate=0.0,
    )
    assert stats.mined_records >= 1
    record = next((spool / "records").iterdir())
    meta = json.loads((record / "record.json").read_text())
    assert meta["session_id"] == session.session_id


def test_second_replay_does_not_duplicate_session_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    wav = tmp_path / "session.wav"
    span = WINDOW_SAMPLES + HOP_SAMPLES * 2
    samples = np.zeros(span, dtype="<i2")
    samples[WINDOW_SAMPLES : WINDOW_SAMPLES + HOP_SAMPLES] = 9000
    write_synthetic_wav(wav, samples)
    session = ingest_session(wav, corpus, session_id="rerun_test")
    mock_model = _install_fake_livekit(monkeypatch, tmp_path)
    provider = LiveKitWakeWordProvider(
        model_path=tmp_path / "sayso.onnx",
        phrase="SaySo",
        threshold=0.5,
        refractory_seconds=0.0,
    )

    original_predict = LiveKitWakeWordProvider.predict_window

    def patched_predict(self, window: np.ndarray, sample_index: int | None = None):
        if int(np.max(np.abs(window))) >= 9000:
            mock_model.predict.return_value = {"sayso": 0.99}
        else:
            mock_model.predict.return_value = {"sayso": 0.0}
        return original_predict(self, window, sample_index=sample_index)

    monkeypatch.setattr(LiveKitWakeWordProvider, "predict_window", patched_predict)

    first_stats, first_imported = replay_and_import_session(
        corpus,
        session,
        provider,
        mine_threshold=0.1,
        detect_threshold=0.5,
        below_sample_rate=0.0,
    )
    second_stats, second_imported = replay_and_import_session(
        corpus,
        session,
        provider,
        mine_threshold=0.1,
        detect_threshold=0.5,
        below_sample_rate=0.0,
    )
    assert first_stats.mined_records >= 1
    assert second_stats.mined_records >= 1
    session_events = [event for event in load_events(corpus) if event.session_id == session.session_id]
    assert len(session_events) == len(first_imported)
    assert {event.event_id for event in session_events} == {event.event_id for event in second_imported}
