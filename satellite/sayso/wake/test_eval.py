"""Colocated tests for wake eval scoring and synthetic audio harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from satellite.sayso.wake.eval import (
    CHUNK_SAMPLES,
    WakeEvalCase,
    compute_latency_percentiles,
    detect_missing_first_word,
    evaluate_case,
    load_wake_cases,
    read_wav_pcm,
    run_wake_eval,
    scan_wake_audio,
    transcript_matches,
    write_synthetic_wav,
)
from satellite.sayso.wake.livekit import HOP_SAMPLES, WINDOW_SAMPLES, LiveKitWakeWordProvider


def test_compute_latency_percentiles() -> None:
    stats = compute_latency_percentiles([10.0, 20.0, 30.0, 40.0])
    assert stats["p50"] == 25.0
    assert stats["p95"] == 38.5


def test_detect_missing_first_word_and_transcript_match() -> None:
    assert detect_missing_first_word("turn off the kitchen", "off the kitchen")
    assert not detect_missing_first_word("turn off the kitchen", "turn off the kitchen")
    assert transcript_matches("turn off the kitchen", "sayso turn off the kitchen")


def test_read_wav_pcm_resamples_to_16k(tmp_path: Path) -> None:
    samples = np.zeros(8000, dtype="<i2")
    samples[100:200] = 5000
    path = tmp_path / "tmp.wav"
    write_synthetic_wav(path, samples, sample_rate=8000)
    pcm, rate = read_wav_pcm(path)
    from satellite.sayso.wake.livekit import SAMPLE_RATE

    assert rate == SAMPLE_RATE
    assert len(pcm) == 16000 * 2


def test_evaluate_case_skips_missing_audio(tmp_path: Path) -> None:
    case = WakeEvalCase(
        id="missing",
        category="positive_sayso",
        audio="audio/missing.wav",
        expect_detection=True,
    )
    provider = MagicMock()
    result = evaluate_case(case, tmp_path, provider)
    assert result.status == "skipped"
    assert result.skip_reason is not None


def test_scan_wake_audio_records_detection_and_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "sayso.onnx"
    model_path.write_bytes(b"fake-onnx")
    mock_model = MagicMock()
    mock_model.predict.return_value = {"sayso": 0.0}
    fake_wakeword = MagicMock(WakeWordModel=MagicMock(return_value=mock_model))
    monkeypatch.setitem(sys.modules, "livekit", MagicMock(wakeword=fake_wakeword))
    monkeypatch.setitem(sys.modules, "livekit.wakeword", fake_wakeword)

    provider = LiveKitWakeWordProvider(
        model_path=model_path,
        phrase="SaySo",
        threshold=0.5,
        refractory_seconds=0.0,
    )

    samples = np.zeros(WINDOW_SAMPLES + HOP_SAMPLES + CHUNK_SAMPLES, dtype="<i2")
    samples[WINDOW_SAMPLES:] = 8000

    def predict(window: np.ndarray) -> SimpleNamespace | None:
        if int(np.max(np.abs(window))) >= 8000:
            mock_model.predict.return_value = {"sayso": 0.99}
            return SimpleNamespace(confidence=0.99, phrase="SaySo", timestamp=0.0)
        mock_model.predict.return_value = {"sayso": 0.0}
        return None

    pcm = samples.tobytes()
    detected, inference_ms, detection_sample = scan_wake_audio(
        provider,
        pcm,
        predict=predict,
    )
    assert detected
    assert detection_sample is not None
    assert inference_ms
    assert compute_latency_percentiles(inference_ms)["p50"] >= 0.0


def test_run_wake_eval_skips_missing_corpus_audio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    cases = {
        "version": 1,
        "cases": [
            {
                "id": "pos",
                "category": "positive_sayso",
                "audio": "audio/positive.wav",
                "expect_detection": True,
            }
        ],
    }
    (eval_root / "cases.json").write_text(json.dumps(cases), encoding="utf-8")

    model_path = tmp_path / "sayso.onnx"
    model_path.write_bytes(b"fake-onnx")
    mock_model = MagicMock()
    mock_model.predict.return_value = {"sayso": 0.0}
    fake_wakeword = MagicMock(WakeWordModel=MagicMock(return_value=mock_model))
    monkeypatch.setitem(sys.modules, "livekit", MagicMock(wakeword=fake_wakeword))
    monkeypatch.setitem(sys.modules, "livekit.wakeword", fake_wakeword)

    report = run_wake_eval(model_path=model_path, eval_root=eval_root)
    assert report["summary"]["skipped"] == 1
    assert report["summary"]["failed"] == 0


def test_run_wake_eval_with_synthetic_positive_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    eval_root = tmp_path / "eval"
    audio_dir = eval_root / "audio"
    audio_dir.mkdir(parents=True)
    wav_path = audio_dir / "positive.wav"
    samples = np.zeros(WINDOW_SAMPLES + HOP_SAMPLES + CHUNK_SAMPLES, dtype="<i2")
    samples[WINDOW_SAMPLES:] = 9000
    write_synthetic_wav(wav_path, samples)

    cases = {
        "version": 1,
        "cases": [
            {
                "id": "pos",
                "category": "positive_sayso",
                "audio": "audio/positive.wav",
                "expect_detection": True,
            }
        ],
    }
    (eval_root / "cases.json").write_text(json.dumps(cases), encoding="utf-8")

    model_path = tmp_path / "sayso.onnx"
    model_path.write_bytes(b"fake-onnx")
    mock_model = MagicMock()
    mock_model.predict.return_value = {"sayso": 0.0}
    fake_wakeword = MagicMock(WakeWordModel=MagicMock(return_value=mock_model))
    monkeypatch.setitem(sys.modules, "livekit", MagicMock(wakeword=fake_wakeword))
    monkeypatch.setitem(sys.modules, "livekit.wakeword", fake_wakeword)

    original_predict = LiveKitWakeWordProvider.predict_window

    def patched_predict(self, window: np.ndarray):
        if int(np.max(np.abs(window))) >= 9000:
            mock_model.predict.return_value = {"sayso": 0.99}
        else:
            mock_model.predict.return_value = {"sayso": 0.0}
        return original_predict(self, window)

    monkeypatch.setattr(LiveKitWakeWordProvider, "predict_window", patched_predict)

    report = run_wake_eval(model_path=model_path, eval_root=eval_root, threshold=0.5)
    assert report["summary"]["passed"] == 1
    entry = report["results"][0]
    assert entry["detected"]
    assert entry["detection_ok"]
    assert "pi_inference_ms" in entry


def test_load_wake_cases_rejects_unknown_category(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "cases": [
                    {
                        "id": "bad",
                        "category": "unknown",
                        "audio": "audio/x.wav",
                        "expect_detection": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown wake eval category"):
        load_wake_cases(path)


def test_evaluate_case_speech_end_to_ack_uses_stt_delay_ms(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    eval_root = tmp_path / "eval"
    audio_dir = eval_root / "audio"
    fixture_dir = eval_root / "fixtures" / "stt"
    audio_dir.mkdir(parents=True)
    fixture_dir.mkdir(parents=True)
    wav_path = audio_dir / "continuous.wav"
    samples = np.zeros(WINDOW_SAMPLES + HOP_SAMPLES + CHUNK_SAMPLES, dtype="<i2")
    write_synthetic_wav(wav_path, samples)
    (fixture_dir / "continuous.json").write_text(
        json.dumps({"text": "turn off the kitchen", "stt_delay_ms": 1500.0, "speech_end_ms": 400.0}),
        encoding="utf-8",
    )

    case = WakeEvalCase(
        id="continuous",
        category="continuous_command",
        audio="audio/continuous.wav",
        expect_detection=True,
        command_transcript="turn off the kitchen",
        expected_transcript="turn off the kitchen",
        transcript_fixture="fixtures/stt/continuous.json",
    )

    model_path = tmp_path / "sayso.onnx"
    model_path.write_bytes(b"fake-onnx")
    mock_model = MagicMock()
    mock_model.predict.return_value = {"sayso": 0.0}
    fake_wakeword = MagicMock(WakeWordModel=MagicMock(return_value=mock_model))
    monkeypatch.setitem(sys.modules, "livekit", MagicMock(wakeword=fake_wakeword))
    monkeypatch.setitem(sys.modules, "livekit.wakeword", fake_wakeword)

    provider = LiveKitWakeWordProvider(
        model_path=model_path,
        phrase="SaySo",
        threshold=0.5,
        refractory_seconds=0.0,
    )
    provider.predict_window = MagicMock(return_value=SimpleNamespace(confidence=0.99, phrase="SaySo", timestamp=0.0))

    result = evaluate_case(case, eval_root, provider)
    assert result.speech_end_to_ack_ms == pytest.approx(1500.0)
