"""Tests for satellite/eval/run.py CLI."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_EVAL_RUN = Path(__file__).resolve().parent / "run.py"
_spec = importlib.util.spec_from_file_location("satellite_eval_run", _EVAL_RUN)
eval_run = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(eval_run)


def test_eval_run_main_skips_missing_audio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    (eval_root / "cases.json").write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )
    model_path = tmp_path / "sayso.onnx"
    model_path.write_bytes(b"fake-onnx")

    mock_model = MagicMock()
    mock_model.predict.return_value = {"sayso": 0.0}
    fake_wakeword = MagicMock(WakeWordModel=MagicMock(return_value=mock_model))
    monkeypatch.setitem(sys.modules, "livekit", MagicMock(wakeword=fake_wakeword))
    monkeypatch.setitem(sys.modules, "livekit.wakeword", fake_wakeword)

    output = tmp_path / "report.json"
    rc = eval_run.main(
        [
            "--model",
            str(model_path),
            "--eval-root",
            str(eval_root),
            "--output",
            str(output),
        ]
    )
    assert rc == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["skipped"] == 1
    assert output.is_file()


def test_eval_run_main_strict_fails_on_missing_audio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    (eval_root / "cases.json").write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )
    model_path = tmp_path / "sayso.onnx"
    model_path.write_bytes(b"fake-onnx")

    mock_model = MagicMock()
    mock_model.predict.return_value = {"sayso": 0.0}
    fake_wakeword = MagicMock(WakeWordModel=MagicMock(return_value=mock_model))
    monkeypatch.setitem(sys.modules, "livekit", MagicMock(wakeword=fake_wakeword))
    monkeypatch.setitem(sys.modules, "livekit.wakeword", fake_wakeword)

    rc = eval_run.main(
        [
            "--model",
            str(model_path),
            "--eval-root",
            str(eval_root),
            "--strict",
            "--refractory-seconds",
            "2.0",
        ]
    )
    assert rc == 1


def test_detect_hardware_darwin_arm64_is_not_pi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eval_run.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(eval_run.sys, "platform", "darwin")
    assert eval_run._detect_hardware() != "pi"


def test_detect_hardware_linux_arm_is_pi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eval_run.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(eval_run.sys, "platform", "linux")
    assert eval_run._detect_hardware() == "pi"


def test_load_wake_defaults_falls_back_to_deployed_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_config() -> None:
        raise RuntimeError("config unavailable")

    monkeypatch.setattr("satellite.sayso.config.load_config", _raise_config)
    threshold, refractory = eval_run._load_wake_defaults()
    assert threshold == eval_run.DEPLOYED_WAKE_THRESHOLD
    assert threshold != 0.65
    assert refractory == eval_run.DEFAULT_REFRACTORY_SECONDS


def test_eval_run_main_strict_fails_on_missing_cases(tmp_path) -> None:
    model_path = tmp_path / "sayso.onnx"
    model_path.write_bytes(b"fake-onnx")

    rc = eval_run.main(
        ["--model", str(model_path), "--eval-root", str(tmp_path), "--strict"]
    )
    assert rc == 1
