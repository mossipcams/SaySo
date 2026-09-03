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
