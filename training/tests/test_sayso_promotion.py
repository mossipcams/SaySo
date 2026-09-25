"""Checks the manual model-promotion escape hatch."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOADER = importlib.machinery.SourceFileLoader("sayso_cli", str(ROOT / "sayso"))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
SAYSO = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(SAYSO)


def test_manual_promotion_keeps_failed_eval_gate(tmp_path: Path, monkeypatch) -> None:
    record = {
        "run_id": "run-1",
        "trained": True,
        "checkpoint": "/models/checkpoint",
        "dataset_sha256": "dataset-hash",
        "git_commit": "commit",
        "config": "config.yaml",
        "final_eval_passed": False,
        "final_metrics": {"promotion": {"passed": False, "overall_pass_rate": 0.5}},
    }
    monkeypatch.setattr(SAYSO, "STATE", tmp_path)
    monkeypatch.setattr(SAYSO, "current", lambda: record)

    try:
        SAYSO.cmd_promote_model(argparse.Namespace(manual_override=False, reason=None))
    except SystemExit as err:
        assert "passing final evaluation" in str(err)
    else:
        raise AssertionError("failed evaluation was promoted without an override")

    try:
        SAYSO.cmd_promote_model(argparse.Namespace(manual_override=True, reason="  "))
    except SystemExit as err:
        assert "non-empty --reason" in str(err)
    else:
        raise AssertionError("manual promotion accepted an empty reason")

    SAYSO.cmd_promote_model(argparse.Namespace(manual_override=True, reason="Compared with champion"))
    promotion = json.loads((tmp_path / "promoted_model.json").read_text())
    assert promotion["promoted"] is True
    assert promotion["promotion_mode"] == "manual_override"
    assert promotion["manual_override_reason"] == "Compared with champion"
    assert promotion["final_eval_passed"] is False
    assert promotion["eval_metrics"] == record["final_metrics"]
