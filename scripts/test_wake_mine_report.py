"""Colocated tests for scripts/wake_mine_report.py."""

from __future__ import annotations

import json
import shutil
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import wake_mine_report  # noqa: E402


def _write_window_wav(path: Path, seed: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    samples = (rng.standard_normal(32000) * 1000).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(samples.tobytes())


def _write_record(spool: Path, capture_id: str, *, seed: int = 0) -> Path:
    record_dir = spool / "records" / capture_id
    window = record_dir / "window.wav"
    _write_window_wav(window, seed)
    meta = {
        "capture_id": capture_id,
        "score": 0.5,
        "sampling_reason": "detection",
        "hashes": {"window_sha256": wake_mine_report._sha256_file(window)},
    }
    (record_dir / "record.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return window


def test_inventory_cleanup_keeps_published_window_wav(tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    window = _write_record(spool, "capture-a", seed=1)
    entries = wake_mine_report.inventory_wake_assets([spool], cleanup=True)
    assert window.is_file()
    keep = [entry for entry in entries if entry.get("path") == str(window)]
    assert keep
    assert all(entry.get("action") != "remove" for entry in keep)


def test_duplicate_cleanup_keeps_retained_copy(tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    kept = _write_record(spool, "capture-a", seed=2)
    duplicate = spool / "z_legacy_dup.wav"
    shutil.copy2(kept, duplicate)
    entries = wake_mine_report.inventory_wake_assets([spool], cleanup=True)
    assert kept.is_file()
    assert not duplicate.exists()
    removed = [entry for entry in entries if entry.get("action") == "remove"]
    assert removed
    assert all("retained_copy=" in str(entry.get("detail", "")) for entry in removed)


def test_duplicate_cleanup_keeps_eval_fixture_over_spool_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_audio = tmp_path / "eval" / "audio"
    eval_audio.mkdir(parents=True)
    eval_fixture = eval_audio / "positive_sayso_near.wav"
    _write_window_wav(eval_fixture, seed=42)

    spool = tmp_path / "spool"
    spool_copy = _write_record(spool, "capture-dup", seed=42)
    assert wake_mine_report._sha256_file(eval_fixture) == wake_mine_report._sha256_file(spool_copy)

    monkeypatch.setattr(
        wake_mine_report,
        "pinned_wake_roots",
        lambda: [eval_audio],
    )
    entries = wake_mine_report.inventory_wake_assets([spool, eval_audio], cleanup=True)
    assert eval_fixture.is_file()
    assert not spool_copy.is_file()
    removed = [entry for entry in entries if entry.get("action") == "remove"]
    assert removed
    assert all(Path(entry["path"]) != eval_fixture for entry in removed)


def test_rglob_does_not_double_index_records_window(tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    window = _write_record(spool, "capture-a", seed=3)
    entries = wake_mine_report._scan_wake_roots([spool])
    window_entries = [entry for entry in entries if entry.get("path") == str(window)]
    assert len(window_entries) == 1
