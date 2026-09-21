"""Colocated tests for scripts/wake_corpus.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import wake_corpus  # noqa: E402
from satellite.sayso.wake.eval import write_synthetic_wav  # noqa: E402
from satellite.sayso.wake.livekit import SAMPLE_RATE  # noqa: E402


def _session_wav(path: Path) -> None:
    samples = np.zeros(SAMPLE_RATE * 2, dtype="<i2")
    samples[SAMPLE_RATE:] = 4000
    write_synthetic_wav(path, samples)


def test_constants_command(tmp_path: Path) -> None:
    rc = wake_corpus.main(["--corpus", str(tmp_path / "corpus"), "constants"])
    assert rc == 0


def test_ingest_and_list_sessions(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    wav = tmp_path / "room.wav"
    _session_wav(wav)
    rc = wake_corpus.main(["--corpus", str(corpus), "ingest", str(wav), "--session-id", "room_a"])
    assert rc == 0
    sessions = corpus / "sessions" / "room_a" / "session.json"
    assert sessions.is_file()


def test_ship_dry_run(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    wav = tmp_path / "room.wav"
    _session_wav(wav)
    assert wake_corpus.main(["--corpus", str(corpus), "ingest", str(wav), "--session-id", "room_a"]) == 0
    session_dir = corpus / "sessions" / "room_a"
    assert session_dir.is_dir()
    rc = wake_corpus.main(
        [
            "--corpus",
            str(corpus),
            "ship",
            "room_a",
            "--dry-run",
            "--remote",
            "ubuntu@192.168.1.140",
            "--remote-corpus",
            "/home/ubuntu/sayso-wake-data/corpus",
        ]
    )
    assert rc == 0
    assert session_dir.is_dir()


def test_split_and_snapshot(tmp_path: Path, monkeypatch) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    from satellite.sayso.wake.corpus import import_spool_record
    from satellite.sayso.wake.livekit import WINDOW_SAMPLES

    spool = tmp_path / "spool"
    record_dir = spool / "records" / "evt-1"
    record_dir.mkdir(parents=True)
    write_synthetic_wav(record_dir / "window.wav", np.full(WINDOW_SAMPLES, 1000, dtype="<i2"))
    (record_dir / "record.json").write_text(
        json.dumps(
            {
                "capture_id": "evt-1",
                "session_id": "s1",
                "score": 0.4,
                "sampling_reason": "near_threshold",
                "sample_start": 0,
                "sample_end": WINDOW_SAMPLES,
                "label": "positive",
                "hashes": {},
            }
        ),
        encoding="utf-8",
    )
    import_spool_record(record_dir, corpus)
    assert wake_corpus.main(["--corpus", str(corpus), "split", "--seed", "5"]) == 0
    assert wake_corpus.main(["--corpus", str(corpus), "snapshot", "--seed", "5"]) == 0
    manifest = next((corpus / "snapshots").glob("*/corpus_snapshot.json"))
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["examples"]


def test_ship_command_success_deletes_local(tmp_path: Path, monkeypatch) -> None:
    corpus = tmp_path / "corpus"
    wav = tmp_path / "room.wav"
    _session_wav(wav)
    assert wake_corpus.main(["--corpus", str(corpus), "ingest", str(wav), "--session-id", "ship_cli"]) == 0
    local_dir = corpus / "sessions" / "ship_cli"
    session = json.loads((local_dir / "session.json").read_text(encoding="utf-8"))

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["ssh", "ubuntu@192.168.1.140"] and "sha256sum" in cmd[2]:
            return type("Completed", (), {"returncode": 0, "stdout": f"{session['audio_sha256']}  audio.wav\n", "stderr": ""})()
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("satellite.sayso.wake.sessions.subprocess.run", fake_run)
    rc = wake_corpus.main(
        [
            "--corpus",
            str(corpus),
            "ship",
            "ship_cli",
            "--remote",
            "ubuntu@192.168.1.140",
            "--remote-corpus",
            "/home/ubuntu/sayso-wake-data/corpus",
        ]
    )
    assert rc == 0
    assert not local_dir.exists()
    assert calls


def test_ship_command_dry_run_keeps_local(tmp_path: Path, monkeypatch) -> None:
    corpus = tmp_path / "corpus"
    wav = tmp_path / "room.wav"
    _session_wav(wav)
    assert wake_corpus.main(["--corpus", str(corpus), "ingest", str(wav), "--session-id", "ship_dry_cli"]) == 0
    local_dir = corpus / "sessions" / "ship_dry_cli"

    def fake_run(cmd, **kwargs):
        raise AssertionError(f"unexpected subprocess: {cmd}")

    monkeypatch.setattr("satellite.sayso.wake.sessions.subprocess.run", fake_run)
    rc = wake_corpus.main(["--corpus", str(corpus), "ship", "ship_dry_cli", "--dry-run"])
    assert rc == 0
    assert local_dir.is_dir()
