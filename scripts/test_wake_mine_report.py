"""Tests for scripts/wake_mine_report.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import wake_mine_report as report  # noqa: E402


def _sidecar(spool: Path, stem: str, score: float, *, fired: bool = False) -> None:
    # load() walks *.wav and requires a sidecar, so both must exist.
    (spool / f"{stem}.wav").write_bytes(b"")
    (spool / f"{stem}.json").write_text(
        json.dumps({"score": score, "fired": fired, "label": None, "captured_utc": "20260913T153857"}),
        encoding="utf-8",
    )


def _ledger(tmp_path: Path) -> Path:
    path = tmp_path / "ledger.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "clusters": [
                    {
                        "index": 0,
                        "label": "positive",
                        "detail": "positive_isolated",
                        "review": "suggested",
                        "transcripts": ["Say so."],
                        "stems": ["a_000_s0.7000", "a_160_s0.6500"],
                    },
                    {
                        "index": 1,
                        "label": "negative",
                        "detail": "negative_preceded",
                        "review": "listen",
                        "transcripts": ["assistant tools say so"],
                        "stems": ["b_000_s0.2500"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _spool(tmp_path: Path) -> Path:
    spool = tmp_path / "spool"
    spool.mkdir()
    _sidecar(spool, "a_000_s0.7000", 0.70, fired=True)
    _sidecar(spool, "a_160_s0.6500", 0.65, fired=True)
    _sidecar(spool, "b_000_s0.2500", 0.25)
    return spool


def test_load_ledger_maps_every_window_to_its_cluster(tmp_path: Path) -> None:
    ledger = report.load_ledger(_ledger(tmp_path))
    assert ledger["a_000_s0.7000"]["detail"] == "positive_isolated"
    assert ledger["a_160_s0.6500"]["label"] == "positive"
    assert ledger["b_000_s0.2500"]["label"] == "negative"


def test_summarise_ledger_counts_clusters_not_windows(tmp_path: Path, capsys) -> None:
    report.summarise_ledger(report.load_ledger(_ledger(tmp_path)))
    out = capsys.readouterr().out
    assert "1 positive" in out
    assert "1 negative" in out
    assert "1 clusters need a listen" in out


def test_review_only_lists_clips_whose_cluster_needs_a_listen(tmp_path: Path, capsys) -> None:
    spool = _spool(tmp_path)
    rc = report.main(
        [str(spool), "--ledger", str(_ledger(tmp_path)), "--review-only", "--play"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "b_000_s0.2500.wav" in out
    assert "a_000_s0.7000.wav" not in out
    assert "negative? (negative_preceded)" in out


def test_review_only_without_ledger_is_an_error(tmp_path: Path, capsys) -> None:
    assert report.main([str(_spool(tmp_path)), "--review-only"]) == 1
    assert "--review-only needs --ledger" in capsys.readouterr().err


def test_label_cluster_labels_every_window_of_the_utterance(tmp_path: Path, capsys) -> None:
    spool = _spool(tmp_path)
    rc = report.main(
        [str(spool), "--ledger", str(_ledger(tmp_path)), "--label-cluster", "a_160_s0.6500", "positive"]
    )
    assert rc == 0
    assert "a_000_s0.7000 -> positive" in capsys.readouterr().out

    labels = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))["label"]
        for path in sorted(spool.glob("*.json"))
    }
    # The verdict belongs to the utterance, so the neighbouring window gets it too.
    assert labels["a_000_s0.7000"] == "positive"
    assert labels["a_160_s0.6500"] == "positive"
    assert labels["b_000_s0.2500"] is None


def test_label_cluster_rejects_unknown_stem_and_bad_label(tmp_path: Path, capsys) -> None:
    spool = _spool(tmp_path)
    ledger = _ledger(tmp_path)
    assert report.main([str(spool), "--ledger", str(ledger), "--label-cluster", "zzz", "positive"]) == 1
    assert report.main([str(spool), "--ledger", str(ledger), "--label-cluster", "a_000_s0.7000", "maybe"]) == 1
    assert capsys.readouterr().err


@pytest.mark.parametrize("label", ["positive", "negative", "unsure"])
def test_label_single_window(tmp_path: Path, label: str) -> None:
    spool = _spool(tmp_path)
    assert report.main([str(spool), "--label", "b_000_s0.2500", label]) == 0
    meta = json.loads((spool / "b_000_s0.2500.json").read_text(encoding="utf-8"))
    assert meta["label"] == label


def _sidecar_meta(spool: Path, stem: str) -> dict:
    return json.loads((spool / f"{stem}.json").read_text(encoding="utf-8"))


def test_apply_suggestions_labels_every_window_but_holds_back_the_flagged_one(
    tmp_path: Path, capsys
) -> None:
    spool = _spool(tmp_path)
    rc = report.main([str(spool), "--ledger", str(_ledger(tmp_path)), "--apply-suggestions"])
    assert rc == 0

    # A decidable cluster labels all of its windows, and records what was heard.
    for stem in ("a_000_s0.7000", "a_160_s0.6500"):
        meta = _sidecar_meta(spool, stem)
        assert meta["label"] == "positive"
        assert meta["transcript"] == "Say so."
        assert "positive_isolated" in meta["notes"]

    # The carrier cluster is decided by no transcript, so it stays unlabelled and
    # carries the suggestion in notes only.
    flagged = _sidecar_meta(spool, "b_000_s0.2500")
    assert flagged["label"] is None
    assert "needs a listen" in flagged["notes"]
    assert flagged["transcript"] == "assistant tools say so"

    out = capsys.readouterr().out
    assert "labelled        2 windows" in out
    assert "suggested only  1 windows" in out


def test_apply_suggestions_include_review_labels_the_flagged_cluster(tmp_path: Path) -> None:
    spool = _spool(tmp_path)
    rc = report.main(
        [str(spool), "--ledger", str(_ledger(tmp_path)), "--apply-suggestions", "--include-review"]
    )
    assert rc == 0
    meta = _sidecar_meta(spool, "b_000_s0.2500")
    assert meta["label"] == "negative"
    assert "needs a listen" in meta["notes"]


def test_apply_suggestions_never_overwrites_a_reviewed_label(tmp_path: Path, capsys) -> None:
    spool = _spool(tmp_path)
    report.main([str(spool), "--label", "a_000_s0.7000", "negative"])
    capsys.readouterr()

    report.main([str(spool), "--ledger", str(_ledger(tmp_path)), "--apply-suggestions"])
    # Review outranks the rule: the human verdict survives and is reported.
    assert _sidecar_meta(spool, "a_000_s0.7000")["label"] == "negative"
    assert "left alone      1 windows" in capsys.readouterr().out


def test_apply_suggestions_reports_windows_the_spool_no_longer_has(tmp_path: Path, capsys) -> None:
    spool = _spool(tmp_path)
    (spool / "a_160_s0.6500.json").unlink()
    report.main([str(spool), "--ledger", str(_ledger(tmp_path)), "--apply-suggestions"])
    assert "not in spool    1 ledger windows" in capsys.readouterr().out


def test_apply_suggestions_needs_a_ledger(tmp_path: Path, capsys) -> None:
    assert report.main([str(_spool(tmp_path)), "--apply-suggestions"]) == 1
    assert "--apply-suggestions needs --ledger" in capsys.readouterr().err
