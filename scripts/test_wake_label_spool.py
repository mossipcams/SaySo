"""Tests for scripts/wake_label_spool.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import wake_label_spool as labeler  # noqa: E402


def _window(spool: Path, stem: str, score: float, *, fired: bool = False) -> None:
    (spool / f"{stem}.json").write_text(
        json.dumps(
            {
                "score": score,
                "fired": fired,
                "captured_utc": stem.split("_")[0],
                "label": None,
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        ("Say so.", labeler.ISOLATED),
        ("say so", labeler.ISOLATED),
        ('"Say so."', labeler.ISOLATED),
        ("Seizo.", labeler.ISOLATED),
        ("ZZO", labeler.ISOLATED),
        # A word after the phrase is the continuous-command shape, not a carrier.
        ("say so turn on the TV", labeler.ISOLATED),
        # ...but anything before it is.
        ("and say so turn off the kitchen", labeler.PRECEDED),
        ("if you say so", labeler.PRECEDED),
        ("assistant tools say so turn on the TV", labeler.PRECEDED),
        # Sentence boundary: only a listen can settle it, so it must not be a verdict.
        ("and tools. Say so. Turn on the little...", labeler.BOUNDARY),
        # Different words, including the ones the config lists as near-misses.
        ("say something", labeler.NO_PHRASE),
        ("says so", labeler.NO_PHRASE),
        ("Okay, so turn on the web.", labeler.NO_PHRASE),
        ("", labeler.NO_PHRASE),
        ("See you soon.", labeler.NEAR_MISS),
        ("Stay soon.", labeler.NEAR_MISS),
    ],
)
def test_classify_transcript_applies_the_isolation_rule(transcript: str, expected: str) -> None:
    assert labeler.classify_transcript(transcript) == expected


def _cluster(*transcripts: str, scores: list[float] | None = None) -> labeler.Cluster:
    score_list = scores or [0.5 + 0.01 * i for i in range(len(transcripts))]
    windows = [
        labeler.Window(stem=f"stem{i}", text=text, score=score, fired=False, time=float(i))
        for i, (text, score) in enumerate(zip(transcripts, score_list))
    ]
    return labeler.Cluster(index=0, windows=windows)


def test_cluster_preceding_word_wins_over_isolated_window() -> None:
    # The window that starts after the preceding word legitimately looks isolated,
    # so one carrier window decides the cluster.
    cluster = _cluster("Say so. Turn on the TV", "assistant tools say so turn on the TV")
    assert cluster.detail == labeler.PRECEDED
    assert cluster.label == labeler.NEGATIVE
    assert cluster.disagreement
    assert cluster.review == "listen"


def test_cluster_isolated_with_unrelated_window_still_positive_but_flagged() -> None:
    # One window heard the phrase and the other did not. The cluster is still a
    # positive, but that disagreement is exactly how a dropped carrier word
    # looks, so it must be reviewed rather than trusted.
    cluster = _cluster("Good boy.", "Say so.")
    assert cluster.detail == labeler.ISOLATED
    assert cluster.label == labeler.POSITIVE
    assert cluster.disagreement
    assert cluster.review == "listen"


def test_cluster_of_agreeing_windows_needs_no_listen() -> None:
    cluster = _cluster("Say so.", "say so", "Say so.")
    assert cluster.label == labeler.POSITIVE
    assert not cluster.disagreement
    assert cluster.review == "suggested"


def test_cluster_boundary_and_near_miss_need_a_listen() -> None:
    boundary = _cluster("and tools. Say so.")
    assert boundary.label == labeler.UNSURE
    assert boundary.review == "listen"

    near_miss = _cluster("Stay soon.")
    assert near_miss.label == labeler.NEGATIVE
    assert near_miss.review == "listen"


def test_cluster_worst_case_precedence_order() -> None:
    # BOUNDARY outranks ISOLATED but not PRECEDED; NEAR_MISS is last.
    assert _cluster("Say so.", "and tools. Say so.").detail == labeler.BOUNDARY
    assert _cluster("Stay soon.", "Say so.").detail == labeler.ISOLATED
    assert _cluster("Stay soon.", "Good boy.").detail == labeler.NEAR_MISS


def test_load_and_cluster_windows_orders_by_millisecond_field(tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    spool.mkdir()
    _window(spool, "20260913T153857_136_s0.2571", 0.2571)
    _window(spool, "20260913T153857_026_s0.3414", 0.3414)
    _window(spool, "20260913T153901_026_s0.1000", 0.1)
    (spool / "20260913T153902_000_s0.9.json").write_text("{not json", encoding="utf-8")

    windows = labeler.load_windows(spool, {})
    assert [w.stem for w in windows] == [
        "20260913T153857_026_s0.3414",
        "20260913T153857_136_s0.2571",
        "20260913T153901_026_s0.1000",
    ]

    clusters = labeler.cluster_windows(windows, gap_seconds=3.0)
    assert [len(c.windows) for c in clusters] == [2, 1]
    assert clusters[0].duration_s == pytest.approx(0.110)


def test_main_writes_a_ledger_and_never_labels_a_sidecar(tmp_path: Path, capsys) -> None:
    spool = tmp_path / "spool"
    spool.mkdir()
    _window(spool, "20260913T153857_000_s0.5170", 0.517, fired=True)
    _window(spool, "20260913T153857_160_s0.7400", 0.74, fired=True)
    _window(spool, "20260913T153910_000_s0.3100", 0.31)
    _window(spool, "20260913T153925_000_s0.7920", 0.792, fired=True)

    transcripts = {
        "20260913T153857_000_s0.5170": {"text": "Say so."},
        "20260913T153857_160_s0.7400": {"text": "say so"},
        "20260913T153910_000_s0.3100": {"text": "assistant tools say so"},
        "20260913T153925_000_s0.7920": {"text": "Stay so."},
    }
    transcripts_path = tmp_path / "transcripts.json"
    transcripts_path.write_text(json.dumps(transcripts), encoding="utf-8")
    ledger_path = tmp_path / "ledger.json"

    rc = labeler.main(
        [str(spool), "--transcripts", str(transcripts_path), "--ledger", str(ledger_path)]
    )
    assert rc == 0

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["cluster_count"] == 3
    assert ledger["summary"]["labels"]["positive"] == 1
    assert ledger["summary"]["labels"]["negative"] == 2
    assert ledger["summary"]["labels"]["unsure"] == 0
    # The combined cluster carries the carrier verdict, so the last window is the
    # only one left over: "Stay so." is a near-miss, not a phrase.
    assert ledger["summary"]["details"]["negative_preceded"] == 1
    assert ledger["summary"]["details"]["negative_no_phrase_near_miss"] == 1
    assert ledger["version"] == 1

    # Labelling is a listening decision; the ledger must not touch the sidecar.
    for sidecar in spool.glob("*.json"):
        assert json.loads(sidecar.read_text(encoding="utf-8"))["label"] is None

    assert "needs listening" not in capsys.readouterr().out


def test_main_rejects_missing_spool(tmp_path: Path) -> None:
    assert labeler.main([str(tmp_path / "nope")]) == 1
