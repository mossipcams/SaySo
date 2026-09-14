#!/usr/bin/env python3
"""Summarise and label the wake hard-negative spool.

Clips land unlabelled: the satellite knows a window scored high, not whether
anyone actually said "SaySo". Labelling is a listening job, and this script is
the thin wrapper around it.

    # what's in the spool
    python scripts/wake_mine_report.py /var/lib/sayso-satellite/wake-mining

    # highest-scoring unreviewed clips first, with a play command per row
    python scripts/wake_mine_report.py SPOOL --unreviewed --play

    # same, with the isolation rule's verdict per row and only the rows that
    # need a human ear
    python scripts/wake_label_spool.py SPOOL --transcripts T.json --ledger L.json \
        --review-only
    python scripts/wake_mine_report.py SPOOL --ledger L.json --review-only --play

    # record a verdict
    python scripts/wake_mine_report.py SPOOL --label 20260910T001432_412_s0.3120 negative

    # a verdict is about an utterance, not a window: label every window of the
    # cluster the ledger built around this stem
    python scripts/wake_mine_report.py SPOOL --ledger L.json \
        --label-cluster 20260910T001432_412_s0.3120 negative

Deliberately not doing clustering or dedup here. Both need the score
distribution of an actual corpus to calibrate against. Clustering is done by
`wake_label_spool.py`, which owns the isolation rule and writes the ledger this
script reads; this script reports and records verdicts, and never guesses one
from the score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LABELS = ("positive", "negative", "unsure")


def load(spool: Path) -> list[dict]:
    rows = []
    for wav in sorted(spool.glob("*.wav")):
        sidecar = wav.with_suffix(".json")
        if not sidecar.is_file():
            continue
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"skipping malformed sidecar: {sidecar}", file=sys.stderr)
            continue
        meta["_wav"] = wav
        meta["_json"] = sidecar
        rows.append(meta)
    return rows


def summarise(rows: list[dict]) -> None:
    if not rows:
        print("Spool is empty. Mining writes a clip only when a window clears "
              "wake_word.mine_threshold; a quiet room can go hours.")
        return
    fired = sum(1 for r in rows if r.get("fired"))
    labelled = [r for r in rows if r.get("label")]
    scores = sorted(float(r["score"]) for r in rows)
    print(f"clips           {len(rows)}")
    print(f"  fired         {fired}   (>= detect threshold; the rest are near-misses)")
    print(f"  near-miss     {len(rows) - fired}")
    print(f"  labelled      {len(labelled)} / {len(rows)}")
    for label in LABELS:
        n = sum(1 for r in labelled if r.get("label") == label)
        if n:
            print(f"    {label:<10}  {n}")
    print(f"score  min {scores[0]:.4f}  p50 {scores[len(scores) // 2]:.4f}  max {scores[-1]:.4f}")


def load_ledger(path: Path) -> dict[str, dict]:
    """Map every window stem to the verdict its cluster carries.

    The ledger is cluster-scoped and the sidecar is window-scoped, so one
    utterance becomes N identical suggestions. That is the point: a verdict is
    about what was said, and every window of that utterance shows the same thing.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read ledger {path}: {exc}", file=sys.stderr)
        raise SystemExit(1)
    return {
        stem: cluster
        for cluster in payload.get("clusters", [])
        for stem in cluster.get("stems", [])
    }


def summarise_ledger(ledger: dict[str, dict]) -> None:
    clusters = {id(cluster): cluster for cluster in ledger.values()}
    if not clusters:
        return
    counts = {label: 0 for label in LABELS}
    for cluster in clusters.values():
        label = cluster.get("label", "unsure")
        counts[label] = counts.get(label, 0) + 1
    listening = sum(1 for cluster in clusters.values() if cluster.get("review") == "listen")
    parts = "  ".join(f"{counts[label]} {label}" for label in LABELS if counts.get(label))
    print(f"suggested     {parts}  ({listening} clusters need a listen)")


def write_label(spool: Path, stems: list[str], label: str) -> int:
    written = 0
    for stem in stems:
        sidecar = spool / f"{stem}.json"
        if not sidecar.is_file():
            print(f"No sidecar for stem {stem}", file=sys.stderr)
            return 1
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        meta["label"] = label
        sidecar.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        print(f"{stem} -> {label}")
        written += 1
    return 0 if written else 1


def _clusters(ledger: dict[str, dict]) -> list[dict]:
    return list({id(cluster): cluster for cluster in ledger.values()}.values())


def apply_suggestions(spool: Path, ledger: dict[str, dict], *, include_review: bool) -> int:
    """Write the rule's verdict onto the sidecars, cluster by cluster.

    A cluster the rule flagged for a listen gets its suggestion recorded in
    ``notes`` and its ``label`` left null: the transcript cannot settle it, and
    a machine guess is not a human verdict. A sidecar that already carries a
    label is never overwritten -- review outranks the rule.
    """
    labelled = 0
    suggested_only = 0
    preserved = 0
    missing = 0

    for cluster in _clusters(ledger):
        detail = cluster.get("detail", "?")
        review = cluster.get("review", "listen")
        transcripts = " | ".join(cluster.get("transcripts", []))
        note = f"isolation rule verdict: {detail}"
        decidable = review == "suggested" or include_review
        if review != "suggested":
            note += " -- windows disagree or the phrase is ambiguous; needs a listen"

        for stem in cluster.get("stems", []):
            sidecar = spool / f"{stem}.json"
            if not sidecar.is_file():
                missing += 1
                continue
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            if meta.get("label"):
                preserved += 1
                continue
            if transcripts:
                meta["transcript"] = transcripts
            meta["notes"] = note
            if decidable:
                meta["label"] = cluster.get("label")
                labelled += 1
            else:
                suggested_only += 1
            sidecar.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(f"labelled        {labelled} windows")
    print(f"suggested only  {suggested_only} windows (label left null; needs a listen)")
    print(f"left alone      {preserved} windows already labelled by review")
    if missing:
        print(f"not in spool    {missing} ledger windows (spool has moved on)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spool", type=Path)
    ap.add_argument("--unreviewed", action="store_true", help="List only clips with no label")
    ap.add_argument("--play", action="store_true", help="Print a play command per row")
    ap.add_argument("--label", nargs=2, metavar=("STEM", "LABEL"), help=f"Set label; one of {LABELS}")
    ap.add_argument(
        "--label-cluster",
        nargs=2,
        metavar=("STEM", "LABEL"),
        help="Set one label on every window of that stem's cluster (needs --ledger)",
    )
    ap.add_argument("--ledger", type=Path, help="Ledger from scripts/wake_label_spool.py")
    ap.add_argument(
        "--review-only",
        action="store_true",
        help="List only clips whose cluster the rule flagged for a listen",
    )
    ap.add_argument(
        "--apply-suggestions",
        action="store_true",
        help="Write the ledger's verdict onto the sidecars (needs --ledger)",
    )
    ap.add_argument(
        "--include-review",
        action="store_true",
        help="With --apply-suggestions: label the flagged clusters too, recording the doubt in notes",
    )
    ap.add_argument("--limit", type=int, default=25)
    args = ap.parse_args(argv)

    if not args.spool.is_dir():
        print(f"No such spool directory: {args.spool}", file=sys.stderr)
        return 1

    ledger = load_ledger(args.ledger) if args.ledger else {}
    if args.review_only and not ledger:
        print("--review-only needs --ledger", file=sys.stderr)
        return 1
    if args.apply_suggestions:
        if not ledger:
            print("--apply-suggestions needs --ledger", file=sys.stderr)
            return 1
        return apply_suggestions(args.spool, ledger, include_review=args.include_review)

    if args.label:
        stem, label = args.label
        if label not in LABELS:
            print(f"Label must be one of {LABELS}", file=sys.stderr)
            return 1
        return write_label(args.spool, [stem], label)

    if args.label_cluster:
        stem, label = args.label_cluster
        if label not in LABELS:
            print(f"Label must be one of {LABELS}", file=sys.stderr)
            return 1
        cluster = ledger.get(stem)
        if cluster is None:
            print(f"No ledger cluster for stem {stem}; pass --ledger", file=sys.stderr)
            return 1
        return write_label(args.spool, list(cluster.get("stems", [])), label)

    rows = load(args.spool)
    summarise(rows)
    if ledger:
        summarise_ledger(ledger)

    listing = [r for r in rows if not r.get("label")] if args.unreviewed else rows
    if args.review_only:
        listing = [r for r in listing if ledger.get(r["_wav"].stem, {}).get("review") == "listen"]
    if not listing:
        return 0
    listing.sort(key=lambda r: float(r["score"]), reverse=True)
    header = f"{'score':>7}  {'fired':<5}  {'label':<9}  "
    if ledger:
        header += f"{'suggested':<18}  "
    print(f"\n{header}clip")
    for row in listing[: args.limit]:
        line = (
            f"{float(row['score']):>7.4f}  {str(bool(row.get('fired'))):<5}  "
            f"{str(row.get('label') or '-'):<9}  "
        )
        if ledger:
            cluster = ledger.get(row["_wav"].stem)
            if cluster:
                flag = "?" if cluster.get("review") == "listen" else ""
                suggestion = f"{cluster.get('label', '?')}{flag} ({cluster.get('detail', '?')})"
            else:
                suggestion = "-"
            line += f"{suggestion:<18}  "
        print(f"{line}{row['_wav'].name}")
        if args.play:
            print(f"         aplay {row['_wav']}")
    if len(listing) > args.limit:
        print(f"... {len(listing) - args.limit} more (--limit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
