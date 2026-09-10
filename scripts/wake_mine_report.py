#!/usr/bin/env python3
"""Summarise and label the wake hard-negative spool.

Clips land unlabelled: the satellite knows a window scored high, not whether
anyone actually said "SaySo". Labelling is a listening job, and this script is
the thin wrapper around it.

    # what's in the spool
    python scripts/wake_mine_report.py /var/lib/sayso-satellite/wake-mining

    # highest-scoring unreviewed clips first, with a play command per row
    python scripts/wake_mine_report.py SPOOL --unreviewed --play

    # record a verdict
    python scripts/wake_mine_report.py SPOOL --label 20260910T001432_412_s0.3120 negative

Deliberately not doing clustering or dedup here. Both need the score
distribution of an actual corpus to calibrate against, and inventing a
similarity threshold before any clips exist is how you end up with a pipeline
tuned to nothing. Revisit once the spool has a few hundred labelled clips.
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spool", type=Path)
    ap.add_argument("--unreviewed", action="store_true", help="List only clips with no label")
    ap.add_argument("--play", action="store_true", help="Print a play command per row")
    ap.add_argument("--label", nargs=2, metavar=("STEM", "LABEL"), help=f"Set label; one of {LABELS}")
    ap.add_argument("--limit", type=int, default=25)
    args = ap.parse_args()

    if not args.spool.is_dir():
        print(f"No such spool directory: {args.spool}", file=sys.stderr)
        return 1

    if args.label:
        stem, label = args.label
        if label not in LABELS:
            print(f"Label must be one of {LABELS}", file=sys.stderr)
            return 1
        sidecar = args.spool / f"{stem}.json"
        if not sidecar.is_file():
            print(f"No sidecar for stem {stem}", file=sys.stderr)
            return 1
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        meta["label"] = label
        sidecar.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        print(f"{stem} -> {label}")
        return 0

    rows = load(args.spool)
    summarise(rows)

    listing = [r for r in rows if not r.get("label")] if args.unreviewed else rows
    if not listing:
        return 0
    listing.sort(key=lambda r: float(r["score"]), reverse=True)
    print(f"\n{'score':>7}  {'fired':<5}  {'label':<9}  clip")
    for row in listing[: args.limit]:
        print(
            f"{float(row['score']):>7.4f}  {str(bool(row.get('fired'))):<5}  "
            f"{str(row.get('label') or '-'):<9}  {row['_wav'].name}"
        )
        if args.play:
            print(f"         aplay {row['_wav']}")
    if len(listing) > args.limit:
        print(f"... {len(listing) - args.limit} more (--limit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
