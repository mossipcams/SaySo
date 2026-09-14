#!/usr/bin/env python3
"""Apply the isolation wake rule to a mining spool and emit a review ledger.

The rule (``docs/WAKE_WORD_DATA.md``): the phrase wakes only when nothing is
spoken in front of it. A word *after* the phrase is fine -- "SaySo turn on the
TV" is the continuous-command shape the product depends on -- but a word before
it makes the whole event a negative, however high the classifier scored it.

Mining writes windows, not labels, and the first real drain was mostly genuine
wakes (44 of 66 clusters). So this script answers "what does the rule make of
each cluster", not "how close to threshold was it". It emits **suggestions**:
sidecar ``label`` is never written, because labelling is a listening decision
and that field belongs to the human review.

    # verdicts only, no side effects
    python scripts/wake_label_spool.py /var/lib/sayso-satellite/wake-mining \
        --transcripts transcripts.json --ledger satellite/models/wake-spool-labels.json

    # only the clusters whose windows disagree, i.e. the ones to listen to first
    python scripts/wake_label_spool.py SPOOL --transcripts T.json --review-only

Transcript shape: ``{stem: {"text": ...}}`` (what the drain tooling writes) or
``{stem: "text"}``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Whisper spellings that count as realizations of /seɪ soʊ/. "stay soon" and
# "see you soon" are deliberately NOT here: they are near-misses the current
# model fires on, and whether they are mis-transcribed wakes or real false
# activations is a listening question, not a transcript question.
REALIZATIONS = (r"\bsay\s*so\b", r"\bsayso\b", r"\bseizo\b", r"\bzzo\b")

# Same reasoning, for windows that contain no realization: flag them for a
# listen rather than silently filing them as unrelated negatives.
NEAR_MISSES = (r"\bstay\s+soon\b", r"\bsee\s+you\s+soon\b", r"\bstay\s+so\b", r"\bso\s+so\b")

# Leading punctuation that a transcript may prepend without speaking. Punctuation
# *inside* the prefix is evidence and must survive.
_LEADING_NOISE = " \t\"'`([{-–—:;,"

POSITIVE, NEGATIVE, UNSURE = "positive", "negative", "unsure"

ISOLATED = "positive_isolated"
PRECEDED = "negative_preceded"
NO_PHRASE = "negative_no_phrase"
NEAR_MISS = "negative_no_phrase_near_miss"
BOUNDARY = "unsure_boundary"

_LABEL_FOR_DETAIL = {
    ISOLATED: POSITIVE,
    PRECEDED: NEGATIVE,
    NO_PHRASE: NEGATIVE,
    NEAR_MISS: NEGATIVE,
    BOUNDARY: UNSURE,
}


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def _earliest_realization(text: str) -> re.Match[str] | None:
    """The leftmost realization match, so its span is the one that decides."""
    matches = [m for pattern in REALIZATIONS if (m := re.search(pattern, text))]
    if not matches:
        return None
    return min(matches, key=lambda m: m.start())


def _find_realization(text: str) -> int | None:
    """Index of the earliest realization of the phrase, or None."""
    match = _earliest_realization(text)
    return match.start() if match else None


def phrase_trailing_words(text: str) -> str:
    """Words spoken after the phrase, ``""`` when the phrase ends the utterance.

    The isolation rule only looks backwards, so this is what makes the
    continuous-command shape ("SaySo turn on the TV") distinguishable from a bare
    wake in a transcript.
    """
    normalised = _normalise(text).lstrip(_LEADING_NOISE)
    match = _earliest_realization(normalised)
    if match is None:
        return ""
    return normalised[match.end() :].strip(" .,!?;:\"'")


def classify_transcript(text: str) -> str:
    """What the isolation rule makes of one window's transcript.

    ``unsure_boundary`` is the honest answer when the only thing in front of the
    phrase is a finished sentence ("and tools. Say so."): that is either a new
    utterance (a wake) or the same breath (not one), and no transcript can say
    which. Only ``positive_isolated`` and ``negative_preceded`` are decisions.
    """
    normalised = _normalise(text).lstrip(_LEADING_NOISE)
    if not normalised:
        return NO_PHRASE

    index = _find_realization(normalised)
    if index is None:
        if any(re.search(pattern, normalised) for pattern in NEAR_MISSES):
            return NEAR_MISS
        return NO_PHRASE

    prefix = normalised[:index].strip().strip("\"'`([{-–—:;,")
    if not prefix:
        return ISOLATED
    if re.search(r"[.!?…]", prefix):
        return BOUNDARY
    return PRECEDED


@dataclass
class Window:
    stem: str
    text: str
    score: float
    fired: bool
    time: float


@dataclass
class Cluster:
    index: int
    windows: list[Window]

    @property
    def best(self) -> Window:
        return max(self.windows, key=lambda w: w.score)

    @property
    def details(self) -> list[str]:
        return [classify_transcript(w.text) for w in self.windows]

    @property
    def detail(self) -> str:
        """Worst case wins.

        A window that starts after the preceding word legitimately looks
        isolated, so one ``negative_preceded`` window is decisive for the
        cluster while the reverse is not. Same for a sentence boundary, which is
        never overwritten by a neighbouring isolated reading.
        """
        details = self.details
        for candidate in (PRECEDED, BOUNDARY, ISOLATED, NEAR_MISS):
            if candidate in details:
                return candidate
        return NO_PHRASE

    @property
    def label(self) -> str:
        return _LABEL_FOR_DETAIL[self.detail]

    @property
    def disagreement(self) -> bool:
        """Windows of one utterance that the transcripts do not call the same thing.

        This includes the case a naive agreement check would miss: one window
        transcribes "Say so." and its neighbour transcribes "Okay, so turn on the
        web". The phrase was heard once and not the other time, which is also how
        a dropped carrier word looks.
        """
        return len(set(self.details)) > 1

    @property
    def review(self) -> str:
        """``listen`` when no transcript can settle it, ``suggested`` otherwise."""
        if self.disagreement or self.detail in (PRECEDED, BOUNDARY, NEAR_MISS):
            return "listen"
        return "suggested"

    @property
    def duration_s(self) -> float:
        return self.windows[-1].time - self.windows[0].time

    @property
    def transcripts(self) -> list[str]:
        return sorted({w.text for w in self.windows if w.text})

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start_utc": datetime.fromtimestamp(self.windows[0].time, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "windows": len(self.windows),
            "duration_s": round(self.duration_s, 3),
            "best_score": round(self.best.score, 6),
            "fired": any(w.fired for w in self.windows),
            "label": self.label,
            "detail": self.detail,
            "review": self.review,
            "disagreement": self.disagreement,
            "transcripts": self.transcripts,
            "best_stem": self.best.stem,
            "stems": [w.stem for w in self.windows],
        }


def _text_for(stem: str, transcripts: dict) -> str:
    entry = transcripts.get(stem)
    if entry is None:
        return ""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return str(entry.get("text", ""))
    return ""


def _window_time(stem: str, captured_utc: str) -> float:
    """Epoch seconds from the sidecar stamp plus the stem's millisecond field.

    Filenames are ``<utc stamp>_<ms within second>_s<score>``; the millisecond
    field is what orders windows inside one second, which a 160 ms hop needs.
    """
    stamp = stem.split("_")[0]
    milliseconds = 0
    parts = stem.split("_")
    if len(parts) > 1 and parts[1].isdigit():
        milliseconds = int(parts[1])
    try:
        base = datetime.strptime(stamp or captured_utc, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return 0.0
    return base.timestamp() + milliseconds / 1000.0


def load_windows(spool: Path, transcripts: dict) -> list[Window]:
    windows: list[Window] = []
    for sidecar_path in sorted(spool.glob("*.json")):
        try:
            meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skipping malformed sidecar {sidecar_path}: {exc}", file=sys.stderr)
            continue
        stem = sidecar_path.stem
        windows.append(
            Window(
                stem=stem,
                text=_text_for(stem, transcripts),
                score=float(meta.get("score", 0.0)),
                fired=bool(meta.get("fired")),
                time=_window_time(stem, str(meta.get("captured_utc", ""))),
            )
        )
    windows.sort(key=lambda w: w.time)
    return windows


def cluster_windows(windows: list[Window], gap_seconds: float) -> list[Cluster]:
    """Group windows into utterances; a gap longer than ``gap_seconds`` starts one."""
    clusters: list[list[Window]] = []
    for window in windows:
        if clusters and window.time - clusters[-1][-1].time <= gap_seconds:
            clusters[-1].append(window)
        else:
            clusters.append([window])
    return [Cluster(index=i, windows=group) for i, group in enumerate(clusters)]


def build_ledger(spool: Path, clusters: list[Cluster], transcripts_path: Path | None) -> dict:
    labels = {POSITIVE: 0, NEGATIVE: 0, UNSURE: 0}
    details: dict[str, int] = {}
    for cluster in clusters:
        labels[cluster.label] += 1
        details[cluster.detail] = details.get(cluster.detail, 0) + 1
    return {
        "version": 1,
        "rule": "isolation: the phrase wakes only when nothing is spoken in front of it",
        "spool": str(spool),
        "transcripts": str(transcripts_path) if transcripts_path else None,
        "window_count": sum(len(c.windows) for c in clusters),
        "cluster_count": len(clusters),
        "summary": {
            "labels": labels,
            "details": dict(sorted(details.items())),
            "disagreements": sum(1 for c in clusters if c.disagreement),
            "needs_listening": sum(1 for c in clusters if c.review == "listen"),
        },
        "clusters": [cluster.to_dict() for cluster in clusters],
    }


def print_report(ledger: dict, clusters: list[Cluster], review_only: bool) -> None:
    summary = ledger["summary"]
    print(f"spool    {ledger['spool']}")
    print(f"rule     {ledger['rule']}")
    print(f"windows  {ledger['window_count']}   clusters {ledger['cluster_count']}")
    print()
    print("label      clusters")
    for label in (POSITIVE, NEGATIVE, UNSURE):
        print(f"  {label:<8} {summary['labels'][label]:>4}")
    print()
    print("detail                              clusters")
    for detail, count in summary["details"].items():
        print(f"  {detail:<32} {count:>4}")
    print()
    print(
        f"windows of the same utterance disagree in {summary['disagreements']} clusters; "
        f"{summary['needs_listening']} need a listen before their label means anything."
    )

    if not review_only:
        return
    print()
    print("needs listening (best window first):")
    for cluster in sorted(clusters, key=lambda c: -c.best.score):
        if cluster.review != "listen":
            continue
        heard = " | ".join(cluster.transcripts) or "<no transcript>"
        print(
            f"  {cluster.best.score:.4f}  {cluster.detail:<20} "
            f"{cluster.windows[0].stem}  {heard[:80]}"
        )
        print(f"          aplay {cluster.best.stem}.wav")


def load_transcripts(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read transcripts {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"transcripts must be a JSON object: {path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("spool", type=Path, help="Mining spool directory (sidecar JSON per clip)")
    parser.add_argument(
        "--transcripts",
        type=Path,
        default=None,
        help="Whisper dump: {stem: {'text': ...}} or {stem: 'text'}",
    )
    parser.add_argument("--ledger", type=Path, default=None, help="Write the ledger JSON here")
    parser.add_argument(
        "--cluster-gap",
        type=float,
        default=3.0,
        help="Seconds of silence that start a new utterance (default: 3.0)",
    )
    parser.add_argument(
        "--review-only",
        action="store_true",
        help="List only the clusters that need a listen",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.spool.is_dir():
        print(f"No such spool directory: {args.spool}", file=sys.stderr)
        return 1

    try:
        transcripts = load_transcripts(args.transcripts)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    windows = load_windows(args.spool, transcripts)
    if not windows:
        print(f"Spool has no sidecars: {args.spool}", file=sys.stderr)
        return 1

    clusters = cluster_windows(windows, args.cluster_gap)
    if not transcripts:
        print(
            "No --transcripts given: every cluster will be negative_no_phrase. "
            "The rule is only decidable with transcripts.",
            file=sys.stderr,
        )

    ledger = build_ledger(args.spool, clusters, args.transcripts)
    print_report(ledger, clusters, args.review_only)

    if args.ledger is not None:
        args.ledger.parent.mkdir(parents=True, exist_ok=True)
        args.ledger.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nledger written: {args.ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
