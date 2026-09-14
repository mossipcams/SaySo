#!/usr/bin/env python3
"""Turn a labelled mining spool into the two corpora the wake model needs.

This is the step between ``docs/PLAN_WAKE_WORD_REAL_DATA.md`` item 3 (label the
spool) and a retrain, and it exists because the upstream pipeline ignores real
audio unless it is placed exactly right. Three constraints, all read off the
installed ``livekit/wakeword``:

1. **Feature extraction reads only ``clip_\\d{6}_r\\d+.wav``**
   (``data/features.py::extract_features_from_directory``). A real clip dropped
   in as ``my_recording.wav`` is never trained on, silently.
2. **Split directories live under ``output/<model_name>/``**
   (``WakeWordConfig.model_output_dir``), not under ``data_dir``. ``data_dir``
   holds setup artifacts only.
3. **``run_augment`` deletes every ``clip_\\d{6}_r\\d+\\.wav`` before it
   starts** ("clean up old augmented files"). The first seeding attempt named
   real windows ``clip_9XXXXX_r0.wav`` to dodge round-0 alignment; augment's own
   cleanup erased all 8220 of them and the model trained on zero real audio.
   Real windows are therefore seeded as plain originals, ``clip_9XXXXX.wav``:
   the cleanup ignores them, ``generate`` counts them as already-finished work,
   and both augmentation rounds apply to them like any TTS clip. For a
   full-length 2 s clip, round-0 alignment drops up to 200 ms of *lead-in* and
   leaves the phrase intact, so nothing is lost by taking that route.

Two subcommands:

    # a corpus the eval harness can read, kept outside the repo (public repo,
    # real household speech in it -- see satellite/eval/README.md)
    python scripts/wake_prepare_real_data.py eval-corpus \\
        --spool /var/lib/sayso-satellite/wake-mining \\
        --ledger satellite/models/wake-spool-labels-20260914.json \\
        --eval-root ~/sayso-eval-corpus \\
        --held-out satellite/models/wake-held-out-20260914.json

    # seed output/sayso/{positive,negative}_train for the retrain
    python scripts/wake_prepare_real_data.py training-data \\
        --spool /var/lib/sayso-satellite/wake-mining \\
        --ledger satellite/models/wake-spool-labels-20260914.json \\
        --model-dir output/sayso \\
        --held-out satellite/models/wake-held-out-20260914.json

Nothing is guessed from the score: which cluster belongs in which class comes
from the ledger, which applies the isolation rule.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import wake_label_spool as labeler  # noqa: E402

#: case id -> (ledger detail class, extra requirement). Order matters: each entry
#: takes the best window not already claimed, and falls back to another window of
#: the same utterance once the class runs out -- the spool has one carrier
#: cluster, and both halves of the isolation rule need a fixture. Ids must exist
#: in satellite/eval/cases.json, which is checked at run time.
EVAL_CASE_PLAN: tuple[tuple[str, str, str], ...] = (
    ("pos_say_so_two_word", labeler.ISOLATED, "any"),
    ("pos_continuous_kitchen", labeler.ISOLATED, "trailing_words"),
    ("neg_say_so_carrier_if_you", labeler.PRECEDED, "any"),
    ("neg_say_so_carrier_leading_word", labeler.PRECEDED, "any"),
    ("neg_tv_conversation", labeler.NO_PHRASE, "any"),
    ("neg_distance_noise", labeler.NO_PHRASE, "any"),
)

#: Cases whose audio cannot come from this spool, with the reason.
EVAL_CASE_GAPS = {
    "pos_sayso_near": "no fused 'Sayso' realization is attested in 152 windows",
    "pos_sayso_distance": "no reviewed capture at distance exists yet",
}

# Real clips are named like generate's own output so the pipeline reads them, but
# numbered far above the clip_000000..n_samples range.
DEFAULT_INDEX_BASE = 900000


def load_ledger(path: Path) -> list[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read ledger {path}: {exc}") from exc
    clusters = payload.get("clusters")
    if not isinstance(clusters, list) or not clusters:
        raise ValueError(f"ledger has no clusters: {path}")
    return clusters


def load_held_out(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return set(payload.get("stems", []))


def _ranked(clusters: list[dict], detail: str) -> list[dict]:
    return sorted((c for c in clusters if c["detail"] == detail), key=lambda c: -float(c["best_score"]))


def _next_unused_stem(cluster: dict, used: set[str]) -> str | None:
    """The cluster's best window, or its next window, or None if all are claimed."""
    best = cluster.get("best_stem")
    if best and best not in used and best in cluster["stems"]:
        return best
    return next((stem for stem in cluster["stems"] if stem not in used), None)


def _has_trailing_words(cluster: dict) -> bool:
    return any(labeler.phrase_trailing_words(text) for text in cluster.get("transcripts", []))


def select_eval_stems(clusters: list[dict]) -> tuple[dict[str, tuple[dict, str]], list[tuple[str, str]]]:
    """One (cluster, stem) per case, plus the cases nothing could fill.

    Two passes. The first only takes a cluster no other case has used, so the
    corpus gets distinct utterances wherever the spool can provide them. The
    second relaxes that to another window of an already-used cluster, which is
    what lets both carrier cases exist when the spool has one carrier cluster.
    """
    used_stems: set[str] = set()
    used_clusters: set[int] = set()
    chosen: dict[str, tuple[dict, str]] = {}
    pending = list(EVAL_CASE_PLAN)

    for allow_reuse in (False, True):
        still_pending: list[tuple[str, str, str]] = []
        for case_id, detail, requirement in pending:
            picked: tuple[dict, str] | None = None
            for cluster in _ranked(clusters, detail):
                if requirement == "trailing_words" and not _has_trailing_words(cluster):
                    continue
                if not allow_reuse and cluster["index"] in used_clusters:
                    continue
                stem = _next_unused_stem(cluster, used_stems)
                if stem is None:
                    continue
                picked = (cluster, stem)
                break
            if picked is None:
                still_pending.append((case_id, detail, requirement))
                continue
            cluster, stem = picked
            used_stems.add(stem)
            used_clusters.add(cluster["index"])
            chosen[case_id] = (cluster, stem)
        pending = still_pending
        if not pending:
            break

    gaps: list[tuple[str, str]] = []
    for case_id, detail, requirement in pending:
        suffix = " with words after the phrase" if requirement == "trailing_words" else ""
        gaps.append((case_id, f"no {detail} cluster{suffix} with an unclaimed window"))
    for case_id, reason in EVAL_CASE_GAPS.items():
        gaps.append((case_id, reason))
    return chosen, gaps


def build_eval_corpus(
    spool: Path, clusters: list[dict], eval_root: Path, held_out_path: Path | None
) -> dict:
    cases_path = eval_root / "cases.json"
    if not cases_path.is_file():
        raise ValueError(f"no cases.json in {eval_root}")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    chosen, gaps = select_eval_stems(clusters)

    case_ids = {case["id"] for case in cases.get("cases", [])}
    for case_id, _, _ in EVAL_CASE_PLAN:
        if case_id not in case_ids:
            print(f"plan case id not in cases.json: {case_id}", file=sys.stderr)
            gaps.append((case_id, "case id not in cases.json"))

    (eval_root / "audio").mkdir(parents=True, exist_ok=True)

    # A missing fixture makes even a populated continuous_command case skip, so
    # the stub is part of populating the corpus. It is text the case already
    # declared, never a claim about what the audio says.
    for case in cases.get("cases", []):
        relative = case.get("transcript_fixture")
        if not relative:
            continue
        stub = eval_root / relative
        stub.parent.mkdir(parents=True, exist_ok=True)
        text = case.get("expected_transcript") or case.get("command_transcript") or ""
        stub.write_text(
            json.dumps({"text": text, "stt_delay_ms": 1200.0, "stub": True}) + "\n",
            encoding="utf-8",
        )
        print(f"stt stub      {relative}  (text={text!r}, a stub -- not real STT output)")

    written: list[dict] = []
    cluster_stems: set[str] = set()
    for case in cases.get("cases", []):
        case_id = case["id"]
        if case_id not in chosen:
            continue
        cluster, stem = chosen[case_id]
        shutil.copyfile(spool / f"{stem}.wav", eval_root / case["audio"])
        cluster_stems.update(cluster["stems"])
        written.append(
            {
                "case_id": case_id,
                "audio": case["audio"],
                "cluster_index": cluster["index"],
                "stem": stem,
                "label": cluster["label"],
                "detail": cluster["detail"],
                "best_score": cluster["best_score"],
                "review": cluster.get("review"),
                "transcripts": cluster.get("transcripts", []),
            }
        )
        print(
            f"populated     {case['audio']:<42} <- {stem} "
            f"({cluster['detail']}, {float(cluster['best_score']):.4f})"
        )

    for case_id, reason in gaps:
        print(f"NOT populated {case_id:<42} {reason}")

    if held_out_path is not None:
        held_out_path.parent.mkdir(parents=True, exist_ok=True)
        held_out_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "reason": (
                        "every cluster the recorded-audio eval corpus draws on, excluded from "
                        "positive_train, negative_train and the *_test validation splits so the "
                        "operating point is never selected on the clips it is judged on"
                    ),
                    "stems": sorted(cluster_stems),
                    "clusters": written,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nheld-out list {held_out_path} ({len(cluster_stems)} windows in {len({e['cluster_index'] for e in written})} clusters)")

    return {"populated": written, "gaps": [{"case_id": c, "reason": r} for c, r in gaps]}


def _clip_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def clear_seeded(model_dir: Path, index_base: int) -> int:
    """Remove previously seeded real clips (index >= index_base) and the manifest.

    Generated TTS clips sit below ``index_base`` and are never touched. Feature
    files are left alone but become stale: re-run augment and train afterwards.
    """
    removed = 0
    for split in ("positive_train", "negative_train"):
        for path in (model_dir / split).glob("clip_*.wav"):
            match = re.match(r"clip_(\d{6})(?:_r\d+)?\.wav$", path.name)
            if match and int(match.group(1)) >= index_base:
                path.unlink()
                removed += 1
    (model_dir / "real_clip_manifest.json").unlink(missing_ok=True)
    if removed:
        print(
            f"replaced       removed {removed} previously seeded clips; "
            "*_features_*.npy are now stale, re-run augment and train"
        )
    return removed


def copy_live_backgrounds(
    spool: Path, clusters: list[dict], out_dir: Path, held_out: set[str], *, dry_run: bool
) -> int:
    """Copy phrase-free real windows so augmentation mixes the live room, not just MUSAN.

    A cluster that merely *contains* a wake phrase (the carrier, or a window the
    transcripts cannot settle) is never a background: mixing it under a positive
    would teach the phrase next to itself. Held-out clusters are excluded too, or
    the eval corpus would leak into training.
    """
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for cluster in clusters:
        if cluster["detail"] not in (labeler.NO_PHRASE, labeler.NEAR_MISS):
            continue
        if held_out and set(cluster["stems"]) & held_out:
            continue
        for stem in cluster["stems"]:
            source = spool / f"{stem}.wav"
            if not source.is_file():
                print(f"missing audio for stem {stem}", file=sys.stderr)
                continue
            if not dry_run:
                shutil.copyfile(source, out_dir / f"bg_{stem}.wav")
            copied += 1
    return copied


def seed_split(
    stems: list[str],
    spool: Path,
    out_dir: Path,
    *,
    repeat: int,
    index: int,
    dry_run: bool,
) -> tuple[int, int]:
    """Write each stem ``repeat`` times as ``clip_<index>.wav``. Returns (files, next index)."""
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for stem in stems:
        source = spool / f"{stem}.wav"
        if not source.is_file():
            print(f"missing audio for stem {stem}", file=sys.stderr)
            continue
        for _ in range(repeat):
            if not dry_run:
                # Plain original naming: augment's startup cleanup targets only
                # clip_dddddd_rN.wav, so originals survive it.
                shutil.copyfile(source, out_dir / f"clip_{index:06d}.wav")
            index += 1
            written += 1
    return written, index


def build_training_data(
    spool: Path,
    clusters: list[dict],
    model_dir: Path,
    held_out_path: Path | None,
    *,
    repeat: int,
    negative_repeat: int,
    index_base: int,
    dry_run: bool,
    all_windows: bool = False,
    live_backgrounds: Path | None = None,
) -> dict:
    held_out = load_held_out(held_out_path)
    positives: list[tuple[str, dict]] = []
    negatives: list[tuple[str, dict]] = []
    excluded = 0

    for cluster in clusters:
        if held_out and set(cluster["stems"]) & held_out:
            excluded += 1
            continue
        if all_windows:
            # Every window the live pipeline scored, at the alignment it scored
            # them at -- not one canonical window per utterance. The windows
            # overlap, but each gets its own augmentation pass, so the variation
            # is real even though the underlying utterances are few.
            stems = list(cluster["stems"])
        else:
            best = _next_unused_stem(cluster, set())
            stems = [best] if best else []
        target = positives if cluster["detail"] == labeler.ISOLATED else negatives
        target.extend((stem, cluster) for stem in stems)

    if not positives:
        print("no positive clusters to seed: is the ledger labelled?", file=sys.stderr)
        raise SystemExit(1)

    durations = [_clip_duration(spool / f"{stem}.wav") for stem, _ in positives[:5] if (spool / f"{stem}.wav").is_file()]

    index = index_base
    pos_files, index = seed_split(
        [stem for stem, _ in positives],
        spool,
        model_dir / "positive_train",
        repeat=repeat,
        index=index,
        dry_run=dry_run,
    )
    neg_files, index = seed_split(
        [stem for stem, _ in negatives],
        spool,
        model_dir / "negative_train",
        repeat=negative_repeat,
        index=index,
        dry_run=dry_run,
    )

    print(f"positive_train {len(positives)} real windows -> {pos_files} clips (repeat {repeat})")
    print(f"negative_train {len(negatives)} real windows -> {neg_files} clips (repeat {negative_repeat})")
    print(f"held out       {excluded} clusters claimed by the eval corpus, seeded nowhere")
    if live_backgrounds is not None:
        copied = copy_live_backgrounds(
            spool, clusters, live_backgrounds, held_out, dry_run=dry_run
        )
        print(f"live bg        {copied} phrase-free real windows -> {live_backgrounds}")
    print(f"names          clip_{index_base:06d}.wav .. clip_{index - 1:06d}.wav")
    if durations:
        print(f"durations      first {len(durations)} positive clips: {' '.join(f'{d:.2f}s' for d in durations)}")

    manifest = {
        "version": 1,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "spool": str(spool),
        "model_dir": str(model_dir),
        "index_base": index_base,
        "repeat": {"positive": repeat, "negative": negative_repeat},
        "all_windows": all_windows,
        "live_backgrounds": str(live_backgrounds) if live_backgrounds else None,
        "held_out_clusters": excluded,
        "positive_utterances": [
            {"stem": stem, "cluster_index": cluster["index"], "best_score": cluster["best_score"]}
            for stem, cluster in positives
        ],
        "negative_utterances": [
            {
                "stem": stem,
                "cluster_index": cluster["index"],
                "detail": cluster["detail"],
                "best_score": cluster["best_score"],
            }
            for stem, cluster in negatives
        ],
        "why_index_base": (
            "seeded as plain clip_dddddd originals at index 900000+: augment's startup "
            "cleanup deletes every clip_dddddd_rN.wav, so _r0-named seeds are erased "
            "before they are ever trained on; the high index keeps them clear of the "
            "range generate writes into"
        ),
    }
    if not dry_run:
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "real_clip_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"manifest       {model_dir / 'real_clip_manifest.json'}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(sub_parser: argparse.ArgumentParser) -> None:
        sub_parser.add_argument("--spool", type=Path, required=True, help="Mining spool directory")
        sub_parser.add_argument(
            "--ledger", type=Path, required=True, help="Ledger from wake_label_spool.py"
        )
        sub_parser.add_argument(
            "--held-out",
            type=Path,
            default=None,
            help="Held-out stem list JSON (written by eval-corpus, read by training-data)",
        )

    eval_parser = sub.add_parser("eval-corpus", help="Populate a recorded-audio eval root")
    common(eval_parser)
    eval_parser.add_argument(
        "--eval-root", type=Path, required=True, help="Directory holding cases.json"
    )

    train_parser = sub.add_parser(
        "training-data", help="Seed output/<model>/ positive and negative training splits"
    )
    common(train_parser)
    train_parser.add_argument("--model-dir", type=Path, default=Path("output/sayso"))
    train_parser.add_argument("--repeat", type=int, default=60, help="Copies per positive window")
    train_parser.add_argument("--negative-repeat", type=int, default=60)
    train_parser.add_argument(
        "--all-windows",
        action="store_true",
        help="Seed every scored window of every cluster, not one canonical window per utterance",
    )
    train_parser.add_argument(
        "--live-backgrounds",
        type=Path,
        default=None,
        help="Copy phrase-free real windows here so augmentation mixes the live room",
    )
    train_parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete and re-seed previously seeded real clips (index >= --index-base)",
    )
    train_parser.add_argument(
        "--index-base",
        type=int,
        default=DEFAULT_INDEX_BASE,
        help="Starting clip index; keep far above n_samples so generate cannot collide",
    )
    train_parser.add_argument("--dry-run", action="store_true", help="Report the plan, write nothing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.spool.is_dir():
        print(f"No such spool directory: {args.spool}", file=sys.stderr)
        return 1
    try:
        clusters = load_ledger(args.ledger)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.command == "eval-corpus":
        try:
            result = build_eval_corpus(args.spool, clusters, args.eval_root, args.held_out)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(
            f"\n{len(result['populated'])} cases populated, "
            f"{len(result['gaps'])} still need a capture"
        )
        return 0

    if args.dry_run:
        build_training_data(
            args.spool,
            clusters,
            args.model_dir,
            args.held_out,
            repeat=args.repeat,
            negative_repeat=args.negative_repeat,
            index_base=args.index_base,
            dry_run=True,
            all_windows=args.all_windows,
            live_backgrounds=args.live_backgrounds,
        )
        print("\ndry run: nothing written")
        return 0

    if args.replace:
        clear_seeded(args.model_dir, args.index_base)
    elif (args.model_dir / "positive_train").exists():
        print(
            f"{args.model_dir}/positive_train already exists; pass --replace to reseed the real "
            "clips, or move it aside (generate and augment resume from what is there, so seeding "
            "twice duplicates audio)",
            file=sys.stderr,
        )
        return 1

    build_training_data(
        args.spool,
        clusters,
        args.model_dir,
        args.held_out,
        repeat=args.repeat,
        negative_repeat=args.negative_repeat,
        index_base=args.index_base,
        dry_run=False,
        all_windows=args.all_windows,
        live_backgrounds=args.live_backgrounds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
