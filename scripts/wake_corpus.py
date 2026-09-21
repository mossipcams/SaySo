#!/usr/bin/env python3
"""Wake corpus pipeline: session ingest, replay mining, splits, and snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SATELLITE_ROOT = _REPO_ROOT / "satellite"
if str(_SATELLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SATELLITE_ROOT))

from sayso.wake.corpus import labeled_positive_events, load_events, set_event_label  # noqa: E402
from sayso.wake.replay import production_replay_constants, replay_and_import_session  # noqa: E402
from sayso.wake.sessions import (  # noqa: E402
    DEFAULT_SHIP_REMOTE,
    DEFAULT_SHIP_REMOTE_CORPUS,
    ShipSessionError,
    ingest_session,
    list_sessions,
    load_session,
    ship_session,
)
from sayso.wake.snapshot import (  # noqa: E402
    corpus_snapshot_id,
    derive_snapshot_examples,
    ensure_session_splits,
    holdout_sessions,
    write_snapshot_manifest,
)


def cmd_ingest(args: argparse.Namespace) -> int:
    session = ingest_session(
        args.wav,
        args.corpus,
        session_id=args.session_id,
        notes=args.notes,
        copy_audio=not args.link,
    )
    print(json.dumps(session.to_dict(), indent=2, sort_keys=True))
    return 0


def cmd_list_sessions(args: argparse.Namespace) -> int:
    sessions = list_sessions(args.corpus)
    for session in sessions:
        print(f"{session.session_id}\t{session.duration_seconds:.1f}s\t{session.source_path}")
    print(f"total {len(sessions)}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from sayso.wake.livekit import LiveKitWakeWordProvider

    session = load_session(args.corpus, args.session_id)
    provider = LiveKitWakeWordProvider(
        model_path=args.model,
        phrase=args.phrase,
        threshold=args.detect_threshold,
        refractory_seconds=args.refractory,
        verifier_path=args.verifier,
    )
    if not provider.available:
        print(f"wake model unavailable: {args.model}", file=sys.stderr)
        return 1
    stats, imported = replay_and_import_session(
        args.corpus,
        session,
        provider,
        mine_threshold=args.mine_threshold,
        detect_threshold=args.detect_threshold,
        model_path=args.model,
        below_sample_rate=args.below_rate,
    )
    print(
        f"session={session.session_id} windows={stats.windows_scored} "
        f"detections={stats.detections} events={len(imported)}"
    )
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    assignments = ensure_session_splits(
        args.corpus,
        seed=args.seed,
        holdout_session_ids=args.holdout or None,
        eval_fraction=args.eval_fraction,
    )
    for session_id, split in sorted(assignments.items()):
        print(f"{session_id}\t{split}")
    print(f"holdouts {len(holdout_sessions(assignments))}")
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    assignments = ensure_session_splits(
        args.corpus,
        seed=args.seed,
        holdout_session_ids=args.holdout or None,
    )
    examples = derive_snapshot_examples(load_events(args.corpus), assignments)
    snapshot_id = corpus_snapshot_id(examples, seed=args.seed)
    path = write_snapshot_manifest(
        args.corpus,
        snapshot_id,
        examples=examples,
        session_splits=assignments,
        seed=args.seed,
    )
    train_count = sum(1 for example in examples if example.split == "train")
    eval_count = sum(1 for example in examples if example.split == "eval")
    print(f"snapshot_id={snapshot_id}")
    print(f"manifest={path}")
    print(f"examples={len(examples)} train={train_count} eval={eval_count}")
    return 0


def cmd_holdout_eval(args: argparse.Namespace) -> int:
    from sayso.wake.eval import evaluate_holdout_sessions
    from sayso.wake.livekit import LiveKitWakeWordProvider

    assignments = ensure_session_splits(
        args.corpus,
        seed=args.seed,
        holdout_session_ids=args.holdout or None,
    )
    holdout_ids = holdout_sessions(assignments)
    if args.session_id:
        if args.session_id not in holdout_ids:
            print(f"session is not holdout: {args.session_id}", file=sys.stderr)
            return 1
        holdout_ids = [args.session_id]

    provider = LiveKitWakeWordProvider(
        model_path=args.model,
        phrase=args.phrase,
        threshold=args.detect_threshold,
        refractory_seconds=args.refractory,
        verifier_path=args.verifier,
    )
    if not provider.available:
        print(f"wake model unavailable: {args.model}", file=sys.stderr)
        return 1

    events = load_events(args.corpus)
    labeled_by_session = {
        session_id: labeled_positive_events(events, session_id) for session_id in holdout_ids
    }
    sessions = [load_session(args.corpus, session_id) for session_id in holdout_ids]
    report = evaluate_holdout_sessions(
        sessions,
        provider,
        labeled_positives_by_session=labeled_by_session,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_label(args: argparse.Namespace) -> int:
    event = set_event_label(args.corpus, args.event_id, args.label)
    print(f"{event.event_id} -> {event.label}")
    return 0


def cmd_constants(_args: argparse.Namespace) -> int:
    print(json.dumps(production_replay_constants(), indent=2, sort_keys=True))
    return 0


def cmd_ship(args: argparse.Namespace) -> int:
    try:
        result = ship_session(
            args.corpus,
            args.session_id,
            remote=args.remote,
            remote_corpus=args.remote_corpus,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, ValueError, ShipSessionError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if result.dry_run:
        print(
            f"dry-run ship session={result.session_id} "
            f"remote={result.remote} corpus={result.remote_corpus}"
        )
        return 0
    print(
        f"shipped session={result.session_id} "
        f"remote={result.remote} corpus={result.remote_corpus}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus root directory")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest a long-form WAV as a named session")
    ingest.add_argument("wav", type=Path)
    ingest.add_argument("--session-id", default=None)
    ingest.add_argument("--notes", default=None)
    ingest.add_argument("--link", action="store_true", help="Symlink source audio instead of copying")
    ingest.set_defaults(func=cmd_ingest)

    listing = sub.add_parser("list-sessions", help="List ingested sessions")
    listing.set_defaults(func=cmd_list_sessions)

    replay = sub.add_parser("replay", help="Replay one session and import candidate events")
    replay.add_argument("session_id")
    replay.add_argument("--model", type=Path, default=_REPO_ROOT / "satellite" / "models" / "output-living2" / "sayso" / "sayso.onnx")
    replay.add_argument("--verifier", type=Path, default=_REPO_ROOT / "satellite" / "models" / "output-living2" / "sayso" / "verifier.npz")
    replay.add_argument("--phrase", default="SaySo")
    replay.add_argument("--detect-threshold", type=float, default=0.28)
    replay.add_argument("--mine-threshold", type=float, default=0.1)
    replay.add_argument("--refractory", type=float, default=2.0)
    replay.add_argument("--below-rate", type=float, default=0.002)
    replay.set_defaults(func=cmd_replay)

    split = sub.add_parser("split", help="Write deterministic session-level splits")
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--eval-fraction", type=float, default=0.15)
    split.add_argument("--holdout", action="append", default=[])
    split.set_defaults(func=cmd_split)

    snapshot = sub.add_parser("snapshot", help="Write a deterministic training snapshot manifest")
    snapshot.add_argument("--seed", type=int, default=42)
    snapshot.add_argument("--holdout", action="append", default=[])
    snapshot.set_defaults(func=cmd_snapshot)

    holdout_eval = sub.add_parser(
        "holdout-eval",
        help="Continuous replay of holdout sessions with FA/hour (labeled positives excluded)",
    )
    holdout_eval.add_argument("--seed", type=int, default=42)
    holdout_eval.add_argument("--holdout", action="append", default=[])
    holdout_eval.add_argument("--session-id", default=None)
    holdout_eval.add_argument("--model", type=Path, default=_REPO_ROOT / "satellite" / "models" / "output-living2" / "sayso" / "sayso.onnx")
    holdout_eval.add_argument("--verifier", type=Path, default=_REPO_ROOT / "satellite" / "models" / "output-living2" / "sayso" / "verifier.npz")
    holdout_eval.add_argument("--phrase", default="SaySo")
    holdout_eval.add_argument("--detect-threshold", type=float, default=0.28)
    holdout_eval.add_argument("--refractory", type=float, default=2.0)
    holdout_eval.set_defaults(func=cmd_holdout_eval)

    label = sub.add_parser("label", help="Set a human label on one corpus event")
    label.add_argument("event_id")
    label.add_argument("label", choices=("positive", "negative", "unsure"))
    label.set_defaults(func=cmd_label)

    constants = sub.add_parser("constants", help="Print production replay window/hop constants")
    constants.set_defaults(func=cmd_constants)

    ship = sub.add_parser(
        "ship",
        help="Copy one session to the train VM via rsync-over-SSH, verify, then delete locally",
    )
    ship.add_argument("session_id")
    ship.add_argument("--remote", default=DEFAULT_SHIP_REMOTE, help="SSH destination user@host")
    ship.add_argument(
        "--remote-corpus",
        type=Path,
        default=Path(DEFAULT_SHIP_REMOTE_CORPUS),
        help="Remote corpus root directory",
    )
    ship.add_argument("--dry-run", action="store_true", help="Print the ship target without copying")
    ship.set_defaults(func=cmd_ship)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.corpus.exists() and args.command not in {"ingest", "constants"}:
        print(f"corpus root missing: {args.corpus}", file=sys.stderr)
        return 1
    args.corpus.mkdir(parents=True, exist_ok=True)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
