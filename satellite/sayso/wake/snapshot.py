"""Deterministic wake corpus snapshots with session-level splits."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .corpus import CandidateEvent, TRAIN_LABELS, load_events
from .sessions import RecordingSession, list_sessions

SPLITS_NAME = "corpus_splits.json"
SNAPSHOT_MANIFEST = "corpus_snapshot.json"
VALID_SPLITS = frozenset({"train", "eval", "holdout"})


def _stable_fraction(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def assign_session_splits(
    session_ids: Sequence[str],
    *,
    seed: int,
    holdout_session_ids: Sequence[str] | None = None,
    eval_fraction: float = 0.15,
) -> dict[str, str]:
    """Deterministically assign whole sessions to train/eval/holdout."""
    holdouts = {sid for sid in (holdout_session_ids or []) if sid}
    assignments: dict[str, str] = {}
    for session_id in sorted(set(session_ids)):
        if session_id in holdouts:
            assignments[session_id] = "holdout"
            continue
        fraction = _stable_fraction(session_id, seed)
        if fraction < eval_fraction:
            assignments[session_id] = "eval"
        else:
            assignments[session_id] = "train"
    return assignments


def write_session_splits(
    corpus_root: Path,
    assignments: Mapping[str, str],
    *,
    seed: int,
    holdout_session_ids: Sequence[str] | None = None,
) -> Path:
    for session_id, split in assignments.items():
        if split not in VALID_SPLITS:
            raise ValueError(f"invalid split {split!r} for session {session_id}")
    path = Path(corpus_root) / SPLITS_NAME
    payload = {
        "version": 1,
        "seed": seed,
        "holdout_session_ids": sorted(set(holdout_session_ids or [])),
        "assignments": dict(sorted(assignments.items())),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_session_splits(corpus_root: Path) -> dict[str, str]:
    path = Path(corpus_root) / SPLITS_NAME
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    assignments = payload.get("assignments") or {}
    return {str(k): str(v) for k, v in assignments.items()}


def _load_splits_payload(corpus_root: Path) -> dict:
    path = Path(corpus_root) / SPLITS_NAME
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _known_session_ids(corpus_root: Path) -> set[str]:
    sessions = list_sessions(corpus_root)
    events = load_events(corpus_root)
    session_ids = {session.session_id for session in sessions}
    session_ids.update(event.session_id for event in events if event.session_id)
    return session_ids


def ensure_session_splits(
    corpus_root: Path,
    *,
    seed: int,
    holdout_session_ids: Sequence[str] | None = None,
    eval_fraction: float = 0.15,
) -> dict[str, str]:
    path = Path(corpus_root) / SPLITS_NAME
    requested_holdouts = {sid for sid in (holdout_session_ids or []) if sid}
    session_ids = _known_session_ids(corpus_root)

    if path.is_file():
        payload = _load_splits_payload(corpus_root)
        assignments = {str(k): str(v) for k, v in (payload.get("assignments") or {}).items()}
        stored_seed = int(payload.get("seed") or seed)
        stored_holdouts = {str(sid) for sid in (payload.get("holdout_session_ids") or [])}
        merged_holdouts = stored_holdouts | requested_holdouts
        changed = False

        for session_id in sorted(requested_holdouts):
            if session_id in assignments and assignments[session_id] != "holdout":
                assignments[session_id] = "holdout"
                changed = True

        for session_id in sorted(session_ids):
            if session_id in assignments:
                continue
            if session_id in merged_holdouts:
                assignments[session_id] = "holdout"
            else:
                fraction = _stable_fraction(session_id, stored_seed)
                assignments[session_id] = "eval" if fraction < eval_fraction else "train"
            changed = True

        if merged_holdouts != stored_holdouts:
            changed = True

        if changed:
            write_session_splits(
                corpus_root,
                assignments,
                seed=stored_seed,
                holdout_session_ids=sorted(merged_holdouts),
            )
        return assignments

    assignments = assign_session_splits(
        sorted(session_ids),
        seed=seed,
        holdout_session_ids=holdout_session_ids,
        eval_fraction=eval_fraction,
    )
    write_session_splits(
        corpus_root,
        assignments,
        seed=seed,
        holdout_session_ids=holdout_session_ids,
    )
    return assignments


@dataclass(frozen=True)
class CorpusExample:
    source_id: str
    session_id: str
    wav_path: Path
    label: str
    split: str
    origin: str = "corpus_event"
    meta: dict = field(default_factory=dict)


def _derive_split_examples(
    events: Sequence[CandidateEvent],
    session_splits: Mapping[str, str],
    *,
    target_split: str,
) -> list[CorpusExample]:
    examples: list[CorpusExample] = []
    for event in events:
        if event.label not in TRAIN_LABELS:
            continue
        if not event.session_id:
            continue
        split = session_splits.get(event.session_id)
        if split is None or split != target_split:
            continue
        wav = event.record_dir / "window.wav"
        if not wav.is_file():
            continue
        examples.append(
            CorpusExample(
                source_id=event.event_id,
                session_id=event.session_id,
                wav_path=wav,
                label=str(event.label),
                split=target_split,
                meta=event.to_row(),
            )
        )
    return examples


def derive_training_examples(
    events: Sequence[CandidateEvent],
    session_splits: Mapping[str, str],
) -> list[CorpusExample]:
    """Derive train examples from labeled events using session-level splits only."""
    return _derive_split_examples(events, session_splits, target_split="train")


def derive_eval_examples(
    events: Sequence[CandidateEvent],
    session_splits: Mapping[str, str],
) -> list[CorpusExample]:
    """Derive eval examples from labeled events on eval sessions."""
    return _derive_split_examples(events, session_splits, target_split="eval")


def derive_snapshot_examples(
    events: Sequence[CandidateEvent],
    session_splits: Mapping[str, str],
) -> list[CorpusExample]:
    """Derive train and eval examples; holdout sessions stay out of snapshots."""
    train = derive_training_examples(events, session_splits)
    eval_examples = derive_eval_examples(events, session_splits)
    assert_no_split_leak(train, eval_examples)
    return train + eval_examples


def holdout_sessions(session_splits: Mapping[str, str]) -> list[str]:
    return sorted(session_id for session_id, split in session_splits.items() if split == "holdout")


def assert_no_split_leak(
    train_examples: Sequence[CorpusExample],
    eval_examples: Sequence[CorpusExample],
) -> None:
    train_sessions = {example.session_id for example in train_examples}
    eval_sessions = {example.session_id for example in eval_examples}
    overlap = train_sessions & eval_sessions
    if overlap:
        raise RuntimeError(f"session split leak: {sorted(overlap)}")


def snapshot_manifest_path(corpus_root: Path, snapshot_id: str) -> Path:
    return Path(corpus_root) / "snapshots" / snapshot_id / SNAPSHOT_MANIFEST


def write_snapshot_manifest(
    corpus_root: Path,
    snapshot_id: str,
    *,
    examples: Sequence[CorpusExample],
    session_splits: Mapping[str, str],
    seed: int,
) -> Path:
    root = Path(corpus_root) / "snapshots" / snapshot_id
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "snapshot_id": snapshot_id,
        "seed": seed,
        "session_splits": dict(sorted(session_splits.items())),
        "examples": [
            {
                "source_id": ex.source_id,
                "session_id": ex.session_id,
                "label": ex.label,
                "split": ex.split,
                "wav_path": str(ex.wav_path),
            }
            for ex in sorted(examples, key=lambda item: item.source_id)
        ],
    }
    path = root / SNAPSHOT_MANIFEST
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def corpus_snapshot_id(examples: Sequence[CorpusExample], *, seed: int) -> str:
    payload = sorted((ex.source_id, ex.session_id, ex.label, str(ex.wav_path)) for ex in examples)
    digest = hashlib.sha256(json.dumps({"examples": payload, "seed": seed}, sort_keys=True).encode()).hexdigest()
    return digest[:16]
