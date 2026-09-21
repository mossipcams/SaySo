"""Wake corpus candidate events derived from long-form session replay."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

if TYPE_CHECKING:
    from .livekit import LiveKitWakeWordProvider

from .mining import (
    SAMPLING_BELOW,
    SAMPLING_DETECTION,
    SAMPLING_NEAR,
    ingest_record,
)
from .replay import ReplayConfig, extract_context_pcm
from .sessions import RecordingSession, read_session_pcm

EVENTS_DIR = "events"
LABELS = ("positive", "negative", "unsure")
TRAIN_LABELS = frozenset({"positive", "negative"})


def _utc_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_json_write(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_window_samples(window_path: Path) -> np.ndarray:
    with wave.open(str(window_path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit PCM in {window_path}")
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype="<i2")


def score_window_verifier(provider: LiveKitWakeWordProvider, window_path: Path) -> float | None:
    """Score the exact 2s window with the provider's production verifier."""
    verifier = getattr(provider, "_verifier", None)
    model = getattr(provider, "_model", None)
    if verifier is None or model is None:
        return None
    window = _read_window_samples(window_path)
    score = verifier.score(window, model)
    return float(score) if score is not None else None


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".wav.tmp")
    pcm = np.ascontiguousarray(samples, dtype="<i2")
    with wave.open(str(tmp), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    tmp.replace(path)


def corpus_events_dir(corpus_root: Path) -> Path:
    return Path(corpus_root) / EVENTS_DIR


def event_dir(corpus_root: Path, event_id: str) -> Path:
    return corpus_events_dir(corpus_root) / event_id


@dataclass(frozen=True)
class CandidateEvent:
    event_id: str
    session_id: str
    sample_start: int
    sample_end: int
    livekit_score: float
    verifier_score: float | None
    sampling_reason: str
    label: str | None
    model_path: str | None
    model_sha256: str | None
    provider: str
    detect_threshold: float
    mine_threshold: float
    fired: bool
    record_dir: Path

    @property
    def capture_id(self) -> str:
        return self.event_id

    def to_row(self) -> dict[str, Any]:
        return {
            "capture_id": self.event_id,
            "session_id": self.session_id,
            "score": self.livekit_score,
            "verifier_score": self.verifier_score,
            "sampling_reason": self.sampling_reason,
            "label": self.label,
            "fired": self.fired,
            "_record_dir": self.record_dir,
            "_wav": self.record_dir / "window.wav",
            "_json": self.record_dir / "record.json",
        }


def _parse_event(record_dir: Path) -> CandidateEvent | None:
    meta_path = record_dir / "record.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    event_id = str(meta.get("capture_id") or meta.get("event_id") or record_dir.name)
    label = meta.get("label")
    if label is not None:
        label = str(label)
    verifier = meta.get("verifier_score")
    return CandidateEvent(
        event_id=event_id,
        session_id=str(meta.get("session_id") or ""),
        sample_start=int(meta.get("sample_start") or 0),
        sample_end=int(meta.get("sample_end") or 0),
        livekit_score=float(meta.get("score") or 0.0),
        verifier_score=float(verifier) if verifier is not None else None,
        sampling_reason=str(meta.get("sampling_reason") or ""),
        label=label,
        model_path=str(meta.get("model_path")) if meta.get("model_path") else None,
        model_sha256=str(meta.get("model_sha256")) if meta.get("model_sha256") else None,
        provider=str(meta.get("provider") or "livekit"),
        detect_threshold=float(meta.get("detect_threshold") or 0.0),
        mine_threshold=float(meta.get("mine_threshold") or 0.0),
        fired=bool(meta.get("fired")),
        record_dir=record_dir,
    )


def load_events(corpus_root: Path) -> list[CandidateEvent]:
    root = corpus_events_dir(corpus_root)
    if not root.is_dir():
        return []
    events: list[CandidateEvent] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        event = _parse_event(entry)
        if event is not None:
            events.append(event)
    return events


def load_labeled_train_events(corpus_root: Path) -> list[CandidateEvent]:
    return [event for event in load_events(corpus_root) if event.label in TRAIN_LABELS]


def set_event_label(corpus_root: Path, event_id: str, label: str) -> CandidateEvent:
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")
    record_dir = event_dir(corpus_root, event_id)
    meta_path = record_dir / "record.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing event record: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["label"] = label
    meta["labeled_utc"] = _utc_stamp()
    _safe_json_write(meta_path, meta)
    event = _parse_event(record_dir)
    if event is None:
        raise RuntimeError(f"failed to reload labeled event {event_id}")
    return event


def import_spool_record(
    record_dir: Path,
    corpus_root: Path,
    *,
    session: RecordingSession | None = None,
    provider: LiveKitWakeWordProvider | None = None,
    verifier_score: float | None = None,
    config: ReplayConfig | None = None,
) -> CandidateEvent:
    """Copy a verified mining record into the corpus with optional context enrichment."""
    ok, message = ingest_record(record_dir)
    if not ok:
        raise ValueError(f"cannot import record: {message}")
    meta = json.loads((record_dir / "record.json").read_text(encoding="utf-8"))
    event_id = str(meta.get("capture_id") or record_dir.name)
    dest = event_dir(corpus_root, event_id)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(record_dir, dest)

    if verifier_score is None and provider is not None:
        verifier_score = score_window_verifier(provider, dest / "window.wav")
    if verifier_score is not None:
        meta = json.loads((dest / "record.json").read_text(encoding="utf-8"))
        meta["verifier_score"] = round(float(verifier_score), 6)
        _safe_json_write(dest / "record.json", meta)

    if session is not None and (dest / "context.wav").is_file() is False:
        pcm = read_session_pcm(session)
        sample_end = int(meta.get("sample_end") or 0)
        if sample_end > 0:
            context_pcm = extract_context_pcm(pcm, sample_end, config=config)
            context = np.frombuffer(context_pcm, dtype="<i2")
            if context.size:
                _write_wav(dest / "context.wav", context, session.sample_rate)
                meta = json.loads((dest / "record.json").read_text(encoding="utf-8"))
                hashes = dict(meta.get("hashes") or {})
                hashes["context_sha256"] = _sha256_file(dest / "context.wav")
                meta["hashes"] = hashes
                _safe_json_write(dest / "record.json", meta)

    event = _parse_event(dest)
    if event is None:
        raise RuntimeError(f"failed to import event {event_id}")
    return event


def import_spool_records(
    spool_root: Path,
    corpus_root: Path,
    *,
    sessions_by_id: dict[str, RecordingSession] | None = None,
    provider: LiveKitWakeWordProvider | None = None,
) -> list[CandidateEvent]:
    records_dir = spool_root / "records"
    if not records_dir.is_dir():
        return []
    imported: list[CandidateEvent] = []
    sessions = sessions_by_id or {}
    for record_dir in sorted(records_dir.iterdir()):
        if not record_dir.is_dir():
            continue
        meta_path = record_dir / "record.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        session_id = str(meta.get("session_id") or "")
        session = sessions.get(session_id)
        imported.append(
            import_spool_record(record_dir, corpus_root, session=session, provider=provider)
        )
    return imported


def events_for_session(events: Iterable[CandidateEvent], session_id: str) -> list[CandidateEvent]:
    return [event for event in events if event.session_id == session_id]


def prune_unlabeled_session_events(corpus_root: Path, session_id: str) -> int:
    """Remove unlabeled corpus events for one session before a fresh replay import."""
    removed = 0
    for event in events_for_session(load_events(corpus_root), session_id):
        if event.label is not None or _has_manual_label(event):
            continue
        shutil.rmtree(event.record_dir)
        removed += 1
    return removed


def labeled_positive_events(
    events: Iterable[CandidateEvent],
    session_id: str,
) -> list[CandidateEvent]:
    return [
        event
        for event in events_for_session(events, session_id)
        if event.label == "positive" and _has_manual_label(event)
    ]


def assert_no_automatic_positive(events: Iterable[CandidateEvent]) -> None:
    for event in events:
        if event.label == "positive" and not _has_manual_label(event):
            raise RuntimeError(f"automatic positive label forbidden: {event.event_id}")


def _has_manual_label(event: CandidateEvent) -> bool:
    meta_path = event.record_dir / "record.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(meta.get("labeled_utc"))


def new_event_id() -> str:
    return str(uuid.uuid4())
