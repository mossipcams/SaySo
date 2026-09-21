"""Long-form recording session ingest for the wake-word corpus pipeline."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import wave
from dataclasses import dataclass
from pathlib import Path

from .eval import read_wav_pcm
from .livekit import SAMPLE_RATE

SESSIONS_DIR = "sessions"
SESSION_MANIFEST = "session.json"
SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")


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


def _write_pcm_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".wav.tmp")
    with wave.open(str(tmp), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    tmp.replace(path)


def corpus_sessions_dir(corpus_root: Path) -> Path:
    return Path(corpus_root) / SESSIONS_DIR


def session_dir(corpus_root: Path, session_id: str) -> Path:
    return corpus_sessions_dir(corpus_root) / session_id


def _validate_session_id(session_id: str) -> str:
    if not SESSION_ID_RE.match(session_id):
        raise ValueError(f"invalid session id: {session_id!r}")
    return session_id


def derive_session_id(wav_path: Path) -> str:
    stem = wav_path.stem.strip() or "session"
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", stem).strip("._-")
    if not cleaned:
        cleaned = "session"
    digest = hashlib.sha256(str(wav_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return _validate_session_id(f"{cleaned}_{digest}")


@dataclass(frozen=True)
class RecordingSession:
    session_id: str
    audio_path: Path
    source_path: str
    sample_rate: int
    sample_count: int
    duration_seconds: float
    ingested_utc: str
    audio_sha256: str
    notes: str | None = None

    def to_dict(self) -> dict:
        payload = {
            "session_id": self.session_id,
            "audio_path": "audio.wav",
            "source_path": self.source_path,
            "sample_rate": self.sample_rate,
            "sample_count": self.sample_count,
            "duration_seconds": round(self.duration_seconds, 6),
            "ingested_utc": self.ingested_utc,
            "audio_sha256": self.audio_sha256,
        }
        if self.notes:
            payload["notes"] = self.notes
        return payload

    @classmethod
    def from_dict(cls, payload: dict, *, root: Path) -> RecordingSession:
        audio_name = str(payload.get("audio_path") or "audio.wav")
        return cls(
            session_id=str(payload["session_id"]),
            audio_path=root / audio_name,
            source_path=str(payload.get("source_path") or ""),
            sample_rate=int(payload["sample_rate"]),
            sample_count=int(payload["sample_count"]),
            duration_seconds=float(payload["duration_seconds"]),
            ingested_utc=str(payload.get("ingested_utc") or ""),
            audio_sha256=str(payload.get("audio_sha256") or ""),
            notes=str(payload["notes"]) if payload.get("notes") else None,
        )


def ingest_session(
    wav_path: Path,
    corpus_root: Path,
    *,
    session_id: str | None = None,
    notes: str | None = None,
    copy_audio: bool = True,
) -> RecordingSession:
    """Register a long-form WAV as a named recording session."""
    wav_path = Path(wav_path)
    if not wav_path.is_file():
        raise FileNotFoundError(f"missing session wav: {wav_path}")

    pcm, sample_rate = read_wav_pcm(wav_path, target_rate=SAMPLE_RATE)
    sample_count = len(pcm) // 2
    if sample_count < SAMPLE_RATE:
        raise ValueError(f"session too short for wake replay: {wav_path}")

    sid = _validate_session_id(session_id or derive_session_id(wav_path))
    root = session_dir(corpus_root, sid)
    root.mkdir(parents=True, exist_ok=True)
    dest_audio = root / "audio.wav"
    if copy_audio:
        _write_pcm_wav(dest_audio, pcm, sample_rate)
    else:
        if dest_audio.exists() and dest_audio.resolve() != wav_path.resolve():
            dest_audio.unlink()
        dest_audio.symlink_to(wav_path.resolve())

    session = RecordingSession(
        session_id=sid,
        audio_path=dest_audio,
        source_path=str(wav_path.resolve()),
        sample_rate=sample_rate,
        sample_count=sample_count,
        duration_seconds=sample_count / float(sample_rate),
        ingested_utc=_utc_stamp(),
        audio_sha256=_sha256_file(dest_audio),
        notes=notes,
    )
    _safe_json_write(root / SESSION_MANIFEST, session.to_dict())
    return session


def load_session(corpus_root: Path, session_id: str) -> RecordingSession:
    root = session_dir(corpus_root, session_id)
    manifest = root / SESSION_MANIFEST
    if not manifest.is_file():
        raise FileNotFoundError(f"missing session manifest: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    session = RecordingSession.from_dict(payload, root=root)
    if not session.audio_path.is_file():
        raise FileNotFoundError(f"missing session audio: {session.audio_path}")
    return session


def list_sessions(corpus_root: Path) -> list[RecordingSession]:
    root = corpus_sessions_dir(corpus_root)
    if not root.is_dir():
        return []
    sessions: list[RecordingSession] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        manifest = entry / SESSION_MANIFEST
        if not manifest.is_file():
            continue
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            sessions.append(RecordingSession.from_dict(payload, root=entry))
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            continue
    return sessions


def read_session_pcm(session: RecordingSession) -> bytes:
    pcm, rate = read_wav_pcm(session.audio_path, target_rate=SAMPLE_RATE)
    if rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz session audio, got {rate}")
    return pcm


def verify_session(session: RecordingSession) -> tuple[bool, str]:
    if not session.audio_path.is_file():
        return False, "missing audio.wav"
    digest = _sha256_file(session.audio_path)
    if session.audio_sha256 and session.audio_sha256 != digest:
        return False, "audio hash mismatch"
    try:
        with wave.open(str(session.audio_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
    except (OSError, wave.Error) as exc:
        return False, f"corrupt audio: {exc}"
    if frames != session.sample_count or rate != session.sample_rate:
        return False, "manifest/audio metadata mismatch"
    return True, "ok"
