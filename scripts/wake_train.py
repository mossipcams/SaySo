#!/usr/bin/env python3
"""Batch wake-word training: ingest -> select -> snapshot -> train -> evaluate -> save.

Conservative auto-labeling, immutable snapshots, run-key idempotency, and local
candidate bundles. Default tests use stub teacher/trainer paths (no LiveKit,
Faster Whisper, or CUDA). Scheduling stays disabled while baseline/seed gates fail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SATELLITE_ROOT = _REPO_ROOT / "satellite"
if str(_SATELLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SATELLITE_ROOT))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sayso.wake.eval import (  # noqa: E402
    run_wake_eval,
    write_synthetic_wav,
)
from sayso.wake.livekit import HOP_SAMPLES, SAMPLE_RATE, WINDOW_SAMPLES
from sayso.wake.snapshot import (  # noqa: E402
    CorpusExample,
    derive_training_examples,
    ensure_session_splits,
    holdout_sessions,
    load_session_splits,
)
from scripts.wake_mine_report import load as load_spool_records  # noqa: E402

TEACHER_POLICY_VERSION = "conservative-v1"
LABEL_POLICY_VERSION = "conservative-v1"
FRONTEND_VERSION = "livekit-wakeword-0.2.1"
WEAK_NEGATIVE_FRACTION_CAP = 0.25
FA_PER_HOUR_BOUND = 0.02
MIN_BACKGROUND_HOURS_FOR_QUAL = 150.0
WINDOW_SECONDS = 2.0

FINAL_STATUSES = frozenset(
    {"trained", "evaluated", "rejected", "insufficient_evidence", "qualified"}
)
TRUSTED_LABELS = frozenset({"positive", "negative"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(text.encode("utf-8"))


def _read_wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def repo_eval_root() -> Path:
    return _REPO_ROOT / "satellite" / "eval"


def default_recipe_path() -> Path:
    return _REPO_ROOT / "satellite" / "models" / "sayso-training.yaml"


def default_baseline_path() -> Path:
    return repo_eval_root() / "baseline.json"


def default_splits_path() -> Path:
    return repo_eval_root() / "splits.json"


def clip_basename(index: int) -> str:
    return f"clip_{index:06d}"


@dataclass(frozen=True)
class TeacherVerdict:
    label: str | None
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)


class Teacher(Protocol):
    version: str

    def label_window(self, wav_path: Path, meta: dict[str, Any]) -> TeacherVerdict: ...


class Trainer(Protocol):
    version: str

    def train_and_export(self, snapshot: Snapshot, output_dir: Path) -> TrainOutput: ...


class StubTeacher:
    """Deterministic conservative teacher for hermetic tests."""

    version = "stub-v1"

    def __init__(
        self,
        *,
        fail: bool = False,
        homophone_unknown: bool = True,
    ) -> None:
        self.fail = fail
        self.homophone_unknown = homophone_unknown

    def label_window(self, wav_path: Path, meta: dict[str, Any]) -> TeacherVerdict:
        if self.fail:
            raise RuntimeError("teacher failure injected")
        manual = meta.get("label")
        if manual in TRUSTED_LABELS:
            return TeacherVerdict(str(manual), "trusted_manual", {"source": "manual"})
        phrase_hint = str(meta.get("phrase_hint") or meta.get("sampling_reason") or "")
        if self.homophone_unknown and "say_so" in phrase_hint:
            return TeacherVerdict(None, "homophone_ambiguous", {"phrase_hint": phrase_hint})
        if meta.get("quality_flags", {}).get("clipped"):
            return TeacherVerdict(None, "clipping", meta.get("quality_flags", {}))
        if meta.get("asr_empty"):
            return TeacherVerdict(None, "empty_asr", {})
        if meta.get("teacher_disagreement"):
            return TeacherVerdict(None, "teacher_disagreement", {})
        peak = _wav_peak(path=wav_path)
        if peak < 100:
            return TeacherVerdict("negative", "quiet_window", {"peak": peak})
        if peak >= 8000 and meta.get("sampling_reason") == "below_threshold":
            return TeacherVerdict(None, "uncertain_alignment", {"peak": peak})
        if peak >= 5000:
            return TeacherVerdict(None, "no_automatic_positive", {"peak": peak})
        return TeacherVerdict("negative", "conservative_negative", {"peak": peak})


def _wav_peak(*, path: Path) -> int:
    with wave.open(str(path), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    if not frames:
        return 0
    samples = np.frombuffer(frames, dtype="<i2")
    return int(np.max(np.abs(samples))) if samples.size else 0


_HOMOPHONE_PATTERNS = (
    "say so",
    "says so",
    "said so",
    "saying so",
    "if you say so",
    "just say so",
    "you don't say so",
    "so so",
    "stay so",
    "same so",
    "safe so",
    "say slow",
    "say show",
    "say sorry",
    "say something",
    "essay so",
    "hey so",
    "they say so",
    "lasso",
)
_WAKE_PHRASE_RE = re.compile(r"\bsay\s*so\b|\bsayso\b", re.IGNORECASE)


def _load_whisper_model_class() -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is required for non-stub training; "
            "install satellite/models/requirements-wake-train.txt on the host"
        ) from exc
    return WhisperModel


def _normalize_transcript(text: str) -> str:
    cleaned = text.lower().strip()
    cleaned = re.sub(r"[^\w\s']", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _homophone_ambiguous(text: str, meta: Mapping[str, Any]) -> bool:
    phrase_hint = str(meta.get("phrase_hint") or meta.get("sampling_reason") or "")
    if "say_so" in phrase_hint:
        return True
    return any(pattern in text for pattern in _HOMOPHONE_PATTERNS)


def _wake_phrase_detected(text: str) -> bool:
    return bool(_WAKE_PHRASE_RE.search(text))


class FasterWhisperTeacher:
    """Conservative ASR teacher with word timestamps; never emits positives."""

    version = "faster-whisper-v1"

    def __init__(
        self,
        *,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        whisper_model = _load_whisper_model_class()
        self._model = whisper_model(model_size, device=device, compute_type=compute_type)

    def label_window(self, wav_path: Path, meta: dict[str, Any]) -> TeacherVerdict:
        duration = _read_wav_duration_seconds(wav_path)
        if abs(duration - WINDOW_SECONDS) > 0.05:
            return TeacherVerdict(
                None,
                "uncertain_alignment",
                {"duration_seconds": duration, "expected_seconds": WINDOW_SECONDS},
            )
        if meta.get("quality_flags", {}).get("clipped"):
            return TeacherVerdict(None, "clipping", meta.get("quality_flags", {}))
        if meta.get("teacher_disagreement"):
            return TeacherVerdict(None, "teacher_disagreement", {})
        if meta.get("asr_empty"):
            return TeacherVerdict(None, "empty_asr", {})

        segments, info = self._model.transcribe(
            str(wav_path),
            word_timestamps=True,
            vad_filter=False,
        )
        segment_list = list(segments)
        transcript = _normalize_transcript(" ".join(segment.text for segment in segment_list))
        words: list[dict[str, Any]] = []
        for segment in segment_list:
            if not segment.words:
                continue
            for word in segment.words:
                words.append(
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability,
                    }
                )

        evidence = {
            "transcript": transcript,
            "words": words,
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
            "no_speech_prob": getattr(info, "no_speech_prob", None),
        }

        if not transcript:
            return TeacherVerdict(None, "empty_asr", evidence)

        language_probability = float(getattr(info, "language_probability", 1.0) or 0.0)
        no_speech_prob = float(getattr(info, "no_speech_prob", 0.0) or 0.0)
        if language_probability < 0.5 or no_speech_prob > 0.6:
            return TeacherVerdict(None, "low_quality", evidence)

        if words:
            word_probs = [float(word["probability"]) for word in words if word.get("probability") is not None]
            if word_probs and float(np.mean(word_probs)) < 0.35:
                return TeacherVerdict(None, "low_quality", evidence)

        if _homophone_ambiguous(transcript, meta):
            return TeacherVerdict(None, "homophone_ambiguous", evidence)

        if _wake_phrase_detected(transcript):
            return TeacherVerdict(None, "no_automatic_positive", evidence)

        return TeacherVerdict("negative", "conservative_negative", evidence)


@dataclass(frozen=True)
class Example:
    source_id: str
    wav_path: Path
    label: str
    split: str
    origin: str
    weak_negative: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str
    root: Path
    mapping: dict[str, str]
    recipe_path: Path
    counts: dict[str, int]
    hashes: dict[str, str]
    seed: int


@dataclass(frozen=True)
class TrainOutput:
    model_path: Path
    threshold: float
    provider: str
    metrics: dict[str, Any]


@dataclass(frozen=True)
class PipelineResult:
    run_key: str
    status: str
    bundle_dir: Path | None
    noop: bool = False
    reason: str | None = None
    comparison: dict[str, Any] | None = None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_baseline(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"version": 0, "status": "blocked", "blocker": f"missing {path}"}
    return load_json(path)


def load_splits(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing splits contract: {path}")
    return load_json(path)


def trusted_seed_present(seed_dir: Path | None) -> bool:
    if seed_dir is None or not seed_dir.is_dir():
        return False
    pos = list((seed_dir / "positive").glob("*.wav"))
    neg = list((seed_dir / "negative").glob("*.wav"))
    return bool(pos) and bool(neg)


def eval_audio_present(eval_root: Path) -> bool:
    cases_path = eval_root / "cases.json"
    if not cases_path.is_file():
        return False
    cases = load_json(cases_path).get("cases", [])
    for entry in cases:
        rel = entry.get("audio")
        if rel and (eval_root / str(rel)).is_file():
            return True
    return False


def feasibility_gate(*, baseline: dict[str, Any], seed_dir: Path | None, eval_root: Path) -> dict[str, Any]:
    blockers: list[str] = []
    if baseline.get("status") == "blocked":
        blockers.append(str(baseline.get("blocker") or "baseline blocked"))
    if not trusted_seed_present(seed_dir):
        blockers.append("trusted seed recordings missing")
    if not eval_audio_present(eval_root):
        blockers.append("satellite/eval/audio fixtures absent")
    return {
        "ok": not blockers,
        "blockers": blockers,
        "scheduling_enabled": False,
    }


def ingest_sources(
    spool: Path,
    seed_dir: Path | None,
    *,
    corpus_root: Path | None = None,
    corpus_seed: int = 42,
    corpus_holdouts: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if spool.is_dir():
        rows.extend(load_spool_records(spool))
    if seed_dir and seed_dir.is_dir():
        for label in ("positive", "negative"):
            for wav in sorted((seed_dir / label).glob("*.wav")):
                rows.append(
                    {
                        "capture_id": f"seed_{label}_{wav.stem}",
                        "label": label,
                        "split": "train",
                        "origin": "trusted_seed",
                        "_wav": wav,
                        "_record_dir": None,
                    }
                )
    if corpus_root and corpus_root.is_dir():
        rows.extend(ingest_corpus_rows(corpus_root, seed=corpus_seed, holdout_session_ids=corpus_holdouts))
    return rows


def ingest_corpus_rows(
    corpus_root: Path,
    *,
    seed: int,
    holdout_session_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    from sayso.wake.corpus import load_events

    session_splits = ensure_session_splits(
        corpus_root,
        seed=seed,
        holdout_session_ids=holdout_session_ids,
    )
    examples = derive_training_examples(load_events(corpus_root), session_splits)
    rows: list[dict[str, Any]] = []
    for example in examples:
        row = dict(example.meta)
        row.update(
            {
                "capture_id": example.source_id,
                "session_id": example.session_id,
                "label": example.label,
                "split": example.split,
                "origin": example.origin,
                "_wav": example.wav_path,
                "_record_dir": row.get("_record_dir"),
            }
        )
        rows.append(row)
    return rows


def corpus_examples_for_snapshot(corpus_root: Path, *, seed: int) -> tuple[list[CorpusExample], dict[str, str]]:
    from sayso.wake.corpus import load_events

    session_splits = ensure_session_splits(corpus_root, seed=seed)
    examples = derive_training_examples(load_events(corpus_root), session_splits)
    return examples, session_splits


def label_records(
    rows: Sequence[dict[str, Any]],
    teacher: Teacher,
) -> tuple[list[Example], dict[str, Any]]:
    examples: list[Example] = []
    audit = {
        "teacher_version": teacher.version,
        "policy_version": LABEL_POLICY_VERSION,
        "unknown": 0,
        "trusted": 0,
        "weak_negative": 0,
        "automatic_positive": 0,
    }
    for row in rows:
        wav = row.get("_wav")
        if not isinstance(wav, Path) or not wav.is_file():
            continue
        capture_id = str(row.get("capture_id") or wav.stem)
        manual = row.get("label")
        if manual in TRUSTED_LABELS and row.get("origin") == "trusted_seed":
            examples.append(
                Example(
                    source_id=capture_id,
                    wav_path=wav,
                    label=str(manual),
                    split=str(row.get("split") or "train"),
                    origin="trusted_seed",
                    weak_negative=False,
                    meta=dict(row),
                )
            )
            audit["trusted"] += 1
            continue
        if manual in TRUSTED_LABELS:
            examples.append(
                Example(
                    source_id=capture_id,
                    wav_path=wav,
                    label=str(manual),
                    split=str(row.get("split") or "train"),
                    origin="manual",
                    weak_negative=False,
                    meta=dict(row),
                )
            )
            audit["trusted"] += 1
            continue
        verdict = teacher.label_window(wav, row)
        if verdict.label == "positive":
            audit["automatic_positive"] += 1
            audit["unknown"] += 1
            continue
        if verdict.label is None:
            audit["unknown"] += 1
            continue
        if verdict.label == "negative":
            examples.append(
                Example(
                    source_id=capture_id,
                    wav_path=wav,
                    label="negative",
                    split=str(row.get("split") or "train"),
                    origin="teacher",
                    weak_negative=True,
                    meta={**dict(row), "teacher_reason": verdict.reason, "teacher_evidence": verdict.evidence},
                )
            )
            audit["weak_negative"] += 1
    if audit["automatic_positive"]:
        raise RuntimeError("automatic positive labels are out of scope")
    return examples, audit


def _holdout_source_ids(splits: dict[str, Any]) -> set[str]:
    holdouts: set[str] = set()
    for group in splits.get("groups", []):
        if group.get("holdout"):
            holdouts.update(str(member) for member in group.get("members", []))
            holdouts.add(str(group.get("id", "")))
    return {item for item in holdouts if item}


def select_examples(
    examples: Sequence[Example],
    splits: dict[str, Any],
    *,
    require_known_real: str | None = None,
    corpus_root: Path | None = None,
) -> list[Example]:
    holdouts = _holdout_source_ids(splits)
    if corpus_root and corpus_root.is_dir():
        for session_id in holdout_sessions(load_session_splits(corpus_root)):
            holdouts.add(session_id)
    train: list[Example] = []
    weak_negs: list[Example] = []
    seen: set[str] = set()
    for ex in examples:
        if ex.source_id in holdouts:
            raise RuntimeError(f"split leakage: holdout source {ex.source_id}")
        session_id = str(ex.meta.get("session_id") or "")
        if session_id in holdouts:
            continue
        if ex.split != "train":
            continue
        if ex.source_id in seen:
            continue
        seen.add(ex.source_id)
        if ex.weak_negative:
            weak_negs.append(ex)
        else:
            train.append(ex)
    max_weak = max(1, int(len(train) * WEAK_NEGATIVE_FRACTION_CAP))
    train.extend(weak_negs[:max_weak])
    positives = [ex for ex in train if ex.label == "positive"]
    negatives = [ex for ex in train if ex.label == "negative"]
    if not positives or not negatives:
        raise RuntimeError("nonempty positive and negative classes required")
    if require_known_real and not any(ex.source_id == require_known_real for ex in train):
        raise RuntimeError(f"known real capture {require_known_real} did not reach training")
    return train


def snapshot_content_id(
    examples: Sequence[Example],
    *,
    recipe_path: Path,
    seed: int,
    teacher_audit: dict[str, Any],
) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "examples": [(ex.source_id, ex.label, str(ex.wav_path)) for ex in examples],
                "recipe": str(recipe_path),
                "seed": seed,
                "teacher": teacher_audit,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]


def build_snapshot(
    examples: Sequence[Example],
    *,
    work_dir: Path,
    recipe_path: Path,
    seed: int,
    teacher_audit: dict[str, Any],
    snapshot_id: str | None = None,
) -> Snapshot:
    snapshot_id = snapshot_id or snapshot_content_id(
        examples, recipe_path=recipe_path, seed=seed, teacher_audit=teacher_audit
    )
    root = work_dir / "snapshots" / snapshot_id
    clips = root / "clips"
    if root.exists():
        shutil.rmtree(root)
    clips.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    clip_labels: dict[str, str] = {}
    class_counts = {"positive": 0, "negative": 0}
    feature_paths: list[str] = []
    for index, ex in enumerate(sorted(examples, key=lambda item: item.source_id), start=1):
        clip = clip_basename(index)
        wav_name = f"{clip}.wav"
        feature_name = f"{clip}_r0.wav"
        dest = clips / wav_name
        shutil.copy2(ex.wav_path, dest)
        shutil.copy2(ex.wav_path, clips / feature_name)
        mapping[ex.source_id] = clip
        clip_labels[clip] = ex.label
        class_counts[ex.label] = class_counts.get(ex.label, 0) + 1
        feature_paths.append(feature_name)
    if len(feature_paths) != len(examples):
        raise RuntimeError("source-to-feature count mismatch")
    recipe_copy = root / "recipe.yaml"
    shutil.copy2(recipe_path, recipe_copy)
    manifest = {
        "snapshot_id": snapshot_id,
        "created_utc": _utc_now(),
        "mapping": mapping,
        "clip_labels": clip_labels,
        "class_counts": class_counts,
        "source_count": len(examples),
        "feature_count": len(feature_paths),
        "teacher_audit": teacher_audit,
        "seed": seed,
        "frontend_version": FRONTEND_VERSION,
        "hashes": {
            "recipe_sha256": _sha256_file(recipe_copy),
            "clips_sha256": _sha256_json({"clips": sorted(str(p) for p in clips.glob("*.wav"))}),
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "mapping.json").write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return Snapshot(
        snapshot_id=snapshot_id,
        root=root,
        mapping=mapping,
        recipe_path=recipe_copy,
        counts=class_counts,
        hashes=manifest["hashes"],
        seed=seed,
    )


def compute_run_key(
    *,
    snapshot_id: str,
    recipe_sha256: str,
    clips_sha256: str,
    teacher_audit: dict[str, Any],
    stub_mode: bool,
    seed: int,
    deps_hash: str,
) -> str:
    payload = {
        "snapshot_id": snapshot_id,
        "recipe_sha256": recipe_sha256,
        "clips_sha256": clips_sha256,
        "teacher_version": teacher_audit.get("teacher_version"),
        "policy_version": teacher_audit.get("policy_version"),
        "frontend_version": FRONTEND_VERSION,
        "seed": seed,
        "stub_mode": stub_mode,
        "deps_hash": deps_hash,
    }
    return _sha256_json(payload)


def preview_run_key(
    examples: Sequence[Example],
    *,
    recipe_path: Path,
    teacher_audit: dict[str, Any],
    stub_mode: bool,
    seed: int,
    deps_hash: str,
) -> tuple[str, str, str]:
    snapshot_id = snapshot_content_id(examples, recipe_path=recipe_path, seed=seed, teacher_audit=teacher_audit)
    recipe_sha256 = _sha256_file(recipe_path)
    clips_sha256 = _sha256_json(
        {
            "examples": sorted(
                (ex.source_id, ex.label, _sha256_file(ex.wav_path)) for ex in examples
            )
        }
    )
    run_key = compute_run_key(
        snapshot_id=snapshot_id,
        recipe_sha256=recipe_sha256,
        clips_sha256=clips_sha256,
        teacher_audit=teacher_audit,
        stub_mode=stub_mode,
        seed=seed,
        deps_hash=deps_hash,
    )
    return run_key, snapshot_id, clips_sha256


def deps_hash_from_requirements(path: Path) -> str:
    if not path.is_file():
        return "missing-requirements"
    return _sha256_file(path)


def _run_dir(work_dir: Path, run_key: str) -> Path:
    return work_dir / "runs" / run_key


def _load_run_state(run_dir: Path) -> dict[str, Any] | None:
    state_path = run_dir / "run.json"
    if not state_path.is_file():
        return None
    return load_json(state_path)


def _write_run_state(run_dir: Path, payload: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def production_refractory_seconds() -> float:
    try:
        from satellite.sayso.config import load_config

        return float(load_config().wake_word.refractory_seconds)
    except Exception:
        return 2.0


def acquire_lock(work_dir: Path) -> Path:
    lock = work_dir / ".wake_train.lock"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(lock), flags)
        try:
            os.write(fd, str(os.getpid()).encode("utf-8"))
        finally:
            os.close(fd)
    except FileExistsError:
        raise RuntimeError("another wake_train run holds the lock") from None
    return lock


def release_lock(lock: Path) -> None:
    if lock.is_file():
        lock.unlink()


class StubTrainer:
    version = "stub-v1"

    def __init__(
        self,
        *,
        export_ok: bool = True,
        quality: str = "good",
    ) -> None:
        self.export_ok = export_ok
        self.quality = quality

    def train_and_export(self, snapshot: Snapshot, output_dir: Path) -> TrainOutput:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "sayso.onnx"
        if not self.export_ok:
            raise RuntimeError("export failure injected")
        model_path.write_bytes(b"stub-onnx-" + snapshot.snapshot_id.encode("utf-8"))
        threshold = 0.45 if self.quality == "good" else 0.95
        metrics = {
            "recall": 0.8 if self.quality == "good" else 0.2,
            "fp_per_hour": 0.01 if self.quality == "good" else 0.5,
            "threshold": threshold,
            "provider": "livekit",
            "stub": True,
        }
        (output_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        return TrainOutput(
            model_path=model_path,
            threshold=threshold,
            provider="livekit",
            metrics=metrics,
        )


def _import_livekit_wakeword() -> tuple[Any, Any, Any, Any]:
    try:
        from livekit.wakeword import load_config, run_export, run_train
        from livekit.wakeword.data.features import run_extraction
    except ImportError as exc:
        raise RuntimeError(
            "livekit-wakeword is required for non-stub training; "
            "install satellite/models/requirements-wake-train.txt on the host"
        ) from exc
    return load_config, run_extraction, run_train, run_export


def _run_livekit_cli(stage: str, config_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "livekit.wakeword", stage, str(config_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip() or f"exit code {proc.returncode}"
        raise RuntimeError(f"livekit.wakeword {stage} failed: {detail}")


def _stage_snapshot_clips(snapshot: Snapshot, model_dir: Path) -> dict[str, str]:
    manifest = load_json(snapshot.root / "manifest.json")
    clip_labels = manifest.get("clip_labels")
    if not isinstance(clip_labels, dict) or not clip_labels:
        raise RuntimeError("snapshot manifest missing clip_labels for LiveKit staging")

    clips_dir = snapshot.root / "clips"
    split_dirs = {
        "positive": (model_dir / "positive_train", model_dir / "positive_test"),
        "negative": (model_dir / "negative_train", model_dir / "negative_test"),
    }
    for train_dir, test_dir in split_dirs.values():
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

    by_label: dict[str, list[str]] = {"positive": [], "negative": []}
    for clip, label in sorted(clip_labels.items()):
        if label not in by_label:
            raise RuntimeError(f"unsupported clip label {label!r} in snapshot")
        by_label[label].append(clip)

    for label, clips in by_label.items():
        train_dir, test_dir = split_dirs[label]
        if len(clips) >= 2:
            train_clips = clips[:-1]
            holdout = clips[-1]
        else:
            train_clips = clips
            holdout = None
        for clip in train_clips:
            for suffix in (".wav", "_r0.wav"):
                src = clips_dir / f"{clip}{suffix}"
                if not src.is_file():
                    raise RuntimeError(f"missing snapshot clip {src.name}")
                shutil.copy2(src, train_dir / src.name)
        if holdout is not None:
            for suffix in (".wav", "_r0.wav"):
                src = clips_dir / f"{holdout}{suffix}"
                shutil.copy2(src, test_dir / src.name)

    return {str(clip): str(label) for clip, label in clip_labels.items()}


def _write_livekit_config(snapshot: Snapshot, work_root: Path) -> Path:
    recipe = yaml.safe_load(snapshot.recipe_path.read_text(encoding="utf-8"))
    if not isinstance(recipe, dict):
        raise RuntimeError("snapshot recipe must be a YAML mapping")
    model_name = str(recipe.get("model_name") or "sayso")
    model_dir = work_root / "output" / model_name
    _stage_snapshot_clips(snapshot, model_dir)
    recipe["data_dir"] = str(work_root / "data")
    recipe["output_dir"] = str(work_root / "output")
    config_path = work_root / "livekit_config.yaml"
    config_path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")
    return config_path


def _load_livekit_threshold(config_path: Path) -> tuple[float, dict[str, Any]]:
    recipe = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_name = str(recipe.get("model_name") or "sayso")
    model_dir = Path(str(recipe.get("output_dir") or "./output")) / model_name
    metrics_path = model_dir / f"{model_name}_metrics.json"
    threshold = 0.5
    metrics: dict[str, Any] = {"provider": "livekit", "stub": False, "frontend_version": FRONTEND_VERSION}
    if metrics_path.is_file():
        log = json.loads(metrics_path.read_text(encoding="utf-8"))
        if isinstance(log, list):
            for entry in reversed(log):
                if isinstance(entry, dict) and entry.get("note") == "optimal_threshold":
                    threshold = float(entry.get("threshold", threshold))
                    metrics.update(entry)
                    break
    return threshold, metrics


class LiveKitTrainer:
    """Pinned livekit-wakeword 0.2.1 train/export against snapshot clip layout."""

    version = FRONTEND_VERSION

    def train_and_export(self, snapshot: Snapshot, output_dir: Path) -> TrainOutput:
        load_config, run_extraction, _run_train_api, _run_export_api = _import_livekit_wakeword()
        work_root = output_dir / "livekit_workspace"
        if work_root.exists():
            shutil.rmtree(work_root)
        work_root.mkdir(parents=True, exist_ok=True)
        config_path = _write_livekit_config(snapshot, work_root)
        config = load_config(config_path)
        try:
            run_extraction(config)
            _run_livekit_cli("train", config_path)
            _run_livekit_cli("export", config_path)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"livekit-wakeword train/export failed: {exc}") from exc

        model_name = str(config.model_name)
        exported = config.model_output_dir / f"{model_name}.onnx"
        if not exported.is_file():
            raise RuntimeError(f"livekit-wakeword export missing ONNX at {exported}")

        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "sayso.onnx"
        shutil.copy2(exported, model_path)
        threshold, metrics = _load_livekit_threshold(config_path)
        metrics_path = output_dir / "train_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return TrainOutput(
            model_path=model_path,
            threshold=threshold,
            provider="livekit",
            metrics=metrics,
        )


def _resolve_teacher(stub_mode: bool, teacher: Teacher | None) -> Teacher:
    if teacher is not None:
        return teacher
    if stub_mode:
        return StubTeacher()
    return FasterWhisperTeacher()


def _resolve_trainer(stub_mode: bool, trainer: Trainer | None) -> Trainer:
    if trainer is not None:
        return trainer
    if stub_mode:
        return StubTrainer()
    return LiveKitTrainer()


def select_calibration_threshold(
    *,
    calibration_metrics: dict[str, float] | None,
    default: float,
) -> float:
    if not calibration_metrics:
        return default
    threshold = calibration_metrics.get("threshold")
    if threshold is None:
        return default
    return float(threshold)


def _category_recall(report: dict[str, Any]) -> dict[str, float]:
    by_category: dict[str, list[bool]] = {}
    for entry in report.get("results", []):
        category = str(entry.get("category", "unknown"))
        ok = entry.get("detection_ok")
        if ok is None:
            continue
        by_category.setdefault(category, []).append(bool(ok))
    return {
        category: (sum(1 for item in values if item) / len(values) if values else 0.0)
        for category, values in by_category.items()
    }


def _false_activations_per_hour(report: dict[str, Any]) -> float:
    background_seconds = float(report.get("aggregate", {}).get("background_duration_seconds", 0.0))
    if background_seconds <= 0:
        return float("inf")
    false_activations = 0
    for entry in report.get("results", []):
        if entry.get("category") in {"negative_tv_conversation", "negative_distance_noise"}:
            activations = entry.get("activation_samples") or []
            false_activations += len(activations)
    hours = background_seconds / 3600.0
    return false_activations / hours if hours > 0 else float("inf")


def compare_with_baseline(
    eval_report: dict[str, Any],
    baseline: dict[str, Any],
    *,
    train_metrics: dict[str, Any],
) -> dict[str, Any]:
    recall_by_category = _category_recall(eval_report)
    overall_passed = int(eval_report.get("summary", {}).get("passed", 0))
    overall_total = int(eval_report.get("summary", {}).get("total", 0))
    overall_recall = overall_passed / overall_total if overall_total else 0.0
    fpph = _false_activations_per_hour(eval_report)
    deployed = baseline.get("deployed_threshold", {})
    baseline_recall = float(deployed.get("recall", 0.0))
    baseline_fpph = float(deployed.get("fpph", float("inf")))
    finite_metrics = bool(np.isfinite(fpph) and np.isfinite(overall_recall))
    return {
        "overall_recall": float(overall_recall),
        "recall_by_category": recall_by_category,
        "false_activations_per_hour": float(fpph) if np.isfinite(fpph) else None,
        "baseline_recall": baseline_recall,
        "baseline_fpph": baseline_fpph,
        "recall_regression": bool(overall_recall + 1e-9 < baseline_recall),
        "fp_regression": bool(fpph > min(baseline_fpph, FA_PER_HOUR_BOUND) + 1e-9),
        "finite_metrics": finite_metrics,
        "train_metrics": train_metrics,
    }


def check_voice_timing(
    eval_report: dict[str, Any],
    eval_root: Path,
    *,
    cases_path: Path | None = None,
) -> dict[str, Any]:
    cases_file = cases_path or (eval_root / "cases.json")
    if not cases_file.is_file():
        return {"status": "skipped", "reason": "no cases manifest"}
    cases = {entry["id"]: entry for entry in load_json(cases_file).get("cases", [])}
    late_cases: list[str] = []
    checked = 0
    for entry in eval_report.get("results", []):
        case_id = entry.get("case_id")
        case = cases.get(str(case_id), {})
        speech_end_ms = case.get("speech_end_ms")
        detection_sample = entry.get("detection_sample")
        if speech_end_ms is None or detection_sample is None:
            continue
        checked += 1
        detection_ms = float(detection_sample) / SAMPLE_RATE * 1000.0
        if detection_ms > float(speech_end_ms) + 50.0:
            late_cases.append(str(case_id))
    return {
        "status": "failed" if late_cases else "passed",
        "checked": checked,
        "late_cases": late_cases,
    }


def classify_candidate(
    *,
    comparison: dict[str, Any],
    timing: dict[str, Any],
    feasibility: dict[str, Any],
    eval_report: dict[str, Any],
    interrupted: bool,
    export_failed: bool,
    teacher_failed: bool,
    empty_eval: bool,
) -> str:
    if interrupted or export_failed or teacher_failed:
        return "rejected"
    if empty_eval:
        return "rejected"
    if not comparison.get("finite_metrics"):
        return "rejected"
    if comparison.get("recall_regression") or comparison.get("fp_regression"):
        return "rejected"
    if timing.get("status") == "failed":
        return "rejected"
    summary = eval_report.get("summary", {})
    if int(summary.get("failed", 0)) > 0 or int(summary.get("errors", 0)) > 0:
        return "rejected"
    if not feasibility.get("ok"):
        return "insufficient_evidence"
    background_hours = float(eval_report.get("aggregate", {}).get("background_duration_seconds", 0.0)) / 3600.0
    fpph = float(comparison.get("false_activations_per_hour", float("inf")))
    if background_hours < MIN_BACKGROUND_HOURS_FOR_QUAL or fpph > FA_PER_HOUR_BOUND:
        return "insufficient_evidence"
    if float(comparison.get("overall_recall", 0.0)) <= float(comparison.get("baseline_recall", 0.0)):
        return "evaluated"
    return "qualified"


def save_candidate_bundle(
    run_dir: Path,
    *,
    run_key: str,
    snapshot: Snapshot,
    train_output: TrainOutput,
    eval_report: dict[str, Any],
    comparison: dict[str, Any],
    timing: dict[str, Any],
    feasibility: dict[str, Any],
    status: str,
    teacher_audit: dict[str, Any],
) -> Path:
    bundle = run_dir / "candidate"
    bundle.mkdir(parents=True, exist_ok=True)
    shutil.copy2(train_output.model_path, bundle / "sayso.onnx")
    candidate = {
        "run_key": run_key,
        "status": status,
        "created_utc": _utc_now(),
        "artifact": "sayso.onnx",
        "threshold": train_output.threshold,
        "provider": train_output.provider,
        "hashes": {
            "model_sha256": _sha256_file(bundle / "sayso.onnx"),
            **snapshot.hashes,
        },
        "dataset_versions": {
            "snapshot_id": snapshot.snapshot_id,
            "class_counts": snapshot.counts,
        },
        "recipe_version": str(snapshot.recipe_path.name),
        "teacher_policy": {
            "teacher_version": teacher_audit.get("teacher_version"),
            "policy_version": teacher_audit.get("policy_version"),
        },
        "comparison_report": comparison,
        "timing_report": timing,
        "feasibility": feasibility,
        "eval_summary": eval_report.get("summary", {}),
    }
    (bundle / "candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (bundle / "eval_report.json").write_text(json.dumps(eval_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (bundle / "comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle


def _make_fixture_eval(tmp_eval: Path, *, late_detection: bool = False) -> None:
    audio_dir = tmp_eval / "audio"
    audio_dir.mkdir(parents=True)
    positive = np.zeros(WINDOW_SAMPLES + HOP_SAMPLES + 512, dtype="<i2")
    positive[WINDOW_SAMPLES:] = 9000
    write_synthetic_wav(audio_dir / "positive.wav", positive)
    background = np.zeros(SAMPLE_RATE * 2, dtype="<i2")
    write_synthetic_wav(audio_dir / "background.wav", background)
    speech_end_ms = 100.0 if late_detection else 5000.0
    cases = {
        "version": 1,
        "cases": [
            {
                "id": "pos",
                "category": "positive_sayso",
                "audio": "audio/positive.wav",
                "expect_detection": True,
                "speech_end_ms": speech_end_ms,
            },
            {
                "id": "background",
                "category": "negative_tv_conversation",
                "audio": "audio/background.wav",
                "expect_detection": False,
            },
        ],
    }
    (tmp_eval / "cases.json").write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")


def run_pipeline(
    *,
    spool: Path,
    seed_dir: Path | None,
    work_dir: Path,
    recipe_path: Path,
    eval_root: Path,
    baseline_path: Path,
    splits_path: Path,
    seed: int = 42,
    stub_mode: bool = True,
    teacher: Teacher | None = None,
    trainer: Trainer | None = None,
    require_known_real: str | None = None,
    eval_root_override: Path | None = None,
    omit_feature_for: str | None = None,
    interrupt_after: str | None = None,
    force_retrain: bool = False,
    eval_runner: Callable[[Path, Path, float], dict[str, Any]] | None = None,
    corpus_root: Path | None = None,
    corpus_holdouts: Sequence[str] | None = None,
    replace_living2: bool = False,
) -> PipelineResult:
    work_dir.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(work_dir)
    interrupted = False
    teacher_failed = False
    export_failed = False
    try:
        baseline = load_baseline(baseline_path)
        splits = load_splits(splits_path)
        feasibility = feasibility_gate(
            baseline=baseline,
            seed_dir=seed_dir,
            eval_root=eval_root,
        )
        teacher_impl = _resolve_teacher(stub_mode, teacher)
        trainer_impl = _resolve_trainer(stub_mode, trainer)
        rows = ingest_sources(
            spool,
            None if replace_living2 else seed_dir,
            corpus_root=corpus_root,
            corpus_seed=seed,
            corpus_holdouts=corpus_holdouts,
        )
        try:
            labeled, teacher_audit = label_records(rows, teacher_impl)
        except RuntimeError:
            teacher_failed = True
            labeled, teacher_audit = [], {"teacher_version": teacher_impl.version, "policy_version": LABEL_POLICY_VERSION}
        if teacher_failed:
            run_key = _sha256_json({"teacher_failed": True, "spool": str(spool)})
            run_dir = _run_dir(work_dir, run_key)
            _write_run_state(
                run_dir,
                {"run_key": run_key, "status": "rejected", "reason": "teacher failure", "completed_utc": _utc_now()},
            )
            return PipelineResult(run_key=run_key, status="rejected", bundle_dir=None, reason="teacher failure")

        selected = select_examples(
            labeled,
            splits,
            require_known_real=require_known_real,
            corpus_root=corpus_root,
        )
        deps_path = _REPO_ROOT / "satellite" / "models" / "requirements-wake-train.txt"
        deps_hash = deps_hash_from_requirements(deps_path)
        run_key, snapshot_id, _clips_hash = preview_run_key(
            selected,
            recipe_path=recipe_path,
            teacher_audit=teacher_audit,
            stub_mode=stub_mode,
            seed=seed,
            deps_hash=deps_hash,
        )
        run_dir = _run_dir(work_dir, run_key)
        prior = _load_run_state(run_dir)
        if prior and prior.get("status") in FINAL_STATUSES and not force_retrain:
            return PipelineResult(
                run_key=run_key,
                status=str(prior["status"]),
                bundle_dir=run_dir / "candidate" if (run_dir / "candidate" / "candidate.json").is_file() else None,
                noop=True,
                reason="identical completed run",
            )
        snapshot = build_snapshot(
            selected,
            work_dir=work_dir,
            recipe_path=recipe_path,
            seed=seed,
            teacher_audit=teacher_audit,
            snapshot_id=snapshot_id,
        )
        if omit_feature_for:
            feature_path = snapshot.root / "clips" / f"{snapshot.mapping[omit_feature_for]}_r0.wav"
            if feature_path.is_file():
                feature_path.unlink()
        try:
            assert_source_feature_counts(snapshot)
        except RuntimeError as exc:
            _write_run_state(run_dir, {"run_key": run_key, "status": "rejected", "reason": str(exc)})
            return PipelineResult(run_key=run_key, status="rejected", bundle_dir=None, reason=str(exc))
        _write_run_state(
            run_dir,
            {"run_key": run_key, "status": "in_progress", "started_utc": _utc_now(), "snapshot_id": snapshot.snapshot_id},
        )
        if interrupt_after == "snapshot":
            interrupted = True
            _write_run_state(run_dir, {"run_key": run_key, "status": "interrupted", "stage": "snapshot"})
            return PipelineResult(run_key=run_key, status="rejected", bundle_dir=None, reason="interrupted")

        train_dir = run_dir / "train"
        try:
            train_output = trainer_impl.train_and_export(snapshot, train_dir)
        except RuntimeError:
            export_failed = True
            _write_run_state(run_dir, {"run_key": run_key, "status": "rejected", "reason": "export failed"})
            return PipelineResult(run_key=run_key, status="rejected", bundle_dir=None, reason="export failed")

        if interrupt_after == "train":
            interrupted = True
            _write_run_state(run_dir, {"run_key": run_key, "status": "interrupted", "stage": "train"})
            return PipelineResult(run_key=run_key, status="rejected", bundle_dir=None, reason="interrupted")

        eval_target = eval_root_override or eval_root
        if stub_mode and eval_root_override is None:
            eval_target = work_dir / "_stub_eval"
            _make_fixture_eval(eval_target, late_detection=False)
        if not eval_target.joinpath("cases.json").is_file():
            eval_target = work_dir / "_stub_eval"
            _make_fixture_eval(eval_target, late_detection=False)

        if eval_runner is not None:
            eval_report = eval_runner(train_output.model_path, eval_target, train_output.threshold)
        else:
            eval_report = run_wake_eval(
                model_path=train_output.model_path,
                eval_root=eval_target,
                threshold=train_output.threshold,
                refractory_seconds=production_refractory_seconds(),
                strict=True,
            )
        empty_eval = int(eval_report.get("summary", {}).get("total", 0)) == 0
        comparison = compare_with_baseline(eval_report, baseline, train_metrics=train_output.metrics)
        timing = check_voice_timing(eval_report, eval_target)
        status = classify_candidate(
            comparison=comparison,
            timing=timing,
            feasibility=feasibility,
            eval_report=eval_report,
            interrupted=interrupted,
            export_failed=export_failed,
            teacher_failed=teacher_failed,
            empty_eval=empty_eval,
        )
        bundle = save_candidate_bundle(
            run_dir,
            run_key=run_key,
            snapshot=snapshot,
            train_output=train_output,
            eval_report=eval_report,
            comparison=comparison,
            timing=timing,
            feasibility=feasibility,
            status=status,
            teacher_audit=teacher_audit,
        )
        _write_run_state(
            run_dir,
            {
                "run_key": run_key,
                "status": status,
                "completed_utc": _utc_now(),
                "snapshot_id": snapshot.snapshot_id,
                "bundle": str(bundle),
            },
        )
        return PipelineResult(run_key=run_key, status=status, bundle_dir=bundle, comparison=comparison)
    finally:
        release_lock(lock)


def assert_source_feature_counts(snapshot: Snapshot) -> None:
    clips = snapshot.root / "clips"
    wavs = sorted(clips.glob("clip_*.wav"))
    sources = [path for path in wavs if "_r" not in path.stem]
    features = [path for path in wavs if "_r" in path.stem]
    if len(sources) != len(features):
        raise RuntimeError(
            f"source-to-feature count mismatch: {len(sources)} sources vs {len(features)} features"
        )
    if not snapshot.counts.get("positive") or not snapshot.counts.get("negative"):
        raise RuntimeError("snapshot must contain positive and negative classes")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spool", type=Path, required=True, help="Wake mining spool directory")
    parser.add_argument("--corpus", type=Path, default=None, help="Wake corpus root with sessions/events")
    parser.add_argument(
        "--corpus-holdout",
        action="append",
        default=[],
        metavar="SESSION_ID",
        help="Session IDs kept entirely outside training for continuous eval",
    )
    parser.add_argument(
        "--replace-living2",
        action="store_true",
        help="Omit trusted seed/living2 clips; train only from spool/corpus inputs",
    )
    parser.add_argument("--seed-dir", type=Path, default=None, help="Trusted seed positive/negative wav dirs")
    parser.add_argument("--work-dir", type=Path, required=True, help="Local training workspace")
    parser.add_argument("--recipe", type=Path, default=default_recipe_path())
    parser.add_argument("--eval-root", type=Path, default=repo_eval_root())
    parser.add_argument("--baseline", type=Path, default=default_baseline_path())
    parser.add_argument("--splits", type=Path, default=default_splits_path())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stub", action="store_true", help="Use stub teacher/trainer (default in CI)")
    parser.add_argument("--force", action="store_true", help="Ignore completed run-key no-op")
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Print cron snippet; refuses while feasibility gate is failed",
    )
    return parser


def schedule_refused_message(*, seed_dir: Path | None, baseline_path: Path, eval_root: Path) -> str | None:
    baseline = load_baseline(baseline_path)
    feasibility = feasibility_gate(baseline=baseline, seed_dir=seed_dir, eval_root=eval_root)
    if feasibility["ok"]:
        return None
    return "; ".join(feasibility["blockers"])


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.schedule:
        reason = schedule_refused_message(
            seed_dir=args.seed_dir,
            baseline_path=args.baseline,
            eval_root=args.eval_root,
        )
        if reason:
            print(f"scheduling refused: {reason}", file=sys.stderr)
            print("# cron disabled until trusted seed and eval audio exist", file=sys.stderr)
            return 2
        print("0 3 * * 0 cd /path/to/repo && python3 scripts/wake_train.py --spool ... --work-dir ...")
        return 0

    stub_mode = args.stub or os.environ.get("WAKE_TRAIN_STUB", "1") != "0"
    result = run_pipeline(
        spool=args.spool,
        seed_dir=args.seed_dir,
        work_dir=args.work_dir,
        recipe_path=args.recipe,
        eval_root=args.eval_root,
        baseline_path=args.baseline,
        splits_path=args.splits,
        seed=args.seed,
        stub_mode=stub_mode,
        force_retrain=args.force,
        corpus_root=args.corpus,
        corpus_holdouts=args.corpus_holdout or None,
        replace_living2=args.replace_living2,
    )
    if result.noop:
        print(f"no-op: {result.reason} (status={result.status}, run_key={result.run_key})")
        return 0
    print(f"status={result.status} run_key={result.run_key}")
    if result.bundle_dir:
        print(f"bundle={result.bundle_dir}")
    if result.status in {"rejected", "insufficient_evidence"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
