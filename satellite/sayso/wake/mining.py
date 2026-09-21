"""Opt-in wake-window capture for offline labelling and hard-negative mining.

Mining is off unless ``wake_word.mine_dir`` is set. Scored 16 kHz mono windows,
bounded ring context, and HA/STT outcomes are published as immutable record
directories via a bounded writer queue so capture and inference never block on
disk. A host ingest step verifies hashes and writes acks; the satellite deletes
only acknowledged records.
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import random
import shutil
import threading
import time
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from .capture import WakeCaptureRing

_LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WINDOW_SAMPLES = SAMPLE_RATE * 2
HOP_SAMPLES = 2560

# ponytail: 2000 two-second clips ~= 128 MB scored audio; raise if mining below 0.05.
DEFAULT_MAX_RECORDS = 2000
DEFAULT_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_QUEUE_SIZE = 32
DEFAULT_PRE_CONTEXT_MS = 500
DEFAULT_POST_CONTEXT_MS = 500
DEFAULT_POST_DEADLINE_MS = 1000
CLUSTER_SETTLE_S = 0.35
DEFAULT_BELOW_SAMPLE_RATE = 0.002

SAMPLING_DETECTION = "detection"
SAMPLING_NEAR = "near_threshold"
SAMPLING_BELOW = "below_threshold"

OUTCOME_WAKE = "wake"
OUTCOME_STT = "stt"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _safe_json_write(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


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


@dataclass
class LossStats:
    queue_full: int = 0
    spool_full: int = 0
    class_cap: int = 0
    write_failed: int = 0
    post_context_timeout: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "queue_full": self.queue_full,
            "spool_full": self.spool_full,
            "class_cap": self.class_cap,
            "write_failed": self.write_failed,
            "post_context_timeout": self.post_context_timeout,
        }


@dataclass
class _Cluster:
    sample_start: int
    sample_end: int
    score: float
    window: np.ndarray
    sample_index: int
    sampling_reason: str
    capture_id: str
    pre_context_index: int
    deadline: float = field(default_factory=lambda: time.monotonic() + CLUSTER_SETTLE_S)


@dataclass
class _WriteJob:
    capture_id: str
    sampling_reason: str
    score: float
    window: np.ndarray
    sample_index: int
    sample_start: int
    sample_end: int
    pre_pcm: bytes = b""
    pre_flags: dict[str, bool] = field(default_factory=dict)
    enqueue_mono: float = field(default_factory=time.monotonic)
    post_deadline_mono: float = 0.0


class HardNegativeMiner:
    """Publish scored wake windows and linked outcomes without blocking inference."""

    def __init__(
        self,
        spool_dir: Path,
        *,
        mine_threshold: float,
        detect_threshold: float,
        sample_rate: int = SAMPLE_RATE,
        window_samples: int = WINDOW_SAMPLES,
        hop_samples: int = HOP_SAMPLES,
        provider: str = "livekit",
        model_path: Optional[Path] = None,
        session_id: Optional[str] = None,
        processing_settings: Optional[dict[str, Any]] = None,
        max_records: int = DEFAULT_MAX_RECORDS,
        max_bytes: int = DEFAULT_MAX_BYTES,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        pre_context_ms: int = DEFAULT_PRE_CONTEXT_MS,
        post_context_ms: int = DEFAULT_POST_CONTEXT_MS,
        post_deadline_ms: int = DEFAULT_POST_DEADLINE_MS,
        detection_cap: int = 800,
        near_cap: int = 1000,
        below_cap: int = 200,
        below_sample_rate: float = DEFAULT_BELOW_SAMPLE_RATE,
        rng: Optional[random.Random] = None,
    ) -> None:
        self._root = Path(spool_dir)
        self._records = self._root / "records"
        self._staging = self._root / ".staging"
        self._outcomes = self._root / "outcomes"
        self._acks = self._root / "acks"
        self._mine_threshold = float(mine_threshold)
        self._detect_threshold = float(detect_threshold)
        self._sample_rate = int(sample_rate)
        self._window_samples = int(window_samples)
        self._hop_samples = int(hop_samples)
        self._provider = str(provider)
        self._model_path = Path(model_path) if model_path else None
        self._model_sha256 = (
            _sha256_file(self._model_path) if self._model_path and self._model_path.is_file() else None
        )
        self._session_id = session_id or str(uuid.uuid4())
        self._processing = dict(processing_settings or {})
        self._max_records = int(max_records)
        self._max_bytes = int(max_bytes)
        self._queue_size = max(1, int(queue_size))
        self._pre_context_samples = max(0, pre_context_ms * sample_rate // 1000)
        self._post_context_samples = max(0, post_context_ms * sample_rate // 1000)
        self._post_deadline_s = max(0.0, post_deadline_ms / 1000.0)
        self._class_caps = {
            SAMPLING_DETECTION: int(detection_cap),
            SAMPLING_NEAR: int(near_cap),
            SAMPLING_BELOW: int(below_cap),
        }
        self._class_counts = {key: 0 for key in self._class_caps}
        self._below_rate = float(below_sample_rate)
        self._rng = rng or random.Random()

        self._ring: Optional[Any] = None
        self._ring_reader: Optional[Callable[[int, int], bytes]] = None
        self._clusters: list[_Cluster] = []
        self._cluster_lock = threading.Lock()
        self._pending_pre: dict[str, tuple[bytes, dict[str, bool]]] = {}
        self._latest_detection_id: Optional[str] = None
        self._losses = LossStats()
        self._spool_full = False
        self._spool_full_logged = False
        self._published_records = 0
        self._published_bytes = 0

        self._queue: queue.Queue[_WriteJob] = queue.Queue(maxsize=self._queue_size)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._inflight = 0
        self._flush_timer: Optional[threading.Timer] = None

    @property
    def mine_threshold(self) -> float:
        return self._mine_threshold

    @property
    def losses(self) -> LossStats:
        return self._losses

    @property
    def latest_detection_capture_id(self) -> Optional[str]:
        return self._latest_detection_id

    @property
    def published_record_count(self) -> int:
        return self._published_records

    def bind_ring(self, ring: Any) -> None:
        self._ring = ring
        self._ring_reader = ring.read

    def start(self) -> None:
        if self._thread is not None:
            return
        if not self._ensure_dirs():
            return
        self._reconcile_spool_state()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="sayso-wake-miner", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        self._flush_clusters(force=True)
        self._stop.set()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._queue.empty() and self._inflight == 0:
                break
            time.sleep(0.01)
        self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
        self._thread = None

    def offer(
        self,
        score: float,
        window: np.ndarray,
        *,
        sample_index: int | None = None,
    ) -> Optional[str]:
        """Queue a scored window for publication. Never raises."""
        if window.size == 0:
            return None
        if not self._ensure_dirs():
            return None
        if self._spool_full:
            self._losses.spool_full += 1
            return None

        reason = self._classify(score)
        if reason is None:
            return None
        if self._class_counts[reason] >= self._class_caps[reason]:
            self._losses.class_cap += 1
            return None

        idx = int(sample_index if sample_index is not None else 0)
        sample_end = idx + 1
        sample_start = sample_end - int(window.size)
        capture_id = str(uuid.uuid4())

        cluster = _Cluster(
            sample_start=sample_start,
            sample_end=sample_end,
            score=float(score),
            window=np.asarray(window, dtype="<i2", copy=True),
            sample_index=idx,
            sampling_reason=reason,
            capture_id=capture_id,
            pre_context_index=idx,
        )
        with self._cluster_lock:
            merged = self._merge_cluster(cluster)
            if merged is None:
                self._clusters.append(cluster)
                if score >= self._detect_threshold:
                    self._latest_detection_id = capture_id
                published_id = capture_id
            else:
                if merged.score >= self._detect_threshold:
                    self._latest_detection_id = merged.capture_id
                published_id = merged.capture_id
            self._schedule_cluster_flush()
        return published_id

    def snapshot_pre_trigger(self, trigger_index: int, *, synthetic_padding: bool = False) -> None:
        """Capture ring audio before rearm/overwrite for pending publications."""
        if self._ring_reader is None or self._pre_context_samples <= 0:
            return
        pre_start = trigger_index - self._window_samples - self._pre_context_samples + 1
        pre_end = trigger_index - self._window_samples + 1
        pcm = self._ring_reader(pre_start, pre_end)
        flags = {
            "pre_context_missing": len(pcm) < self._pre_context_samples * 2,
            "pre_synthetic_padding": bool(synthetic_padding),
        }
        key = f"pre:{trigger_index}"
        self._pending_pre[key] = (pcm, flags)

    def note_rearm(self) -> None:
        """Mark that the next pre-trigger snapshot may include synthetic padding."""
        self._processing["last_rearm_mono"] = time.monotonic()

    def publish_wake_outcome(
        self,
        capture_id: str,
        *,
        accepted: bool,
        suppressed: bool = False,
        reason: str = "",
        ha_run_id: str | None = None,
    ) -> None:
        if not capture_id or not self._ensure_dirs():
            return
        payload = {
            "capture_id": capture_id,
            "kind": OUTCOME_WAKE,
            "accepted": bool(accepted),
            "suppressed": bool(suppressed),
            "reason": reason,
            "ha_run_id": ha_run_id,
            "published_utc": _utc_stamp(),
        }
        self._publish_outcome(capture_id, OUTCOME_WAKE, payload)

    def publish_stt_outcome(
        self,
        capture_id: str,
        *,
        stt_run_id: str,
        transcript: str = "",
        underflow: bool = False,
        failed_reason: str | None = None,
        ha_run_id: str | None = None,
    ) -> None:
        if not capture_id or not self._ensure_dirs():
            return
        payload = {
            "capture_id": capture_id,
            "kind": OUTCOME_STT,
            "stt_run_id": stt_run_id,
            "transcript": transcript,
            "underflow": bool(underflow),
            "failed_reason": failed_reason,
            "ha_run_id": ha_run_id,
            "published_utc": _utc_stamp(),
        }
        self._publish_outcome(capture_id, OUTCOME_STT, payload)

    def drain_acks(self, *, max_age_days: float = 14.0) -> int:
        """Delete only host-acknowledged records. Returns deleted count."""
        if not self._acks.is_dir():
            return 0
        removed = 0
        for ack_path in self._acks.glob("*.json"):
            try:
                meta = json.loads(ack_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            capture_id = str(meta.get("capture_id") or ack_path.stem)
            record_dir = self._records / capture_id
            if record_dir.is_dir():
                shutil.rmtree(record_dir, ignore_errors=True)
                removed += 1
            for outcome in self._outcomes.glob(f"{capture_id}.*.json"):
                outcome.unlink(missing_ok=True)
            if max_age_days > 0:
                try:
                    if time.time() - ack_path.stat().st_mtime > max_age_days * 86400.0:
                        ack_path.unlink(missing_ok=True)
                except OSError:
                    pass
        if removed:
            self._reconcile_spool_state()
        return removed

    def _classify(self, score: float) -> Optional[str]:
        if score >= self._detect_threshold:
            return SAMPLING_DETECTION
        if score >= self._mine_threshold:
            return SAMPLING_NEAR
        if self._rng.random() < self._below_rate:
            return SAMPLING_BELOW
        return None

    def _merge_cluster(self, cluster: _Cluster) -> Optional[_Cluster]:
        for existing in self._clusters:
            if self._ranges_overlap(
                existing.sample_start,
                existing.sample_end,
                cluster.sample_start,
                cluster.sample_end,
            ):
                if cluster.score >= existing.score:
                    existing.score = cluster.score
                    existing.window = cluster.window
                    existing.sample_index = cluster.sample_index
                    existing.sample_start = cluster.sample_start
                    existing.sample_end = cluster.sample_end
                    existing.sampling_reason = cluster.sampling_reason
                existing.deadline = time.monotonic() + CLUSTER_SETTLE_S
                return existing
        return None

    @staticmethod
    def _ranges_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
        return a0 < b1 and b0 < a1

    def _schedule_cluster_flush(self) -> None:
        if self._flush_timer is not None:
            self._flush_timer.cancel()
        self._flush_timer = threading.Timer(CLUSTER_SETTLE_S, self._flush_clusters)
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def _flush_clusters(self, force: bool = False) -> None:
        with self._cluster_lock:
            now = time.monotonic()
            ready = [c for c in self._clusters if force or c.deadline <= now]
            self._clusters = [c for c in self._clusters if c not in ready]
        for cluster in ready:
            self._enqueue_cluster(cluster)

    def _enqueue_cluster(self, cluster: _Cluster) -> None:
        if self._spool_full:
            self._losses.spool_full += 1
            return
        pre_pcm, pre_flags = self._pending_pre.pop(
            f"pre:{cluster.pre_context_index}", (b"", {})
        )
        if not pre_pcm and self._ring_reader is not None and self._pre_context_samples > 0:
            pre_start = (
                cluster.sample_index - self._window_samples - self._pre_context_samples + 1
            )
            pre_end = cluster.sample_index - self._window_samples + 1
            pre_pcm = self._ring_reader(pre_start, pre_end)
            pre_flags = {
                "pre_context_missing": len(pre_pcm) < self._pre_context_samples * 2,
                "pre_synthetic_padding": False,
            }
        job = _WriteJob(
            capture_id=cluster.capture_id,
            sampling_reason=cluster.sampling_reason,
            score=cluster.score,
            window=cluster.window,
            sample_index=cluster.sample_index,
            sample_start=cluster.sample_start,
            sample_end=cluster.sample_end,
            pre_pcm=pre_pcm,
            pre_flags=pre_flags,
            post_deadline_mono=time.monotonic() + self._post_deadline_s,
        )
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            self._losses.queue_full += 1

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                job = self._queue.get(timeout=0.1)
            except queue.Empty:
                self.drain_acks()
                continue
            try:
                self._inflight += 1
                self._publish_job(job)
            except Exception:
                self._losses.write_failed += 1
                _LOGGER.exception("Failed to publish wake mining record %s", job.capture_id)
            finally:
                self._inflight -= 1

    def _publish_job(self, job: _WriteJob) -> None:
        post_pcm, post_flags = self._collect_post_context(job)
        staging = self._staging / job.capture_id
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)

        window_path = staging / "window.wav"
        _write_wav(window_path, job.window, self._sample_rate)
        hashes: dict[str, str] = {"window_sha256": _sha256_file(window_path)}

        pre_path = staging / "pre.wav"
        post_path = staging / "post.wav"
        quality_flags = {
            "pre_context_missing": bool(job.pre_flags.get("pre_context_missing")),
            "post_context_missing": bool(post_flags.get("post_context_missing")),
            "pre_synthetic_padding": bool(job.pre_flags.get("pre_synthetic_padding")),
            "post_synthetic_padding": bool(post_flags.get("post_synthetic_padding")),
        }
        if job.pre_pcm:
            pre_samples = np.frombuffer(job.pre_pcm, dtype="<i2")
            _write_wav(pre_path, pre_samples, self._sample_rate)
            hashes["pre_sha256"] = _sha256_file(pre_path)
        else:
            quality_flags["pre_context_missing"] = True
        if post_pcm:
            post_samples = np.frombuffer(post_pcm, dtype="<i2")
            _write_wav(post_path, post_samples, self._sample_rate)
            hashes["post_sha256"] = _sha256_file(post_path)
        else:
            quality_flags["post_context_missing"] = True

        record = {
            "capture_id": job.capture_id,
            "session_id": self._session_id,
            "provider": self._provider,
            "model_path": str(self._model_path) if self._model_path else None,
            "model_sha256": self._model_sha256,
            "score": round(job.score, 6),
            "detect_threshold": self._detect_threshold,
            "mine_threshold": self._mine_threshold,
            "fired": bool(job.score >= self._detect_threshold),
            "sampling_reason": job.sampling_reason,
            "sample_rate": self._sample_rate,
            "window_samples": int(job.window.size),
            "sample_start": job.sample_start,
            "sample_end": job.sample_end,
            "processing": self._processing,
            "quality_flags": quality_flags,
            "hashes": hashes,
            "label": None,
            "transcript": None,
            "notes": None,
            "published_utc": _utc_stamp(),
        }
        _safe_json_write(staging / "record.json", record)

        published = self._records / job.capture_id
        if published.exists():
            shutil.rmtree(published, ignore_errors=True)
        staging.rename(published)

        self._class_counts[job.sampling_reason] += 1
        self._published_records += 1
        self._published_bytes += sum(
            p.stat().st_size for p in published.iterdir() if p.is_file()
        )
        if self._published_records >= self._max_records or self._published_bytes >= self._max_bytes:
            self._spool_full = True
            if not self._spool_full_logged:
                _LOGGER.warning(
                    "Wake mining spool full (%d records, %d bytes in %s); counting drops.",
                    self._published_records,
                    self._published_bytes,
                    self._root,
                )
                self._spool_full_logged = True
        else:
            self._spool_full = False
            self._spool_full_logged = False

    def _collect_post_context(self, job: _WriteJob) -> tuple[bytes, dict[str, bool]]:
        flags = {"post_context_missing": True, "post_synthetic_padding": False}
        if self._ring_reader is None or self._post_context_samples <= 0:
            return b"", flags
        post_start = job.sample_end
        post_end = post_start + self._post_context_samples
        while time.monotonic() < job.post_deadline_mono:
            pcm = self._ring_reader(post_start, post_end)
            if len(pcm) >= self._post_context_samples * 2:
                flags["post_context_missing"] = False
                return pcm, flags
            if self._stop.is_set():
                break
            time.sleep(0.01)
        self._losses.post_context_timeout += 1
        pcm = self._ring_reader(post_start, post_end)
        flags["post_context_missing"] = len(pcm) < self._post_context_samples * 2
        return pcm, flags

    def _publish_outcome(self, capture_id: str, kind: str, payload: dict[str, Any]) -> None:
        path = self._outcomes / f"{capture_id}.{kind}.json"
        if path.is_file():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
            existing.update(payload)
            payload = existing
        staging = path.with_suffix(path.suffix + ".tmp")
        staging.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        staging.replace(path)

    def _ensure_dirs(self) -> bool:
        try:
            for path in (self._root, self._records, self._staging, self._outcomes, self._acks):
                path.mkdir(parents=True, exist_ok=True)
        except OSError:
            _LOGGER.exception("Cannot create wake mining dir %s; mining disabled", self._root)
            return False
        return True

    def _reconcile_spool_state(self) -> None:
        for partial in list(self._staging.iterdir()):
            if partial.is_dir():
                shutil.rmtree(partial, ignore_errors=True)
        self._published_records = 0
        self._published_bytes = 0
        self._class_counts = {key: 0 for key in self._class_caps}
        for record_dir in self._records.iterdir():
            if not record_dir.is_dir():
                continue
            meta_path = record_dir / "record.json"
            if not meta_path.is_file():
                shutil.rmtree(record_dir, ignore_errors=True)
                continue
            self._published_records += 1
            self._published_bytes += sum(p.stat().st_size for p in record_dir.iterdir() if p.is_file())
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                reason = meta.get("sampling_reason")
                if reason in self._class_counts:
                    self._class_counts[reason] += 1
            except json.JSONDecodeError:
                continue
        if self._published_records >= self._max_records or self._published_bytes >= self._max_bytes:
            self._spool_full = True
        else:
            self._spool_full = False
            self._spool_full_logged = False


def ingest_record(record_dir: Path) -> tuple[bool, str]:
    """Verify a published record directory. Returns (ok, message)."""
    meta_path = record_dir / "record.json"
    if not meta_path.is_file():
        return False, "missing record.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"malformed record.json: {exc}"
    hashes = meta.get("hashes") or {}
    window = record_dir / "window.wav"
    if not window.is_file():
        return False, "missing window.wav"
    window_hash = _sha256_file(window)
    if hashes.get("window_sha256") and hashes["window_sha256"] != window_hash:
        return False, "window hash mismatch"
    pre = record_dir / "pre.wav"
    if pre.is_file():
        pre_hash = _sha256_file(pre)
        if hashes.get("pre_sha256") and hashes["pre_sha256"] != pre_hash:
            return False, "pre hash mismatch"
    post = record_dir / "post.wav"
    if post.is_file():
        post_hash = _sha256_file(post)
        if hashes.get("post_sha256") and hashes["post_sha256"] != post_hash:
            return False, "post hash mismatch"
    return True, "ok"


def write_ack(spool_dir: Path, capture_id: str, *, host: str = "wake_mine_report") -> Path:
    ack_dir = Path(spool_dir) / "acks"
    ack_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "capture_id": capture_id,
        "verified_utc": _utc_stamp(),
        "host": host,
    }
    path = ack_dir / f"{capture_id}.json"
    _safe_json_write(path, payload)
    return path


def demo() -> None:
    """Self-check: sampling, hashes, queue loss, spool cap, and ingest/ack."""
    import tempfile

    rng = np.random.default_rng(0)
    window = (rng.standard_normal(32000) * 1000).astype("<i2")

    with tempfile.TemporaryDirectory() as tmp:
        spool = Path(tmp)
        miner = HardNegativeMiner(
            spool,
            mine_threshold=0.1,
            detect_threshold=0.5,
            model_path=Path("sayso.onnx"),
            below_sample_rate=0.0,
            rng=random.Random(0),
        )
        miner.start()

        assert miner.offer(0.05, window) is None, "below mine threshold must not write"
        assert not list((spool / "records").glob("*"))

        near_id = miner.offer(0.22, window, sample_index=31999)
        assert near_id is not None
        fired_id = miner.offer(0.87, window, sample_index=63999)
        assert fired_id is not None
        miner._flush_clusters(force=True)
        miner.stop(timeout=3.0)

        records = sorted((spool / "records").iterdir())
        assert records, "expected at least one published record"
        meta = json.loads((records[0] / "record.json").read_text())
        assert meta["label"] is None, "capture must not presume a label"
        assert meta["fired"] is False or meta["score"] >= 0.5
        assert meta["capture_id"]
        assert meta["session_id"]

        with wave.open(str(records[0] / "window.wav"), "rb") as wf:
            assert wf.getframerate() == 16000 and wf.getnchannels() == 1
            restored = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
        assert np.array_equal(restored, window), "mined clip must be bit-exact"

        ok, _ = ingest_record(records[0])
        assert ok
        write_ack(spool, meta["capture_id"])
        assert miner.drain_acks() == 1

        assert miner.offer(0.9, np.array([], dtype="<i2")) is None, "empty window"

    with tempfile.TemporaryDirectory() as tmp:
        capped = HardNegativeMiner(
            Path(tmp),
            mine_threshold=0.1,
            detect_threshold=0.5,
            max_records=2,
            below_sample_rate=0.0,
            rng=random.Random(0),
        )
        capped.start()
        assert capped.offer(0.5, window, sample_index=31999) is not None
        capped._flush_clusters(force=True)
        capped.stop(timeout=3.0)
        capped.start()
        assert capped.offer(0.5, window, sample_index=63999) is not None
        capped._flush_clusters(force=True)
        capped.stop(timeout=3.0)
        capped.start()
        third = capped.offer(0.5, window, sample_index=95999)
        capped._flush_clusters(force=True)
        capped.stop(timeout=3.0)
        assert third is None, "spool full must stop new offers"
        assert capped.losses.spool_full >= 1
        assert len(list((Path(tmp) / "records").iterdir())) == 2

    print("mining self-check ok")


if __name__ == "__main__":
    demo()
