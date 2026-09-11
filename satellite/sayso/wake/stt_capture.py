"""Retain the exact post-processing PCM handed to Home Assistant.

Milestone 0 of the audio-path remediation: before tuning gain, adjusting
reverberation, or blaming the model, there must be an artifact that can be
listened to. This records the bytes at the single point where they leave for
Home Assistant, so a WAV of a failed command is byte-identical to what Faster
Whisper received -- not a re-recording, not a re-derivation.

Writes happen on a dedicated thread with a single-slot queue. A slow disk must
never stall the capture loop: a command that cannot be queued is dropped and
counted, never awaited.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_LOGGER = logging.getLogger(__name__)

DEFAULT_CAPTURE_DIR = Path("/var/lib/sayso-satellite/stt_capture")
DEFAULT_MAX_COMMANDS = 500
DEFAULT_MAX_AGE_DAYS = 14


@dataclass
class _Command:
    run_id: str
    samples: bytearray = field(default_factory=bytearray)
    transcript: str = ""
    underflow: bool = False
    failed: bool = False
    reason: str = ""


class SttAudioRecorder:
    """Tap the STT path and persist one WAV per command, plus one per failure.

    Usage from the voice path::

        recorder.begin_command(run_id)
        recorder.tap(pcm)      # exactly the bytes passed to handle_audio
        recorder.end_command(run_id, transcript=text)

    ``tap`` is cheap and never touches disk. ``end_command`` hands the finished
    command to the writer thread and returns immediately.
    """

    def __init__(
        self,
        directory: Path | str | None = None,
        *,
        sample_rate: int = 16000,
        capture_rate: int = 44100,
        mic_gain_db: float = 0.0,
        noise_suppression: int = 0,
        auto_gain: int = 0,
        enabled: bool = True,
        max_pending: int = 8,
        max_commands: int = DEFAULT_MAX_COMMANDS,
        max_age_days: float = DEFAULT_MAX_AGE_DAYS,
    ) -> None:
        self._dir = Path(directory) if directory is not None else DEFAULT_CAPTURE_DIR
        self._commands_dir = self._dir / "commands"
        self._failures_dir = self._dir / "failures"
        self._sample_rate = int(sample_rate)
        self._capture_rate = int(capture_rate)
        self._mic_gain_db = float(mic_gain_db)
        self._noise_suppression = int(noise_suppression)
        self._auto_gain = int(auto_gain)
        self._enabled = bool(enabled)
        self._max_pending = max(1, int(max_pending))
        self._max_commands = int(max_commands)
        self._max_age_days = float(max_age_days)

        self._queue: queue.Queue[_Command] = queue.Queue(maxsize=self._max_pending)
        self._lock = threading.Lock()
        self._active: Optional[_Command] = None
        self._dropped = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._inflight = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def directory(self) -> Path:
        return self._dir

    def start(self) -> None:
        if not self._enabled or self._thread is not None:
            return
        self._commands_dir.mkdir(parents=True, exist_ok=True)
        self._failures_dir.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="sayso-stt-capture", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=timeout)
        self._thread = None

    def flush(self, timeout: float = 2.0) -> None:
        """Wait until the writer queue drains (tests and orderly shutdown)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._queue.empty() and self._inflight == 0:
                return
            time.sleep(0.005)

    def begin_command(self, run_id: str) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._active = _Command(run_id=str(run_id))

    def tap(self, pcm_s16le: bytes) -> None:
        """Record bytes on their way to Home Assistant. Never blocks on IO."""
        if not self._enabled or not pcm_s16le:
            return
        with self._lock:
            active = self._active
            if active is None:
                return
            active.samples.extend(pcm_s16le)

    def mark_failed(self, reason: str) -> None:
        """Flag the active command so a failure WAV is written as well."""
        with self._lock:
            if self._active is not None:
                self._active.failed = True
                self._active.reason = str(reason)

    def end_command(
        self,
        run_id: str,
        *,
        transcript: str = "",
        underflow: bool = False,
        failed_reason: str | None = None,
    ) -> Optional[Path]:
        """Finish the active command and queue it for writing.

        Returns the command WAV path immediately; the file may still be
        draining on the writer thread. ``None`` when disabled or empty.
        """
        if not self._enabled:
            return None
        with self._lock:
            active = self._active
            self._active = None
        if active is None:
            return None
        active.transcript = transcript or ""
        active.underflow = bool(underflow)
        if failed_reason or not active.transcript.strip():
            active.failed = True
            active.reason = failed_reason or "empty_transcript"
        if not active.samples:
            return None
        self._enqueue(active)
        return self._commands_dir / f"{_safe(active.run_id)}.wav"

    def _enqueue(self, command: _Command) -> None:
        try:
            self._queue.put_nowait(command)
        except queue.Full:
            # Drop the oldest pending command rather than stalling the audio
            # thread; the newest failure is the one worth keeping.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(command)
            except queue.Full:
                self._dropped += 1
                _LOGGER.warning(
                    "STT capture queue full; dropped command %s", command.run_id
                )

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                command = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._inflight += 1
                self._write(command)
            except Exception:
                _LOGGER.exception("Failed to write STT capture for %s", command.run_id)
            finally:
                self._inflight -= 1
                self._prune()

    def _write(self, command: _Command) -> None:
        data = np.frombuffer(bytes(command.samples), dtype="<i2")
        name = _safe(command.run_id)
        command_path = self._commands_dir / f"{name}.wav"
        _write_wav(command_path, data, self._sample_rate)
        meta = self._sidecar(command, data)
        _write_sidecar(command_path.with_suffix(".json"), meta)
        if command.failed:
            failure_path = self._failures_dir / f"{name}_{_safe(command.reason)}.wav"
            _write_wav(failure_path, data, self._sample_rate)
            _write_sidecar(failure_path.with_suffix(".json"), meta)

    def _sidecar(self, command: _Command, data: np.ndarray) -> dict:
        # The capture rate and processing chain are recorded so a listening
        # session can be tied back to the exact settings that produced it.
        peak = int(np.max(np.abs(data))) if data.size else 0
        rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2))) if data.size else 0.0
        db = 20.0 * np.log10(rms / 32768.0) if rms > 0 else -120.0
        clip_count = int(np.sum(np.abs(data) >= 32767))
        return {
            "run_id": command.run_id,
            "capture_rate": self._capture_rate,
            "sample_rate": self._sample_rate,
            "mic_gain_db": self._mic_gain_db,
            "noise_suppression": self._noise_suppression,
            "auto_gain": self._auto_gain,
            "samples": int(data.size),
            "duration_s": data.size / self._sample_rate if self._sample_rate else 0.0,
            "peak": peak,
            "rms": rms,
            "rms_dbfs": db,
            "clip_count": clip_count,
            "transcript": command.transcript,
            "underflow": command.underflow,
            "failed": command.failed,
            "reason": command.reason,
        }

    def _prune(self) -> None:
        wavs = sorted(
            self._commands_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime
        )
        excess = len(wavs) - self._max_commands
        for path in wavs[: max(0, excess)]:
            _unlink_with_sidecar(path)
        if self._max_age_days <= 0:
            return
        cutoff = time.time() - self._max_age_days * 86400.0
        for path in list(self._commands_dir.glob("*.wav")) + list(
            self._failures_dir.glob("*.wav")
        ):
            try:
                if path.stat().st_mtime < cutoff:
                    _unlink_with_sidecar(path)
            except FileNotFoundError:
                continue


def _safe(name: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))
    return keep or "run"


def _unlink_with_sidecar(path: Path) -> None:
    for candidate in (path, path.with_suffix(".json")):
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass


def _write_wav(path: Path, data: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".wav.tmp")
    with wave.open(str(tmp), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(np.ascontiguousarray(data).astype("<i2", copy=False).tobytes())
    tmp.replace(path)


def _write_sidecar(path: Path, meta: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(path)
