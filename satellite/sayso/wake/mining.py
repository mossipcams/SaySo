"""Capture high-scoring wake windows from production audio as hard negatives.

The satellite has never retained the audio behind a detection, so every
false-positive investigation has had to guess at what actually fired the model
from the STT transcript of whatever was said *afterwards*. This module closes
that loop: it writes the exact 2 s window the classifier scored, so a mined
clip is byte-identical to what inference saw.

Mining is off unless ``wake_word.mine_dir`` is set. It runs on the wake worker
thread, after predict, so a slow disk delays the next inference rather than the
capture loop.
"""

from __future__ import annotations

import json
import logging
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np

_LOGGER = logging.getLogger(__name__)

# ponytail: a 2 s int16 mono window is ~64 KB, so 2000 clips caps the spool near
# 128 MB. At the measured ~22 events/day above 0.10 that is a couple of months of
# headroom. Raise it if you mine lower than 0.05.
DEFAULT_MAX_CLIPS = 2000


class HardNegativeMiner:
    """Write scored wake windows to a spool directory for offline labelling.

    Every clip is unlabelled at capture time: a window over the mining threshold
    may be a genuine wake or a false positive, and only a human listening to it
    can say which. The sidecar records what the model thought; ``label`` stays
    null until review.
    """

    def __init__(
        self,
        spool_dir: Path,
        *,
        mine_threshold: float,
        detect_threshold: float,
        sample_rate: int = 16000,
        max_clips: int = DEFAULT_MAX_CLIPS,
        model_path: Optional[Path] = None,
    ) -> None:
        self._dir = Path(spool_dir)
        self._mine_threshold = float(mine_threshold)
        self._detect_threshold = float(detect_threshold)
        self._sample_rate = int(sample_rate)
        self._max_clips = int(max_clips)
        self._model = str(model_path) if model_path else None
        self._count: Optional[int] = None
        self._full_logged = False

    @property
    def mine_threshold(self) -> float:
        return self._mine_threshold

    def _ensure_dir(self) -> bool:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            _LOGGER.exception("Cannot create wake mining dir %s; mining disabled", self._dir)
            return False
        if self._count is None:
            self._count = len(list(self._dir.glob("*.wav")))
        return True

    def offer(self, score: float, window: np.ndarray) -> Optional[Path]:
        """Persist ``window`` when ``score`` clears the mining threshold.

        Returns the written WAV path, or None when skipped. Never raises: a
        mining failure must not take down wake detection.
        """
        if score < self._mine_threshold or window.size == 0:
            return None
        if not self._ensure_dir():
            return None
        assert self._count is not None
        if self._count >= self._max_clips:
            if not self._full_logged:
                _LOGGER.warning(
                    "Wake mining spool full (%d clips in %s); not capturing more. "
                    "Drain it with scripts/wake_mine_report.py, then delete the reviewed clips.",
                    self._max_clips,
                    self._dir,
                )
                self._full_logged = True
            return None

        # Wall-clock in the name so clips sort chronologically and collide only
        # within the same millisecond, which the hop rate cannot produce.
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        stem = f"{stamp}_{int(time.monotonic() * 1000) % 1000:03d}_s{score:.4f}"
        wav_path = self._dir / f"{stem}.wav"
        try:
            samples = np.asarray(window, dtype="<i2")
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._sample_rate)
                wf.writeframes(samples.tobytes())
            sidecar = {
                "score": round(float(score), 6),
                "detect_threshold": self._detect_threshold,
                "mine_threshold": self._mine_threshold,
                "fired": bool(score >= self._detect_threshold),
                "captured_utc": stamp,
                "sample_rate": self._sample_rate,
                "samples": int(samples.size),
                "model": self._model,
                # Filled in by review, not by the satellite. null = unreviewed.
                "label": None,
                "transcript": None,
                "notes": None,
            }
            wav_path.with_suffix(".json").write_text(
                json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8"
            )
        except (OSError, ValueError, wave.Error):
            _LOGGER.exception("Failed to write mined wake window %s", wav_path)
            return None

        self._count += 1
        return wav_path


def demo() -> None:
    """Self-check: threshold gating, sidecar contents, and the clip cap."""
    import tempfile

    rng = np.random.default_rng(0)
    window = (rng.standard_normal(32000) * 1000).astype("<i2")

    with tempfile.TemporaryDirectory() as tmp:
        spool = Path(tmp)
        miner = HardNegativeMiner(
            spool, mine_threshold=0.1, detect_threshold=0.5, model_path=Path("sayso.onnx")
        )

        assert miner.offer(0.05, window) is None, "below mine threshold must not write"
        assert not list(spool.glob("*.wav"))

        near_miss = miner.offer(0.22, window)
        assert near_miss is not None and near_miss.is_file()
        meta = json.loads(near_miss.with_suffix(".json").read_text())
        assert meta["fired"] is False, "0.22 is under the 0.5 detect threshold"
        assert meta["label"] is None, "capture must not presume a label"
        assert meta["score"] == 0.22
        assert meta["samples"] == 32000

        fired = miner.offer(0.87, window)
        assert fired is not None
        assert json.loads(fired.with_suffix(".json").read_text())["fired"] is True

        # Round-trip: the mined clip must reproduce the scored window exactly,
        # or mined negatives do not match what inference saw.
        with wave.open(str(near_miss), "rb") as wf:
            assert wf.getframerate() == 16000 and wf.getnchannels() == 1
            restored = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
        assert np.array_equal(restored, window), "mined clip must be bit-exact"

        assert miner.offer(0.9, np.array([], dtype="<i2")) is None, "empty window"

    with tempfile.TemporaryDirectory() as tmp:
        capped = HardNegativeMiner(
            Path(tmp), mine_threshold=0.1, detect_threshold=0.5, max_clips=2
        )
        assert capped.offer(0.5, window) is not None
        assert capped.offer(0.5, window) is not None
        assert capped.offer(0.5, window) is None, "cap must stop further writes"

    print("mining self-check ok")


if __name__ == "__main__":
    demo()
