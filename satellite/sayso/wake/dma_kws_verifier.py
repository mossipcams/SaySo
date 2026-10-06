from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

_LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAMES = 78  # 0.8 s of 10 ms fbank frames; the ONNX model's input shape is fixed
CROP = 12720  # samples that give FRAMES frames: 1 + (12720 - 400) // 160
POST_SAMPLES = 3200  # keep 0.2 s after the detected end of speech
ANCHOR_LEN = 4  # keyword phonemes; baked into the ONNX export

# DMA-KWS Stage II (155k-v2-ft) exported by scripts/export_dma_kws_onnx.py. The service
# loads only a pre-provisioned file whose SHA-256 matches this pin; it never downloads.
MODEL_SHA256 = "a2d65d14cced662a76d8bcc4723c0fc4090c8b9db4b8a94bf598270ea485c6c9"

# ARPAbet token ids 3.. from the DMA-KWS phoneme dictionary (0-2 are blank/unk/sos).
_PHONES = (
    "AA0 AA1 AA2 AE0 AE1 AE2 AH0 AH1 AH2 AO0 AO1 AO2 AW0 AW1 AW2 AY0 AY1 AY2 B CH D DH EH0 "
    "EH1 EH2 ER0 ER1 ER2 EY0 EY1 EY2 F G HH IH0 IH1 IH2 IY0 IY1 IY2 JH K L M N NG OW0 OW1 OW2 "
    "OY0 OY1 OY2 P R S SH T TH UH0 UH1 UH2 UW UW0 UW1 UW2 V W Y Z ZH"
).split()
PHONE_IDS = {p: i + 3 for i, p in enumerate(_PHONES)}


def _mel(freq: np.ndarray) -> np.ndarray:
    return 1127.0 * np.log(1.0 + freq / 700.0)


def _filterbank(bins: int = 80, nfft: int = 512, low: float = 20.0, high: float = 8000.0) -> np.ndarray:
    nb = nfft // 2
    mel = _mel(np.arange(nb) * SAMPLE_RATE / nfft)
    lo, hi = _mel(np.float64(low)), _mel(np.float64(high))
    step = (hi - lo) / (bins + 1)
    fb = np.zeros((bins, nb))
    for i in range(bins):
        left, centre, right = lo + i * step, lo + (i + 1) * step, lo + (i + 2) * step
        fb[i] = np.maximum(0.0, np.minimum((mel - left) / (centre - left), (right - mel) / (right - centre)))
    return np.pad(fb, ((0, 0), (0, 1)))  # Kaldi's extra zero column for the Nyquist bin


_FB = _filterbank()
_WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 399)) ** 0.85  # povey


def fbank(samples: np.ndarray) -> np.ndarray:
    """80-bin Kaldi log-mel filterbank, 25/10 ms, povey window, no dither (matches torchaudio)."""
    x = np.asarray(samples, dtype=np.float32) * 32768.0
    n = 1 + (len(x) - 400) // 160
    idx = np.arange(400)[None] + 160 * np.arange(n)[:, None]
    frames = x[idx].astype(np.float64)
    frames -= frames.mean(1, keepdims=True)
    previous = np.concatenate([frames[:, :1], frames[:, :-1]], 1)  # Kaldi replicate-pads the first sample
    frames = (frames - 0.97 * previous) * _WINDOW
    power = np.abs(np.fft.rfft(frames, 512)) ** 2
    return np.log(np.maximum(power @ _FB.T, np.finfo(np.float32).eps)).astype(np.float32)


def crop_after_speech(audio: np.ndarray) -> Optional[np.ndarray]:
    """The CROP samples ending 0.2 s after the last loud frame, or None for silence."""
    if len(audio) < 320:
        return None
    frames = audio[: len(audio) // 320 * 320].reshape(-1, 320)
    db = 20 * np.log10(np.sqrt((frames.astype(np.float64) ** 2).mean(1)) + 1e-9)
    if db.max() < -80:
        return None
    end = (np.where(db > db.max() - 25)[0].max() + 1) * 320
    padded = np.concatenate([audio, np.zeros(POST_SAMPLES, dtype=audio.dtype)])
    stop = min(len(padded), end + POST_SAMPLES)
    seg = padded[max(0, stop - CROP) : stop]
    return np.pad(seg, (CROP - len(seg), 0)) if len(seg) < CROP else seg


def _verified_model(path: Optional[Path]) -> Path:
    if path is None:
        raise ValueError("wake_word.dma_kws_model is required (no runtime model download)")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"DMA-KWS model missing at {path}; build it with scripts/export_dma_kws_onnx.py "
            "and copy it there (see satellite/models/README.md)"
        )
    if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError(f"DMA-KWS model at {path} does not match the pinned SHA-256")
    return path


def phoneme_ids(phonemes: str) -> np.ndarray:
    tokens = phonemes.upper().split()
    unknown = [t for t in tokens if t not in PHONE_IDS]
    if unknown or len(tokens) != ANCHOR_LEN:
        raise ValueError(
            f"keyword {phonemes!r} must be exactly {ANCHOR_LEN} ARPAbet phonemes "
            f"(like 'K OW1 D AH0'); unknown: {unknown}"
        )
    return np.array([[PHONE_IDS[t] for t in tokens]], dtype=np.int64)


class DmaKwsVerifier:
    """Second stage: DMA-KWS Stage II text-to-audio matcher on a 0.8 s crop at the end of
    the fired window. Same interface as WakeVerifier (score/threshold); the score is the best
    match over the configured pronunciations, 0..1."""

    def __init__(
        self,
        model_path: Optional[Path],
        phonemes: Sequence[str],
        threshold: float,
        session: Any = None,
    ) -> None:
        if not phonemes:
            raise ValueError("wake_word.dma_kws_phonemes is required (for example ['K OW1 D AH0'])")
        self._anchors = [phoneme_ids(p) for p in phonemes]
        self.threshold = float(threshold)
        if session is not None:  # injected for tests
            self._session = session
            return
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # fastest on a Pi 4 and leaves cores for audio
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(_verified_model(model_path)), opts, providers=["CPUExecutionProvider"]
        )

    def score(self, window: np.ndarray, model: Any = None, embeddings: Any = None) -> Optional[float]:
        audio = window.reshape(-1)
        audio = audio.astype(np.float32) / 32768.0 if audio.dtype == np.int16 else audio.astype(np.float32)
        seg = crop_after_speech(audio)
        if seg is None:
            return 0.0
        feats = fbank(seg)[None]
        best = max(float(self._session.run(None, {"feats": feats, "anchor": a})[0][0]) for a in self._anchors)
        _LOGGER.debug("DMA-KWS score %.3f", best)
        return best
