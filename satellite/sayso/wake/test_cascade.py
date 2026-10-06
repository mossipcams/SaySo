from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_SATELLITE_ROOT = Path(__file__).resolve().parents[2]
if str(_SATELLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SATELLITE_ROOT))

from sayso.wake.livekit import HOP_SAMPLES, WINDOW_SAMPLES, LiveKitWakeWordProvider


class _FakeScorer:
    """Scores come from a per-hop script; stands in for CachedEmbeddingScorer."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = iter(scores)
        self.last_embeddings = None
        self.stats = (0, 0)

    def score(self, _window: np.ndarray) -> dict[str, float]:
        return {"koda": next(self._scores)}


class _FakeStage2:
    """Stands in for a Sherpa verifier: per-call verdicts, same interface as WakeVerifier."""

    def __init__(self, verdicts: list[bool]) -> None:
        self._verdicts = iter(verdicts)
        self.threshold = 0.5
        self.calls = 0

    def score(self, _window, _model, embeddings=None) -> float:
        self.calls += 1
        return 1.0 if next(self._verdicts) else 0.0


def _provider(lk_scores: list[float], verdicts: list[bool]) -> LiveKitWakeWordProvider:
    p = object.__new__(LiveKitWakeWordProvider)
    p._phrase = "Koda"
    p._threshold = 0.22
    p._refractory = 2.0
    p._miner = None
    p._verifier = _FakeStage2(verdicts)
    p._enabled = True
    p._suspended = False
    p._available = True
    p._model = SimpleNamespace()
    p._scorer = _FakeScorer(lk_scores)
    p._score_key = "koda"
    p._last_fire_sample = None
    p._last_fire_time = None
    p._logged_keys = True
    p._last_score_log = 1e18
    p._max_score_window = 0.0
    return p


def _hops(p: LiveKitWakeWordProvider, n: int) -> list:
    window = np.zeros(WINDOW_SAMPLES, dtype=np.int16)
    return [p.predict_window(window, sample_index=i * HOP_SAMPLES) for i in range(n)]


def test_stage2_pass_fires() -> None:
    out = _hops(_provider([0.5], [True]), 1)
    assert out[0] is not None


def test_stage2_veto_blocks_fire() -> None:
    out = _hops(_provider([0.5], [False]), 1)
    assert out == [None]


def test_below_threshold_never_reaches_stage2() -> None:
    p = _provider([0.05, 0.05], [True, True])
    _hops(p, 2)
    assert p._verifier.calls == 0


def test_veto_on_early_hop_does_not_lock_out_a_real_wake() -> None:
    # Hop 0: word only partly in window, LiveKit crosses, stage 2 vetoes.
    # Hop 2 (320 ms later): word complete, stage 2 passes -> must fire.
    p = _provider([0.30, 0.10, 0.60], [False, True])
    out = _hops(p, 3)
    assert out[0] is None
    assert out[2] is not None, "veto started the refractory and swallowed the real wake"


# --- DMA-KWS verifier -----------------------------------------------------------

import yaml  # noqa: E402

from sayso.config import load_config, validate_config  # noqa: E402
from sayso.wake import dma_kws_verifier as dk  # noqa: E402
from sayso.wake.dma_kws_verifier import DmaKwsVerifier  # noqa: E402

KODA = ["K OW1 D AH0"]


class _FakeSession:
    """Stands in for onnxruntime: returns scripted scores and records its inputs."""

    def __init__(self, scores) -> None:
        self.scores, self.calls = list(scores), []

    def run(self, _outputs, feed):
        self.calls.append(feed)
        return [np.array([self.scores.pop(0) if len(self.scores) > 1 else self.scores[0]], dtype=np.float32)]


def _speech(end_s: float = 1.7, n: int = 32000) -> np.ndarray:
    """int16 window with a 0.4 s tone burst ending at end_s and quiet noise elsewhere."""
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(n) * 30).astype(np.int16)
    a, b = int((end_s - 0.4) * 16000), int(end_s * 16000)
    x[a:b] = (8000 * np.sin(2 * np.pi * 300 * np.arange(b - a) / 16000)).astype(np.int16)
    return x


def test_fbank_matches_torchaudio_reference() -> None:
    # Reference rows computed with torchaudio.compliance.kaldi.fbank (80 bins, povey, no dither).
    rng = np.random.default_rng(7)
    t = np.arange(12720) / 16000
    x = (0.2 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.standard_normal(12720)).astype(np.float32)
    f = dk.fbank(x)
    assert f.shape == (dk.FRAMES, 80)
    want = {0: [11.701, 16.932, 19.076, 22.602], 40: [9.322, 16.83, 20.042, 22.74], 77: [11.795, 15.667, 20.238, 24.164]}
    for row, vals in want.items():
        assert np.allclose(f[row, [0, 10, 40, 79]], vals, atol=2e-3)


def test_crop_ends_after_speech_and_is_fixed_size() -> None:
    audio = _speech(1.7).astype(np.float32) / 32768.0
    seg = dk.crop_after_speech(audio)
    assert seg.shape == (dk.CROP,)
    # speech burst ends 0.2 s before the crop end; the burst is the loud part of the crop
    loud = np.where(np.abs(seg) > 0.05)[0]
    assert abs((dk.CROP - loud.max()) / 16000 - 0.2) < 0.03
    # speech right at the window end: the crop is padded with zeros after it
    late = dk.crop_after_speech(_speech(2.0).astype(np.float32) / 32768.0)
    assert late.shape == (dk.CROP,) and np.abs(late[-3000:]).max() < 0.01
    assert dk.crop_after_speech(np.zeros(32000, dtype=np.float32)) is None


def test_phoneme_ids_validated() -> None:
    assert dk.phoneme_ids("k ow1 d ah0").tolist() == [[44, 50, 23, 9]]
    for bad in ("K OW1 D", "K OW1 D AH0 X", "K OW D AH0", "HH OW1 L D AA1 N"):
        with pytest.raises(ValueError, match="ARPAbet"):
            dk.phoneme_ids(bad)


def test_dma_kws_score_threshold_and_best_pronunciation() -> None:
    sess = _FakeSession([0.2, 0.97])
    v = DmaKwsVerifier(None, ["K OW1 D AH0", "K OW2 D AH0"], 0.98, session=sess)
    assert v.score(_speech()) == pytest.approx(0.97)  # best of the pronunciations
    assert v.score(_speech()) >= 0 and len(sess.calls) == 4
    assert sess.calls[0]["feats"].shape == (1, dk.FRAMES, 80)
    assert sess.calls[0]["anchor"].tolist() == [[44, 50, 23, 9]]
    assert v.threshold == 0.98 and v.score(np.zeros(32000, dtype=np.int16)) == 0.0


def test_dma_kws_veto_does_not_lock_out_cascade() -> None:
    p = _provider([0.30, 0.10, 0.60], [])
    p._verifier = DmaKwsVerifier(None, KODA, 0.9, session=_FakeSession([0.1, 0.99]))
    hops = []
    window = _speech()
    for i in range(3):
        hops.append(p.predict_window(window, sample_index=i * HOP_SAMPLES))
    assert hops[0] is None and hops[2] is not None


def test_model_file_is_pinned_never_downloaded(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(ValueError, match="dma_kws_model is required"):
        dk._verified_model(None)
    with pytest.raises(FileNotFoundError, match="export_dma_kws_onnx"):
        dk._verified_model(tmp_path / "missing.onnx")
    f = tmp_path / "m.onnx"
    f.write_bytes(b"not the pinned model")
    with pytest.raises(ValueError, match="pinned SHA-256"):
        dk._verified_model(f)
    import hashlib

    monkeypatch.setattr(dk, "MODEL_SHA256", hashlib.sha256(b"not the pinned model").hexdigest())
    assert dk._verified_model(f) == f
    f.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="pinned SHA-256"):
        dk._verified_model(f)


def test_verifier_factory_failure_fails_closed() -> None:
    p = _provider([0.5], [True])
    p._verifier = None

    def boom():
        raise RuntimeError("no model")

    p._load_verifier_factory(boom)
    assert p._available is False
    assert _hops(p, 1) == [None]


def _config(tmp_path: Path, **wake) -> Path:
    sound = Path(__file__).parents[2] / "sounds" / "wake.wav"
    model = tmp_path / "wake.onnx"
    model.write_bytes(b"onnx")
    raw = {
        "satellite": {"name": "L", "device_name": "l", "area": "L"},
        "home_assistant": {"port": 6053},
        "audio": {"input_device": "mic", "output_device": "spk", "sample_rate": 16000,
                  "channels": 1, "noise_suppression": 0, "auto_gain": 0},
        "wake_word": {"provider": "livekit", "phrase": "Koda", "model": str(model), "threshold": 0.4,
                      "refractory_seconds": 2.0, "preroll_ms": 500, "post_tts_cooldown_ms": 500, **wake},
        "sounds": {"wake": str(sound), "failure": str(sound), "unavailable": str(sound)},
    }
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_dma_kws_config_roundtrip_and_validation(tmp_path: Path) -> None:
    kws = tmp_path / "dma.onnx"
    kws.write_bytes(b"x")
    cfg = load_config(_config(tmp_path, dma_kws_model=str(kws), dma_kws_phonemes=["K OW1 D AH0"]))
    assert cfg.wake_word.dma_kws_model == kws
    assert cfg.wake_word.dma_kws_phonemes == ("K OW1 D AH0",) and cfg.wake_word.dma_kws_threshold == 0.98
    assert load_config(_config(tmp_path)).wake_word.dma_kws_model is None  # off by default
    for extra, msg in [
        ({"dma_kws_phonemes": ["K OW1 D AH0"], "dma_kws_model": str(tmp_path / "nope.onnx")}, "model file missing"),
        ({"dma_kws_model": str(kws)}, "dma_kws_phonemes is required"),
        ({"dma_kws_model": str(kws), "dma_kws_phonemes": ["K OW1 D"]}, "ARPAbet"),
        ({"dma_kws_model": str(kws), "dma_kws_phonemes": ["K OW1 D AH0"], "dma_kws_threshold": 1.5}, "dma_kws_threshold"),
        ({"dma_kws_model": str(kws), "dma_kws_phonemes": ["K OW1 D AH0"], "verifier": str(kws)}, "mutually exclusive"),
    ]:
        with pytest.raises(ValueError, match=msg):
            load_config(_config(tmp_path, **extra))


@pytest.mark.parametrize("value,expected", [(None, ()), ("K OW1 D AH0", ("K OW1 D AH0",)), (["a", "b"], ("a", "b")), (5, ("5",))])
def test_str_tuple_tolerates_odd_yaml(value, expected) -> None:
    from sayso.config import _str_tuple

    assert _str_tuple(value) == expected


def test_real_model_smoke_if_provisioned() -> None:
    import os

    path = os.environ.get("DMA_KWS_MODEL")
    if not path:
        pytest.skip("set DMA_KWS_MODEL to the exported ONNX to run")
    v = DmaKwsVerifier(Path(path), KODA, 0.98)
    assert 0.0 <= v.score(_speech()) <= 1.0
