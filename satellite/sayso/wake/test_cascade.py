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


# --- Moonshine verifier wiring -------------------------------------------------

import yaml  # noqa: E402

from sayso.config import load_config, validate_config  # noqa: E402
from sayso.wake.moonshine_verifier import MoonshineVerifier  # noqa: E402


class _FakeTranscriber:
    def __init__(self, text: str) -> None:
        self.text = text

    def transcribe_without_streaming(self, samples, rate):
        return SimpleNamespace(lines=[SimpleNamespace(text=self.text)])


def _moonshine(text: str, accept=None) -> MoonshineVerifier:
    return MoonshineVerifier("Koda", accept=accept, transcriber=_FakeTranscriber(text))


def test_moonshine_hears_phrase_or_variant() -> None:
    win = np.zeros(WINDOW_SAMPLES, dtype=np.int16)
    assert _moonshine("Okay, Koda.").score(win) == 1.0
    assert _moonshine("Coda,", accept=["koda", "coda"]).score(win) == 1.0
    assert _moonshine("Hold on.").score(win) == 0.0
    assert _moonshine("Kodak").score(win) == 0.0


def test_moonshine_veto_does_not_lock_out_cascade() -> None:
    p = _provider([0.30, 0.10, 0.60], [])
    p._verifier = _FakeTranscribeStage(["Hold on.", "Koda."])
    out = _hops(p, 3)
    assert out[0] is None and out[2] is not None


class _FakeTranscribeStage(MoonshineVerifier):
    def __init__(self, texts: list[str]) -> None:
        self._texts = iter(texts)
        super().__init__("Koda", transcriber=self)

    def transcribe_without_streaming(self, samples, rate):
        return SimpleNamespace(lines=[SimpleNamespace(text=next(self._texts))])


def test_verifier_factory_failure_fails_closed() -> None:
    p = _provider([0.5], [True])
    p._verifier = None

    def boom():
        raise RuntimeError("no model")

    p._load_verifier_factory(boom)
    assert p._available is False
    assert _hops(p, 1) == [None]


def test_moonshine_config_roundtrip_and_exclusive_with_npz(tmp_path: Path) -> None:
    sound = Path(__file__).parents[2] / "sounds" / "wake.wav"
    model = tmp_path / "wake.onnx"
    model.write_bytes(b"onnx")
    raw = {
        "satellite": {"name": "L", "device_name": "l", "area": "L"},
        "home_assistant": {"port": 6053},
        "audio": {"input_device": "mic", "output_device": "spk", "sample_rate": 16000,
                  "channels": 1, "noise_suppression": 0, "auto_gain": 0},
        "wake_word": {"provider": "livekit", "phrase": "Koda", "model": str(model),
                      "threshold": 0.4, "refractory_seconds": 2.0, "preroll_ms": 500,
                      "post_tts_cooldown_ms": 500, "moonshine_verifier": True,
                      "moonshine_accept": ["koda", "coda"]},
        "sounds": {"wake": str(sound), "failure": str(sound), "unavailable": str(sound)},
    }
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.wake_word.moonshine_verifier and cfg.wake_word.moonshine_boost == 3.0
    assert cfg.wake_word.moonshine_accept == ("koda", "coda")
    validate_config(cfg, check_port_bind=False)
    raw["wake_word"]["verifier"] = str(tmp_path / "v.npz")
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_config(load_config(path), check_port_bind=False)


def test_moonshine_blank_accept_entry_cannot_match_everything() -> None:
    win = np.zeros(WINDOW_SAMPLES, dtype=np.int16)
    v = MoonshineVerifier("Koda", accept=["", "  "], transcriber=_FakeTranscriber("Hold on."))
    assert v.score(win) == 0.0


def test_moonshine_accept_bare_string_is_one_word(tmp_path: Path) -> None:
    sound = Path(__file__).parents[2] / "sounds" / "wake.wav"
    model = tmp_path / "wake.onnx"
    model.write_bytes(b"onnx")
    raw = {
        "satellite": {"name": "L", "device_name": "l", "area": "L"},
        "home_assistant": {"port": 6053},
        "audio": {"input_device": "mic", "output_device": "spk", "sample_rate": 16000,
                  "channels": 1, "noise_suppression": 0, "auto_gain": 0},
        "wake_word": {"provider": "livekit", "phrase": "Koda", "model": str(model),
                      "threshold": 0.4, "refractory_seconds": 2.0, "preroll_ms": 500,
                      "post_tts_cooldown_ms": 500, "moonshine_accept": "koda"},
        "sounds": {"wake": str(sound), "failure": str(sound), "unavailable": str(sound)},
    }
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert load_config(path).wake_word.moonshine_accept == ("koda",)


def test_moonshine_phrase_always_accepted_with_custom_accept() -> None:
    win = np.zeros(WINDOW_SAMPLES, dtype=np.int16)
    # Moonshine is biased toward the phrase, so "Koda" must pass even if accept omits it.
    assert _moonshine("Koda.", accept=["kota", "coda"]).score(win) == 1.0
    assert _moonshine("Kota.", accept=["kota", "coda"]).score(win) == 1.0
    assert _moonshine("Hold on.", accept=["kota", "coda"]).score(win) == 0.0


@pytest.mark.parametrize("value,expected", [(None, ()), ("koda", ("koda",)), (["a", "b"], ("a", "b")), (5, ("5",))])
def test_moonshine_accept_tolerates_odd_yaml(value, expected) -> None:
    from sayso.config import _str_tuple

    assert _str_tuple(value) == expected


def test_moonshine_boost_must_be_positive(tmp_path: Path) -> None:
    sound = Path(__file__).parents[2] / "sounds" / "wake.wav"
    model = tmp_path / "wake.onnx"
    model.write_bytes(b"onnx")
    raw = {
        "satellite": {"name": "L", "device_name": "l", "area": "L"},
        "home_assistant": {"port": 6053},
        "audio": {"input_device": "mic", "output_device": "spk", "sample_rate": 16000,
                  "channels": 1, "noise_suppression": 0, "auto_gain": 0},
        "wake_word": {"provider": "livekit", "phrase": "Koda", "model": str(model),
                      "threshold": 0.4, "refractory_seconds": 2.0, "preroll_ms": 500,
                      "post_tts_cooldown_ms": 500, "moonshine_boost": 0,
                      "moonshine_cache_dir": str(tmp_path / "ms")},
        "sounds": {"wake": str(sound), "failure": str(sound), "unavailable": str(sound)},
    }
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="moonshine_boost"):
        load_config(path)  # load_config validates
    raw["wake_word"]["moonshine_boost"] = 2.5
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.wake_word.moonshine_boost == 2.5
    assert cfg.wake_word.moonshine_cache_dir == tmp_path / "ms"
