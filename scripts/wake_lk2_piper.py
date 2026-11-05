#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import importlib
import json
import random
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
from livekit.wakeword.data.piper import synthesis as P

G = importlib.import_module("livekit_wakeword_bench.generate")
T = importlib.import_module("livekit_wakeword_bench.tts")
V = importlib.import_module("livekit_wakeword_bench.voices")
TX = importlib.import_module("livekit_wakeword_bench.text")

MODEL = Path("/srv/llm/wake/livekit-data/piper/en-us-libritts-high.pt")
CFG = json.loads(MODEL.with_suffix(".json").read_text())
_lock, _vits = threading.Lock(), []
torch.set_num_threads(4)


def _speak(text: str, voice: str, speed: float) -> np.ndarray:
    with _lock:
        if not _vits:
            _vits.append(P._load_vits_model(MODEL, torch.device("cpu")))
        a, b, w = voice.split("-")
        ids = P.get_phonemes(CFG, text, CFG["espeak"]["voice"])
        with torch.no_grad():
            y = P._generate_audio(
                _vits[0],
                torch.LongTensor([int(a)]),
                torch.LongTensor([int(b)]),
                [ids],
                [len(ids)],
                float(w),
                0.98,
                0.98,
                1.0 / speed,
                torch.device("cpu"),
            )
        return y.squeeze().cpu().numpy().astype(np.float32)


async def _synthesize(self, req, load: bool = True):
    path = self.cache_path(req)
    if path.exists():
        return sf.read(path, dtype="float32")[0] if load else np.zeros(0, np.float32)
    y = soxr.resample(await asyncio.to_thread(_speak, req.text, req.voice, req.speed), CFG["audio"]["sample_rate"], self.sample_rate).astype(
        np.float32
    )
    if len(y) < 0.15 * self.sample_rate or np.abs(y).max() < 1e-3:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y, self.sample_rate)
    return y if load else np.zeros(0, np.float32)


def _speed(model: str, speed: float) -> dict:
    return {} if abs(speed - 1.0) < 1e-3 else {"length_scale": 1.0 / speed}


_rng = random.Random(7)
VOICES = [
    V.Voice("piper/libritts-high", f"{a}-{b}-{w}", f"p{a}x{b}", "unknown", "US", "en-us")
    for a, b, w in (
        (_rng.randrange(CFG["num_speakers"]), _rng.randrange(CFG["num_speakers"]), _rng.choice([0.2, 0.35, 0.5, 0.65, 0.8])) for _ in range(800)
    )
]
T.Synthesizer.synthesize = _synthesize
TX.adversarial_phrases = lambda wake_word, *a, **k: []
T.speed_kwargs = G.speed_kwargs = _speed
G.load_voices = lambda models=None: VOICES
G.voices_for_language = lambda voices, language: voices
assert G.Synthesizer is T.Synthesizer and G.speed_kwargs is _speed and G.load_voices() is VOICES
