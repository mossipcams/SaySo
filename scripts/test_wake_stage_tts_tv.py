import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.wake_stage_tts_tv import active_db, ltas_db, match_fir, mix_at


def test_voice_level_is_absolute_and_tv_keeps_its_shape():
    voice = np.zeros(32000, np.float32)
    voice[24000:32000] = 0.5 * np.sin(2 * np.pi * 200 * np.arange(8000) / 16000)
    tv = np.zeros(32000, np.float32)
    tv[:16000] = 0.01  # TV gated to zero under the word
    y = mix_at(voice, tv, -20.0, 6.0)
    assert abs(active_db(y[24000:]) + 20) < 1e-3  # voice not shrunk to the silent TV under it
    assert np.allclose(y[:16000], 0.01 * 10 ** (6 / 20))


def test_match_fir_imposes_a_band_limit():
    rng = np.random.default_rng(1)
    src = [rng.standard_normal(16000).astype(np.float32) for _ in range(8)]
    k = np.fft.rfftfreq(16000, 1 / 16000)
    lp = lambda x: np.fft.irfft(np.fft.rfft(x) * (k < 6500), 16000).astype(np.float32)
    tgt = [lp(x) for x in src]
    fir = match_fir(tgt, src)
    out = [np.convolve(x, fir, mode="same").astype(np.float32) for x in src]
    f = np.linspace(0, 8000, 257)
    d = ltas_db(out) - ltas_db(src)
    assert d[(f > 7000)].mean() < -15 and abs(d[(f > 500) & (f < 5000)].mean()) < 2
