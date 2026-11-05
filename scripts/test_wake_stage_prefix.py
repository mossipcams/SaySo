import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.wake_stage_prefix import FADE, prefix_cut, speech_span


def _clip() -> np.ndarray:
    # 0.3 s silence, 0.5 s "word", 0.4 s silence
    a = np.zeros(19200, dtype=np.float32)
    t = np.arange(8000) / 16000
    a[4800:12800] = 0.3 * np.sin(2 * np.pi * 220 * t)
    return a


def test_speech_span_finds_the_word():
    assert speech_span(_clip()) == (4800, 12800)


def test_prefix_cut_keeps_fraction_of_the_word_and_fades():
    out = prefix_cut(_clip(), 0.5)
    assert len(out) == 4800 + 4000
    assert out[-1] == 0.0
    assert np.abs(out[-FADE // 4 :]).max() < 0.3 * 0.3
    assert np.array_equal(out[: 8800 - FADE], _clip()[: 8800 - FADE])
