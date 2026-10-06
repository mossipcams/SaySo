from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

_LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class MoonshineVerifier:
    """Second stage: transcribe the fired window with Moonshine v2 tiny-streaming,
    biased toward the wake phrase, and pass only if it hears the phrase.

    Same interface as WakeVerifier (score/threshold) so LiveKitWakeWordProvider
    treats both alike. Score is 1.0 (heard) or 0.0 (not heard).
    """

    threshold = 0.5

    def __init__(
        self,
        phrase: str,
        accept: Optional[Sequence[str]] = None,
        boost: float = 3.0,
        window_ms: int = 1200,
        pad_ms: int = 500,
        transcriber: Any = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        # The phrase is always accepted: Moonshine is biased toward it.
        words = [phrase.lower(), *(w.lower().strip() for w in (accept or ()) if w.strip())]
        self._pattern = re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.I)
        self._window = int(window_ms * SAMPLE_RATE / 1000)
        self._pad = np.zeros(int(pad_ms * SAMPLE_RATE / 1000), dtype=np.float32)
        self.last_text = ""
        if transcriber is not None:  # injected for tests
            self._tr = transcriber
            return
        import moonshine_voice as ms
        from moonshine_voice.transcriber import Transcriber

        arch = ms.ModelArch.TINY_STREAMING
        path, _ = ms.get_model_for_language("en", arch, cache_root=Path(cache_dir) if cache_dir else None)
        self._tr = Transcriber(path, arch, options={"keyterm_boost": str(boost)})
        self._tr.set_keyterms([phrase])

    def score(self, window: np.ndarray, model: Any = None, embeddings: Any = None) -> Optional[float]:
        audio = window.reshape(-1)[-self._window :]
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        x = np.concatenate([audio.astype(np.float32, copy=False), self._pad])
        transcript = self._tr.transcribe_without_streaming(list(x), SAMPLE_RATE)
        self.last_text = " ".join(line.text for line in transcript.lines)
        _LOGGER.debug("Moonshine heard %r", self.last_text)
        return 1.0 if self._pattern.search(self.last_text) else 0.0


def demo() -> None:
    from types import SimpleNamespace

    class FakeTr:
        def __init__(self, text: str) -> None:
            self.text, self.n = text, 0

        def transcribe_without_streaming(self, samples, rate):
            self.n = len(samples)
            return SimpleNamespace(lines=[SimpleNamespace(text=self.text)])

    win = np.zeros(32000, dtype=np.int16)
    v = MoonshineVerifier("Koda", accept=["koda", "coda"], transcriber=FakeTr("Okay, Coda."))
    assert v.score(win) == 1.0
    assert v._tr.n == 19200 + 8000, "last 1.2 s plus 0.5 s pad"
    assert MoonshineVerifier("Koda", transcriber=FakeTr("Hold on.")).score(win) == 0.0
    assert MoonshineVerifier("Koda", transcriber=FakeTr("Kodak moment")).score(win) == 0.0, "word boundary"
    print("moonshine verifier self-check ok")


if __name__ == "__main__":
    demo()
