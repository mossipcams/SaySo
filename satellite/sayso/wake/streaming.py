
from __future__ import annotations

import contextlib
import logging
from typing import Any, Iterator, Optional

import numpy as np

_LOGGER = logging.getLogger(__name__)


@contextlib.contextmanager
def single_threaded_ort() -> Iterator[None]:
    try:
        import onnxruntime as ort
    except ImportError:
        yield
        return

    original = ort.InferenceSession

    def build(path_or_bytes, *args, **kwargs):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        providers = kwargs.pop("providers", None) or ["CPUExecutionProvider"]
        kwargs.pop("sess_options", None)
        return original(path_or_bytes, sess_options=opts, providers=providers, **kwargs)

    ort.InferenceSession = build
    try:
        yield
    finally:
        ort.InferenceSession = original

EMBEDDING_WINDOW = 76
EMBEDDING_STRIDE = 8
MIN_EMBEDDINGS = 16


class CachedEmbeddingScorer:

    def __init__(self, model: Any) -> None:
        self._model = model
        self._prev_mel: Optional[np.ndarray] = None
        self._prev_embeddings: Optional[list[np.ndarray]] = None
        self._supported = all(
            hasattr(model, attr)
            for attr in ("_mel_frontend", "_speech_embedding", "_classifiers")
        )
        if not self._supported:
            _LOGGER.warning(
                "livekit WakeWordModel internals not found; falling back to stateless "
                "predict(). Wake inference will not keep up with a 160 ms hop."
            )
        self._embeddings_computed = 0
        self._embeddings_reused = 0
        self._last_embeddings: Optional[np.ndarray] = None

    @property
    def supported(self) -> bool:
        return self._supported

    @property
    def stats(self) -> tuple[int, int]:
        return self._embeddings_computed, self._embeddings_reused

    def reset(self) -> None:
        self._prev_mel = None
        self._prev_embeddings = None
        self._last_embeddings = None

    @property
    def last_embeddings(self) -> Optional[np.ndarray]:
        return self._last_embeddings

    def _mel(self, audio: np.ndarray) -> np.ndarray:
        mel = self._model._mel_frontend(audio)
        return mel[0] if mel.ndim == 3 else mel

    def _embed(self, mel: np.ndarray, start: int) -> np.ndarray:
        window = mel[start : start + EMBEDDING_WINDOW]
        return self._model._speech_embedding(window[np.newaxis, :, :])[0]

    def _reusable_shift(self, mel: np.ndarray) -> Optional[int]:
        if self._prev_mel is None or self._prev_embeddings is None:
            return None
        if mel.shape != self._prev_mel.shape:
            return None
        n_frames = mel.shape[0]
        for shift_emb in range(1, MIN_EMBEDDINGS):
            shift_frames = shift_emb * EMBEDDING_STRIDE
            if shift_frames >= n_frames:
                break
            if np.array_equal(self._prev_mel[shift_frames:], mel[: n_frames - shift_frames]):
                return shift_emb
        return None

    def _disable(self, reason: str) -> None:
        self._supported = False
        self.reset()
        _LOGGER.warning(
            "Embedding reuse disabled (%s); falling back to stateless predict(). "
            "Wake inference will not keep up with a 160 ms hop.",
            reason,
        )

    def score(self, audio: np.ndarray) -> dict[str, float]:
        classifiers = getattr(self._model, "_classifiers", None)
        if not self._supported or not classifiers:
            return self._model.predict(audio)

        raw = audio
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        audio = audio.flatten()

        mel = self._mel(audio)
        if not isinstance(mel, np.ndarray) or mel.ndim != 2:
            self._disable(f"mel frontend returned {type(mel).__name__}")
            return self._model.predict(raw)

        if mel.shape[0] < EMBEDDING_WINDOW:
            return {name: 0.0 for name in classifiers}

        starts = list(range(0, mel.shape[0] - EMBEDDING_WINDOW + 1, EMBEDDING_STRIDE))
        if len(starts) < MIN_EMBEDDINGS:
            return {name: 0.0 for name in classifiers}
        starts = starts[-MIN_EMBEDDINGS:]

        shift = self._reusable_shift(mel)
        if shift is None:
            embeddings = [self._embed(mel, s) for s in starts]
            self._embeddings_computed += len(starts)
        else:
            assert self._prev_embeddings is not None
            embeddings = list(self._prev_embeddings[shift:])
            for s in starts[len(embeddings) :]:
                embeddings.append(self._embed(mel, s))
            self._embeddings_reused += MIN_EMBEDDINGS - shift
            self._embeddings_computed += shift

        self._prev_mel = mel
        self._prev_embeddings = embeddings

        stacked = np.stack(embeddings, axis=0).astype(np.float32)
        self._last_embeddings = stacked
        emb_input = stacked[np.newaxis, :, :]
        scores: dict[str, float] = {}
        for name, (session, input_name) in classifiers.items():
            outputs = session.run(None, {input_name: emb_input})
            scores[name] = float(outputs[0][0, 0])
        return scores


def demo() -> None:

    class FakeSession:
        def run(self, _out, feed):
            emb = next(iter(feed.values()))
            return [np.array([[float(emb.mean())]], dtype=np.float32)]

    class FakeModel:

        def __init__(self) -> None:
            self._classifiers = {"sayso": (FakeSession(), "embeddings")}
            self.embed_calls = 0

        def _mel_frontend(self, audio):
            n_frames = audio.size // 160 - 3
            offset = round(float(audio[0]) * 1e6 / 160.0)
            base = (np.arange(n_frames, dtype=np.float32) + offset)[:, None]
            return base @ np.ones((1, 32), dtype=np.float32)

        def _speech_embedding(self, window):
            self.embed_calls += 1
            return window.mean(axis=1)[:, :96] if window.shape[2] >= 96 else np.repeat(
                window.mean(axis=(1, 2))[:, None], 96, axis=1
            ).astype(np.float32)

        def predict(self, audio):
            return {"sayso": 0.0}

    model = FakeModel()
    scorer = CachedEmbeddingScorer(model)
    assert scorer.supported

    stream = np.arange(32000 + 2560 * 6, dtype=np.float32) / 1e6
    windows = [stream[i * 2560 : i * 2560 + 32000] for i in range(6)]

    scores = [scorer.score(w)["sayso"] for w in windows]
    after_cached = model.embed_calls

    fresh_model = FakeModel()
    fresh = CachedEmbeddingScorer(fresh_model)
    reference = []
    for w in windows:
        fresh.reset()
        reference.append(fresh.score(w)["sayso"])

    assert scores == reference, f"cached scores diverged: {scores} != {reference}"
    assert after_cached < fresh_model.embed_calls, "cache did not reduce embedding calls"
    computed, reused = scorer.stats
    assert reused > 0, "expected reuse on a clean slide"
    assert computed == after_cached == 16 + 5 * 2, f"stat/reality mismatch: {computed} vs {after_cached}"
    assert reused == 5 * 14, f"expected 70 reused, got {reused}"
    print(
        f"6 sliding windows: {after_cached} embed calls cached vs "
        f"{fresh_model.embed_calls} uncached (reused {reused})"
    )

    scorer.reset()
    before = model.embed_calls
    scorer.score(np.full(32000, 0.5, dtype=np.float32))
    assert model.embed_calls - before == MIN_EMBEDDINGS, "reset must force full recompute"

    before = model.embed_calls
    scorer.score(np.full(32000, 0.9, dtype=np.float32))
    assert model.embed_calls - before == MIN_EMBEDDINGS, "discontinuity must not reuse"

    try:
        import onnxruntime as ort
    except ImportError:
        print("streaming self-check ok (onnxruntime absent; ORT scoping unchecked)")
        return
    before = ort.InferenceSession
    with single_threaded_ort():
        assert ort.InferenceSession is not before, "context manager did not patch"
    assert ort.InferenceSession is before, "context manager did not restore"
    try:
        with single_threaded_ort():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert ort.InferenceSession is before, "must restore after an exception"

    print("streaming self-check ok")


if __name__ == "__main__":
    demo()
