"""Reuse speech embeddings across overlapping wake windows.

``WakeWordModel.predict()`` is stateless by design: it recomputes the mel
spectrogram over the whole 2 s chunk and then runs the speech-embedding model
once per 76-frame window at stride 8 — 16 embedding passes per call. Consecutive
wake windows overlap by 92%, so 14 of those 16 are recomputed from identical
audio every hop.

Measured on the satellite's Pi 4: 249 ms per predict against a 160 ms hop
budget (1.56x realtime), with the embedding passes accounting for ~90-125 ms of
it. ``LatestWindowQueue`` drops whatever arrives mid-inference, so the satellite
was discarding a large share of its own windows -- and a genuine "SaySo" is only
~600 ms, so losing window alignments loses the one where the phrase sits best in
frame.

The reuse here is exact, not approximate. A 160 ms hop shifts the mel by exactly
16 frames, which is two stride-8 steps, so embedding *i* of this window is
embedding *i+2* of the last one over the same mel frames. Verified empirically:
mel frames in the overlap region are bit-identical across a hop (max abs
difference 0.0). Every reuse is re-verified at runtime by comparing the actual
mel overlap, so a discontinuity (rearm, dropped audio, changed hop) falls back to
a full recompute rather than scoring stale audio.

Batching the 16 windows into one embedding call was measured and rejected: 93.1 ms
batched vs 90.5 ms looped (the model is compute-bound, not call-bound), and
batching perturbs the result by ~2.7e-05.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Iterator, Optional

import numpy as np

_LOGGER = logging.getLogger(__name__)


@contextlib.contextmanager
def single_threaded_ort() -> Iterator[None]:
    """Force ORT sessions built inside this block to one non-spinning thread.

    Measured on the satellite's Pi 4 over 60 windows:

        default (all cores)        p50 53.6 ms   p95 68.5 ms   399% CPU
        1 thread                   p50 42.5 ms   p95 44.2 ms   100% CPU
        1 thread, no spinning      p50 39.2 ms   p95 39.5 ms   100% CPU

    Faster *and* a quarter of the CPU: the mel, embedding and classifier graphs
    are far too small to parallelise, so the pool's synchronisation dominates the
    work, and its spin-wait burns three cores doing nothing between hops.

    Scoped to a context manager so only the wake models are affected -- anything
    else in the process that wants ORT's defaults keeps them.
    """
    try:
        import onnxruntime as ort
    except ImportError:  # pragma: no cover - onnxruntime always present on the satellite
        yield
        return

    original = ort.InferenceSession

    def build(path_or_bytes, *args, **kwargs):  # noqa: ANN001, ANN202
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        # args[0] would be a caller-supplied sess_options; overriding it is the
        # whole point, so positional session options are dropped deliberately.
        providers = kwargs.pop("providers", None) or ["CPUExecutionProvider"]
        kwargs.pop("sess_options", None)
        return original(path_or_bytes, sess_options=opts, providers=providers, **kwargs)

    ort.InferenceSession = build
    try:
        yield
    finally:
        ort.InferenceSession = original

EMBEDDING_WINDOW = 76  # mel frames per embedding
EMBEDDING_STRIDE = 8  # mel frames between embeddings
MIN_EMBEDDINGS = 16  # classifier input length


class CachedEmbeddingScorer:
    """Drop-in replacement for ``WakeWordModel.predict`` with embedding reuse.

    Falls back to ``model.predict()`` whenever the upstream internals this
    depends on are absent, so a livekit-wakeword upgrade degrades to the old
    speed rather than breaking wake detection.
    """

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

    @property
    def supported(self) -> bool:
        return self._supported

    @property
    def stats(self) -> tuple[int, int]:
        """(computed, reused) embedding counts since start, for logging."""
        return self._embeddings_computed, self._embeddings_reused

    def reset(self) -> None:
        """Drop cached state. Call after rearm/suspend so no stale audio is reused."""
        self._prev_mel = None
        self._prev_embeddings = None

    def _mel(self, audio: np.ndarray) -> np.ndarray:
        mel = self._model._mel_frontend(audio)
        return mel[0] if mel.ndim == 3 else mel

    def _embed(self, mel: np.ndarray, start: int) -> np.ndarray:
        window = mel[start : start + EMBEDDING_WINDOW]
        return self._model._speech_embedding(window[np.newaxis, :, :])[0]

    def _reusable_shift(self, mel: np.ndarray) -> Optional[int]:
        """Embedding-index shift if this window is a clean stride-aligned slide.

        Returns None when there is no usable cache or the mel overlap does not
        match, which forces a full recompute.
        """
        if self._prev_mel is None or self._prev_embeddings is None:
            return None
        if mel.shape != self._prev_mel.shape:
            return None
        n_frames = mel.shape[0]
        for shift_emb in range(1, MIN_EMBEDDINGS):
            shift_frames = shift_emb * EMBEDDING_STRIDE
            if shift_frames >= n_frames:
                break
            # The new window slid forward by shift_frames, so the tail of the
            # previous mel must equal the head of this one.
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
        """Return ``{model_name: score}`` for a ~2 s int16 or float32 window."""
        classifiers = getattr(self._model, "_classifiers", None)
        if not self._supported or not classifiers:
            return self._model.predict(audio)

        raw = audio
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        audio = audio.flatten()

        mel = self._mel(audio)
        # hasattr() cannot establish that these internals behave as expected -
        # any stub or a reshaped upstream passes it. Verify the actual output and
        # degrade permanently rather than guessing at a MagicMock's shape.
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
            # Embedding i here covers the same mel frames as embedding i+shift
            # of the previous window.
            assert self._prev_embeddings is not None
            embeddings = list(self._prev_embeddings[shift:])
            for s in starts[len(embeddings) :]:
                embeddings.append(self._embed(mel, s))
            # shift embeddings fell off the front, so shift new ones are computed
            # and the remaining MIN_EMBEDDINGS - shift are reused.
            self._embeddings_reused += MIN_EMBEDDINGS - shift
            self._embeddings_computed += shift

        self._prev_mel = mel
        self._prev_embeddings = embeddings

        emb_input = np.stack(embeddings, axis=0)[np.newaxis, :, :].astype(np.float32)
        scores: dict[str, float] = {}
        for name, (session, input_name) in classifiers.items():
            outputs = session.run(None, {input_name: emb_input})
            scores[name] = float(outputs[0][0, 0])
        return scores


def demo() -> None:
    """Self-check against a fake model: reuse must be exact and cache-invalidating.

    Uses a deterministic stand-in for the mel/embedding ONNX models so this runs
    anywhere. Equivalence against the real livekit models is checked on the
    satellite by scripts/wake_bench.py.
    """

    class FakeSession:
        def run(self, _out, feed):
            emb = next(iter(feed.values()))
            return [np.array([[float(emb.mean())]], dtype=np.float32)]

    class FakeModel:
        """Mel frame f is a deterministic function of its absolute audio offset."""

        def __init__(self) -> None:
            self._classifiers = {"sayso": (FakeSession(), "embeddings")}
            self.embed_calls = 0

        def _mel_frontend(self, audio):
            # Real mel frames depend only on absolute audio position, which is
            # what makes overlap bit-identical across a slide. The stream in this
            # demo is arange/1e6, so sample offset recovers as audio[0]*1e6 and
            # the absolute frame index as that over the 160-sample frame hop.
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

    # A continuous stream sliding by exactly one 160 ms hop each step.
    stream = np.arange(32000 + 2560 * 6, dtype=np.float32) / 1e6
    windows = [stream[i * 2560 : i * 2560 + 32000] for i in range(6)]

    scores = [scorer.score(w)["sayso"] for w in windows]
    after_cached = model.embed_calls

    # Same windows, no cache: results must match exactly.
    fresh_model = FakeModel()
    fresh = CachedEmbeddingScorer(fresh_model)
    reference = []
    for w in windows:
        fresh.reset()  # force full recompute every time
        reference.append(fresh.score(w)["sayso"])

    assert scores == reference, f"cached scores diverged: {scores} != {reference}"
    assert after_cached < fresh_model.embed_calls, "cache did not reduce embedding calls"
    computed, reused = scorer.stats
    assert reused > 0, "expected reuse on a clean slide"
    # 16 for the cold window, then 2 per slide; stats must agree with reality.
    assert computed == after_cached == 16 + 5 * 2, f"stat/reality mismatch: {computed} vs {after_cached}"
    assert reused == 5 * 14, f"expected 70 reused, got {reused}"
    print(
        f"6 sliding windows: {after_cached} embed calls cached vs "
        f"{fresh_model.embed_calls} uncached (reused {reused})"
    )

    # Discontinuity must invalidate: an unrelated window cannot reuse anything.
    scorer.reset()
    before = model.embed_calls
    scorer.score(np.full(32000, 0.5, dtype=np.float32))
    assert model.embed_calls - before == MIN_EMBEDDINGS, "reset must force full recompute"

    # A jump in the stream (no mel overlap) must also fall back to full recompute.
    before = model.embed_calls
    scorer.score(np.full(32000, 0.9, dtype=np.float32))
    assert model.embed_calls - before == MIN_EMBEDDINGS, "discontinuity must not reuse"

    # single_threaded_ort must restore the original factory even on failure,
    # or every later ORT user in the process silently inherits our settings.
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
