# Plan: stop the satellite dropping wake windows

**Date:** 2026-11-06
**Problem:** the Pi satellite never scores some 80 ms wake windows.
- With the TV on, 14-19% of windows go unscored. In quiet hours almost none do; there, drops happen mostly at speech onsets, which is exactly when "Koda" is said.
- A dropped window that would have scored ≥ 0.65 is either a missed wake or a late one.
- A late fire clips the command start, because the STT handoff starts at `detection_index - wake_skip_ms`.

## Cause

1. **The full recompute is slow.**
   - The mel ONNX (`power_to_db`, 80 dB `top_db`) clips each 2 s window to a floor 80 dB below that window's maximum.
   - When the loudest sound in the window changes, every frame's clipped bins change. `CachedEmbeddingScorer._reusable_shift` then finds no match, and all 16 speech embeddings are recomputed.
   - Normal reuse costs 40 ms per window. A full recompute costs ~270 ms, at 15 ms per embedding single-threaded on a Pi 4.
2. **The queue holds one window.** `WakeInferenceWorker` uses `LatestWindowQueue` (maxsize 1), so windows submitted during a recompute overwrite each other.
3. **`WakeAudioBuffer.feed` is not the cause.** The mic block (1024 samples) is smaller than the hop (1280), so `feed` never skips.

## Evidence

Real-time replay on the Pi of 8.7 min of 35 real wake recordings through the deployed buffer, worker, provider and scorer. Artifacts are in `llm:/srv/llm/data/wake/proof-hopdrop-20261106/`.

| | Current code | FIFO + 2-thread embedding |
|---|---|---|
| Windows dropped | 16.5% | 0 |
| Ideal wakes caught | 33/37 | 37/37 |
| Late fires | 16 (80-320 ms) | 0 |
| Score vs exact offline | 0.0000 | 0.0000 |
| Scoring lag | p50 42 ms, max 415 ms | p50 39 ms, p99 558 ms, max 1069 ms |

## Scope

Satellite only: `satellite/sayso/wake/`. Model, threshold, 80 ms hop, refractory, mining, verifier and HA path are unchanged.

1. **`worker.py`:** replace `LatestWindowQueue` with a bounded FIFO.
   - Capacity 25 windows, which is 2 s of hops.
   - It stores a copy of each window. `WakeAudioBuffer.window()` returns a view into the ring, so a queued view would be overwritten before it is scored.
   - When full, it drops the oldest window and increments a dropped counter.
   - It exposes `clear()`.
2. **2-thread speech embedding.** The speech-embedding ONNX session uses `intra_op_num_threads = 2`. The mel and classifier sessions stay single-threaded.
   - Measured on the Pi: a 16-embedding recompute goes from 230 ms to 145 ms.
   - Put it where the model is loaded: `livekit.py` `_load`, or a helper in `streaming.py` next to `single_threaded_ort`.
3. **`hook.py`:** clear the worker queue on re-arm, suspend and resume, so windows from before a TTS/re-arm boundary can never fire.
4. **Observability:** the worker logs a warning with the cumulative dropped count when it drops a window, at most once per second.
5. **Handoff ring:** confirm that the audio ring used for the wake→STT handoff holds at least 2 s of queue lag plus `wake_skip_ms` (220 ms) plus `preroll_ms`.
   - Read the capacity, don't assume it.
   - If it is too small, size it up in the same change.

Files to touch: `satellite/sayso/wake/worker.py`, `satellite/sayso/wake/livekit.py`, `satellite/sayso/wake/streaming.py`, `satellite/sayso/wake/hook.py`, plus colocated tests. Touch the ring file only if item 5 requires it.

## Slices

Each slice is dispatched separately and must build and verify on its own.

| Slice | Change | Files |
|---|---|---|
| 1 | Bounded FIFO, window copies, oldest-drop counter and rate-limited warning, `clear()` | `worker.py`, `test_worker.py` |
| 2 | 2-thread speech-embedding session | `livekit.py`, `streaming.py`, `test_livekit.py` |
| 3 | Clear the queue on re-arm, suspend and resume; check the handoff ring capacity | `hook.py`, `test_handoff.py`, plus the ring file only if resizing is needed |

## Out of scope

- Model changes. Quiet-room model misses are issue #186.
- Streaming-feature retrain: the long-term follow-up for a fixed per-hop cost.
- Deploying to the Pi, which needs the user's OK.

## Verification

- **Unit tests (colocated `satellite/sayso/wake/test_*.py`):**
  - FIFO preserves order and never drops while below capacity.
  - Queued windows are independent copies; mutating the source buffer after `offer` does not change them.
  - When full, the oldest window is dropped and counted.
  - `clear()` empties the queue.
  - A slow fake predict (e.g. 270 ms every 10th window) fed at the real 64 ms block cadence scores every hop, in order, with the right `sample_index`.
  - Re-arm and suspend clear pending windows: no detection from a pre-re-arm window.
  - The embedding session is built with 2 intra-op threads; the mel and classifier sessions with 1.
- **CI gate:** `ruff check .`, `ruff format --check .`, `pytest -q satellite/sayso`.
- **Pi replay, before deploy:** rerun the proof replay against the changed code from a temporary path on the Pi, not `/opt`.
  - Acceptance: 0 dropped, 37/37 wakes, 0 late.
  - Scores must match offline truth to within 1e-04. Two intra-op threads change float accumulation order: up to 2e-05 on embeddings and about 1e-05 on scores.
