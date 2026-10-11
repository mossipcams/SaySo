# Plan: address PR #132 review notes

Branch: `ajax/sattelite-recording` (worktree is at PR head). Work in-place.

## Scope (only these files)

- `satellite/sayso/launcher.py`
- `satellite/sayso/wake/hook.py`
- `satellite/sayso/wake/mining.py`
- `satellite/sayso/wake/test_mining.py`
- this plan file

`capture.py` needs no change: `WakeCaptureRing.end_index` already exists and is what the guard compares.

## Changes

1. Drift guard (Note 1) — `mining.py`
   - `bind_raw_ring` stores the raw ring itself (`self._raw_ring`) alongside the reader it stores today.
   - New private `_check_raw_ring_drift()`: only when both rings are bound, compares
     `raw.end_index` vs processed `end_index` (via getattr for duck-typed safety); if drift is
     nonzero and different from the last reported value, logs one warning with both indices.
     Silent and zero-behavior-change when aligned; bounded by the "changed since last report" gate.
   - Called at the two raw-sample read sites: `snapshot_pre_trigger` (after storing pending pre)
     and `_enqueue_cluster` (after its fallback raw read). No other logic changes, so mining
     results are byte-identical.

2. Launcher capacity source of truth (Note 2) — `launcher.py`, `hook.py`
   - `hook.py`: lift the existing capacity formula into a module-level
     `wake_ring_capacity(preroll_ms)`; `__init__` uses it for its default ring.
   - `launcher.py`: compute one explicit `ring_capacity = wake_ring_capacity(cfg.wake_word.preroll_ms)`,
     build the raw `WakeCaptureRing(ring_capacity)` (bound to miner before the hook, as today), and
     pass an explicit `capture_ring=WakeCaptureRing(ring_capacity)` to `SaySoExternalWakeHook`.
     No more `wake_hook._ring` private access. No new hook property needed — mining already holds
     both rings from bind calls.

3. Raw stream metadata (Note 3) — `mining.py`, `test_mining.py`
   - `_write_record`: `streams.raw` always includes `"sample_rate"` and `"channels": 1`;
     `sha256` is added only when `raw_pre.wav` is actually written.
   - Update/extend the missing-coverage test to assert the new metadata shape.

4. PR description (Note 4) — after verification, commit, push, then
   `gh pr edit 132 --body` correcting the raw_pre.wav claim to: stores only bounded pre-context
   (up to 8000 samples / 500ms) before the wake window, not the latest window. Rest of body kept.

## New test (test_mining.py)

`test_raw_ring_drift_warns_without_changing_outputs`: raw ring receives fewer bytes than the
processed ring; after `snapshot_pre_trigger`, a warning mentioning drift is logged (caplog), and
`window.wav`, `pre.wav`, `raw_pre.wav` contents plus `record.json` metadata remain correct.

## Verification

- `uvx ruff check .`
- `uvx ruff format --check .`
- `/Users/matt/Desktop/Projects/SaySo/.venv/bin/python -m pytest -q satellite/sayso`
  (all previously passing tests + new drift test pass; the 15 pre-existing environmental errors unchanged)
- `/Users/matt/Desktop/Projects/SaySo/.venv/bin/python -m pytest tests evals/tests --ignore=tests/test_realistic_v3.py -q`
  (same 36 failed / 252 passed baseline as before)

## Commit / push

Commit `fix(satellite): address PR 132 review notes`, push to origin `ajax/sattelite-recording`.
No merge, rebase, force-push, or branch switch. Then edit the PR description (Note 4).
