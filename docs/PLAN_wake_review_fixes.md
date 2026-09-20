# Plan: fix wake-review defects

Repair the P0–P2 findings from the uncommitted wake-word review. Do not expand
into labeling pilots, cron, or trusted-audio collection.

## Scope

- Inventory `--cleanup` must not delete published `records/<id>/window.wav` or
  the retained copy of a verified duplicate.
- Cluster merge must keep the wakeup `capture_id` (and attach pre-context /
  outcomes to the published record) while still publishing the highest-scoring
  window.
- Production refractory in offline eval must use audio sample time, not wall
  clock, so `--strict` FA counts on long files are real.
- Host acks must drain while the miner is running (no restart).
- Near-miss records must snapshot ring pre-context, not only detections.
- LiveKit staging must not copy the last train clip into the test split.
- FA/hour must count activation events on negative cases, not the per-file
  `detected` boolean.
- `predict_window` without `sample_index` (including `process_pcm`) must still
  apply production cooldown — sample time when an index is present, wall clock
  otherwise.

## Files

- `scripts/wake_mine_report.py` and a colocated check
- `satellite/sayso/wake/{mining,hook,livekit,nanowakeword,eval}.py`
- `satellite/sayso/wake/{test_mining,test_eval}.py`
- `satellite/sayso/launcher.py` only if drain cannot live in the miner
- `scripts/wake_train.py`, `scripts/test_wake_train.py`
- `docs/WAKE_WORD_DATA.md` only if drain/cleanup behavior text would drift

Do not change `ARCHITECTURE.md` unless ownership or trust boundaries change.

## Verification

- Inventory/cleanup on a spool with published records keeps `window.wav`;
  duplicate keeper is the retained path.
- Overlapping detection hops: published `capture_id` equals wakeup/outcome ID;
  highest-score window still published; pre-context survives merge.
- Offline scan of two fires >2 s of audio apart with no wall-clock wait reports
  both activations at refractory 2.0 s.
- After `write_ack`, `drain_acks` (or equivalent) removes the record without
  reconstructing the miner.
- Near-threshold publish has pre-context from the ring when samples exist.
- LiveKit staging: test clip absent from train.
- FA/hour uses `activation_samples` on negative cases.
- Two above-threshold hops without `sample_index` and with refractory 2.0 s
  yield one detection (wall-clock fallback).
- Run colocated mining, eval, livekit/nanowakeword, and wake_train tests plus
  mining self-check.
