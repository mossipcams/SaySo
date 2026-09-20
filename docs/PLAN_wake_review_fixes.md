# Plan: PR #86 review fixes

Fix the P1–P2 findings from the PR #86 review. Do not expand into labeling
pilots, cron, or trusted-audio collection.

## Scope

- `--inventory --cleanup` must never delete or quarantine pinned eval/seed
  paths (`satellite/eval/audio`, `satellite/models/data`). Duplicate removal
  may only drop extra spool copies.
- `wake_train` candidate eval must use production refractory and strict
  required cases (stub fixtures already provide audio).
- Eval hardware must not label Apple Silicon as `pi`.
- Missing satellite config must not silently eval at threshold 0.65; use the
  deployed baseline default (0.5) or fail closed with an explicit note.
- Accepted-wake outcome files must publish even when the STT tap is off.
- `wake_train` lock must be exclusive (`O_CREAT|O_EXCL`), not exists-then-write.
- `WAKE_WORD_DATA.md` candidate statuses must match `classify_candidate`.

## Files

- `scripts/wake_mine_report.py`, `scripts/test_wake_mine_report.py`
- `scripts/wake_train.py`, `scripts/test_wake_train.py`
- `satellite/eval/run.py`, `satellite/eval/test_run.py`
- `satellite/sayso/events.py` and colocated event tests
- `docs/WAKE_WORD_DATA.md`

Do not change `ARCHITECTURE.md`.

## Verification

- Cleanup: spool WAV matching an eval fixture keeps the eval file; spool extra
  is the copy removed.
- wake_train eval path uses strict + production refractory (or equivalent
  production cooldown).
- ARM Darwin is not reported as `pi`; Linux ARM may still be.
- Config-load failure does not use threshold 0.65.
- Accepted wake outcome is written with miner on and `stt_capture` None.
- Two overlapping `acquire_lock` calls: the second fails.
- Colocated tests for the above plus existing mining/eval/wake_train suites.
