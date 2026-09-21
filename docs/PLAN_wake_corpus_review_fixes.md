# PLAN: Wake corpus review fixes

## Goal

Fix the defect-first review of the wake corpus pipeline, then update
`docs/WAKE_TRAINING_DATA_ARCHITECTURE.md` and
`docs/SATELLITE_DATA_COLLECTION.md` so they match the real flow.

## Defects to fix

1. **Holdout flags after first split file.** `ensure_session_splits` must honor
   new `--holdout` / `--corpus-holdout` IDs (reassign those sessions to
   holdout, or error if the file cannot be updated). Silent ignore is a leak.
2. **Holdout FA/hour sample index.** `evaluate_holdout_session` must use the
   same hop/lag path as `replay_session` (`window_end - pending_lag`), not
   `scan_wake_audio`'s `window_end`.
3. **Re-replay duplicates.** Namespace or replace `.replay_spool` per session;
   do not import a second set of UUIDs for the same session without replacing
   unlabeled prior events from that session.
4. **Dead `eval` split.** Derive eval samples from eval sessions in the
   snapshot (session-level split before window derivation). Do not train on
   them. Do not leave eval-session labels unused.
5. **FA/hour wiring + true wakes.** Add a corpus CLI (or wake_train hook) that
   runs holdout continuous eval. Do not count human-labeled positive windows
   as false activations.
6. **Duplicate CLIs.** Keep labeling on `wake_mine_report.py`; ingest/replay
   should call shared helpers (or `wake_corpus`) instead of a second copy of
   provider/spool/import logic.

## Docs

- `docs/WAKE_TRAINING_DATA_ARCHITECTURE.md`: canonical source vs 2 s windows;
  session-level train/eval/holdout; re-replay; FA/hour rules; living2
  `--replace-living2`; CLI commands that actually exist.
- `docs/SATELLITE_DATA_COLLECTION.md`: how Pi mining spool relates to host
  session ingest/replay/labels; do not claim satellite long-form ingest if it
  still only mines 2 s windows live.

## Files

Same wake corpus surface: `satellite/sayso/wake/{corpus,replay,sessions,snapshot,eval}.py`,
tests, `scripts/wake_corpus.py`, `scripts/wake_mine_report.py`,
`scripts/wake_train.py`, the two architecture md files, this plan.

## Verification

```text
python3 -m pytest scripts/test_wake_train.py scripts/test_wake_mine_report.py scripts/test_wake_corpus.py satellite/sayso/wake/test_mining.py satellite/sayso/wake/test_eval.py satellite/sayso/wake/test_sessions.py satellite/sayso/wake/test_replay.py satellite/sayso/wake/test_corpus.py satellite/sayso/wake/test_snapshot.py -q
```

Tests must cover: holdout flag updates existing splits; holdout eval sample
index matches replay lag; second replay does not duplicate session events;
eval-session examples appear with split=eval and not in train; labeled
positives excluded from FA/hour.
