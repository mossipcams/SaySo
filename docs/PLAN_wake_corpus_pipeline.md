# PLAN: Wake-word corpus pipeline

## Goal

One coherent pipeline: long-form WAV sessions are the canonical source; 2 s
windows are derived artifacts for training and scoring. Reuse the existing
miner, `wake_train.py`, split protections, LiveKit trainer, verifier, and eval.

## Invariants

- Human labels are authoritative. No automatic positive labeling.
- Split at recording-session level before deriving train/eval samples. Never
  randomly split overlapping windows from the same recording.
- Keep selected long-form sessions completely outside training for continuous
  audio eval and FA/hour.
- Preserve trusted living2 data and holdouts unless the new corpus snapshot
  builder explicitly replaces them.
- No WavLM, HyperSpotter, database, new wake architecture, or extra trainer.

## Intended flow

```text
long-form recording
  -> named session ingest
  -> production living2 + verifier replay (16 kHz, 2 s window, 160 ms hop)
  -> candidate events (detections, near-threshold, sparse background)
  -> human labels (positive | negative | unsure)
  -> deterministic dataset snapshot (session-level splits + holdout sessions)
  -> existing LiveKit training
  -> continuous real-audio eval
```

## Scope / files to touch

Prefer extending and collapsing duplication over a new layer.

- `docs/PLAN_wake_corpus_pipeline.md` (this plan)
- `docs/WAKE_TRAINING_DATA_ARCHITECTURE.md` — document the canonical source vs
  derived 2 s windows; session-level splits; holdout sessions
- `docs/WAKE_WORD_DATA.md` — align ingestion/labeling with this pipeline (small
  factual updates only)
- `satellite/sayso/wake/mining.py` — reuse event record shape (session id,
  sample range, context, scored 2 s window, scores, model metadata, label)
- `satellite/sayso/wake/eval.py` — continuous-audio / FA-hour scoring over
  holdout sessions if not already present; do not fork a second eval runner
- `scripts/wake_mine_report.py` — session ingest + production replay mining +
  human labeling; do not auto-label positives
- `scripts/wake_train.py` — deterministic snapshots from labeled events;
  session-level split before window derivation; holdout sessions excluded from
  train; living2 sets remain unless snapshot config replaces them
- Colocated tests: `scripts/test_wake_train.py`, `scripts/test_wake_mine_report.py`,
  `satellite/sayso/wake/test_mining.py`, `satellite/sayso/wake/test_eval.py`
- New small modules only if existing files would become unmaintainable; keep
  them under `satellite/sayso/wake/` or `scripts/`

Do not touch: production wake hook beyond reuse, `ARCHITECTURE.md` (no
ownership change), `evals/` LLM cases, `custom_components/`.

## Behavior

1. **Ingest.** Hours-long WAVs become named recording sessions with stable
   session IDs. Store session metadata and the source audio path; do not treat
   chopped 2 s clips as source of truth.
2. **Replay.** Stream each session through the current production living2 +
   verifier path with `SAMPLE_RATE` / `WINDOW_SAMPLES` / `HOP_SAMPLES` from
   `sayso.wake.livekit` (16 kHz, 2 s, 160 ms hop).
3. **Mine.** Candidate events from production detections, near-threshold
   LiveKit scores, and sparse random low-score/background samples. Each event
   keeps: session ID, source timestamp/sample range, 5–6 s surrounding
   context, exact 2 s scored window, LiveKit score, verifier score,
   model/version metadata, label `positive|negative|unsure`.
4. **Label.** Human `--label` remains the only authority. Unlabeled stays
   unlabeled; unsure is not train-positive.
5. **Snapshot.** Deterministic builder: session-level train/eval/holdout
   assignment, then derive 2 s samples. Overlapping windows from one session
   never appear on both sides of a split.
6. **Train.** Feed snapshots into existing LiveKit `wake_train` / living2
   path. Do not reimplement feature extract/train/export.
7. **Eval.** Holdout long-form sessions run continuous replay; report
   detections and false activations per hour using existing eval helpers.

## Out of scope

WavLM, HyperSpotter, a database, new classifier architecture, extra training
framework, expanding living2 gold sets without the snapshot builder, satellite
runtime mining behavior changes except reuse.

## Verification

- Unit tests for: session ingest, replay hop/window constants, event fields,
  no auto-positive labeling, session-level split (no overlapping-window leak),
  holdout sessions excluded from train snapshot, living2 sets preserved unless
  explicitly replaced.
- `python3 -m pytest scripts/test_wake_train.py scripts/test_wake_mine_report.py satellite/sayso/wake/test_mining.py satellite/sayso/wake/test_eval.py -q`
- Manual: CLI help / dry-run of ingest and snapshot if tests cannot cover I/O.
