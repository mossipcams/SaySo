# SaySo wake-word data strategy

Decision record for how `sayso.onnx` gets its training data. Every future model
should be traceable to the rules here.

## Status: blocked on data collection, part 3 command ready

The intended workflow is:

```
real production audio -> identify false positives -> cluster failure modes
-> compare against training coverage -> select real hard negatives
-> targeted synthetic expansion -> retrain -> evaluate on untouched real audio
```

As of 2026-09-09 this cannot start. **No real wake-word audio exists anywhere.**
A full search of the satellite filesystem, this repository, and its git history
found only generated UI chimes, upstream LVA button sounds, and one 88-second
ambient clip in `/tmp`. `satellite/eval/audio/` contains `.gitkeep` and nothing
else, so all five cases in `satellite/eval/cases.json` — including
`negative_natural_say_so` and `negative_tv_conversation` — skip silently.

The satellite never retained detection audio. Its own log says so:
`Wake phrase detected ... (no audio retained)`. Worse for diagnosis,
`flush_preroll()` hands STT only `[detection_index - wake_skip_ms, end)`, and
`wake_skip_ms` is a 120 ms margin sized to the detection lag, so the
transcript attached to a false positive describes what was said *after*
the trigger. The audio that actually fired the model was never observed, by
anyone, at any point.

Since `2861fa6` this is fixed: setting `wake_word.mine_dir` makes
`HardNegativeMiner` publish the exact 2 s window the classifier scored
(`satellite/sayso/wake/mining.py`). Part 1 (2026-09-20) extends that path to
immutable record directories, bounded ring context, a writer queue, separate
outcome files linked by satellite `capture_id`, and host ingest/ack transfer.
The paragraphs below record the state of the world before mining landed.

This is why every previous attempt to fix false positives had to guess at hard
negatives. The hard negatives in `satellite/models/sayso-training.yaml` are
phonetic inference and are explicitly marked as such.

## What the telemetry does establish

126.8 hours of per-second peak scores from the journal (real production audio,
threshold-independent):

| Percentile | Score |
|---|---|
| p50 | 0.0039 |
| p99 | 0.0131 |
| p99.9 | 0.0311 |
| p99.99 | 0.1926 |
| p99.999 | 0.5618 |

Normal conversation is a tight cluster near 0.004; 99.99% of production audio
sits below 0.19. The model is not broadly confused. The entire false-positive
problem lives in an extreme tail of ~116 windows out of 456,382. **Scaling
dataset volume cannot fix a tail this thin** — the failure is coverage of a rare,
specific acoustic event, which is exactly the quality-over-volume argument.
Identifying it requires the audio.

## Ingestion rules (implemented)

`satellite/sayso/wake/mining.py`, wired through wake providers and
`SaySoExternalWakeHook`.

- Capture is opt-in: `wake_word.mine_dir` unset means no mining.
- Mining is evaluated **before** the detect-threshold and refractory early
  returns, so near-misses and refractory-suppressed windows are captured.
- Sampling classes: detections (`score >= threshold`), near-threshold
  (`mine_threshold <= score < threshold`), and a small independent below-threshold
  sample. Overlapping hop windows are grouped; the highest-scoring triggering
  window is published. Per-class caps and byte/count spool limits stop new
  collection and count drops instead of evicting unacknowledged records.
- The scored `window.wav` is the exact `np.ndarray` handed to the classifier.
  Optional `pre.wav` / `post.wav` come from `WakeCaptureRing` with explicit
  offsets; missing context and synthetic rearm padding are flagged in
  `quality_flags`.
- Each record directory includes `capture_id` (UUID), `session_id`, absolute
  sample range, provider, model SHA-256, score/thresholds, processing settings,
  sampling reason, content hashes, and null labels. `fired` is not proof of
  accepted wake or user intent.
- Wake acceptance/suppression and HA/STT outcomes are separate atomically
  published files under `outcomes/`, linked by `capture_id` (not Home
  Assistant's pipeline trace id). STT command recordings remain a byte-exact tap
  correlated via `wake_capture_id` metadata only.
- Disk writes run on a bounded stdlib queue off capture/inference threads.
  Queue-full and spool-full losses are counted; shutdown drains are bounded.
- Host ingest: `python scripts/wake_mine_report.py <mine_dir> --ingest` verifies
  hashes and writes `acks/`; the satellite deletes only acknowledged ids via
  `drain_acks()` and resumes after drain without restart.

Live config (`/etc/sayso-satellite/config.yaml`): `mine_dir:
/var/lib/sayso-satellite/wake-mining`, `mine_threshold: 0.1`. Review and label
with `scripts/wake_mine_report.py`.

## Train/eval split rule

**Split at the session level, never the window level.** Neighbouring 2-second
windows overlap by ~92% at a 160 ms hop, so a random window-level split puts
near-identical audio on both sides and reports a recall that does not exist.
Partition by recording/session/conversation, then take windows.

The checked-in contract is `satellite/eval/splits.json`. Source groups freeze
before augmentation; overlapping recordings, synthetic variants, and known
session/speaker lineage stay in one split. Holdouts are excluded from training,
calibration, background mixing, and derived features. Unknown lineage does not
establish independence.

Label audit: acoustically identical SaySo / say-so inputs in the 2 s classifier
window are unresolved ambiguity, not separable classes. Retain contextual
evaluation challenges and report ambiguity in strict eval output.

Every failure cluster keeps representatives permanently held out. Evaluation runs
continuous audio through the production sliding-window path, not isolated clips.

## Asset inventory and baseline (part 2)

- Inventory/quarantine: `python scripts/wake_mine_report.py <mine_dir> --inventory`
  scans wake-only roots (mine spool `records/`, `satellite/eval/audio`,
  `satellite/models/data`, `satellite/models/output`). Writes
  `wake_cleanup_manifest.json` (path, sha256, reason). Preserves unknown
  evidence; removes only verified duplicates with a retained copy. Never touches
  `training/` LFM artifacts.
- Strict eval: `python3 satellite/eval/run.py --strict` fails on
  missing/skipped cases, resets model state per recording, uses production
  refractory, and reports activation sample times.
- Baseline freeze: `satellite/eval/baseline.json` records deployed and
  calibration thresholds. Status is **blocked** until trusted eval audio exists.

## Batch training command (part 3)

Host-only dependencies live in `satellite/models/requirements-wake-train.txt`
(`livekit-wakeword[training]==0.2.1`, `faster-whisper`). Do not install them on
the satellite runtime.

```bash
pip install -r satellite/models/requirements-wake-train.txt
python3 scripts/wake_train.py \
  --spool /var/lib/sayso-satellite/wake-mining \
  --seed-dir satellite/models/data/seed \
  --work-dir satellite/models/output/wake_runs \
  --stub
```

The batch command runs ingest → select → snapshot → train → evaluate → save.
Conservative auto-labels admit negatives from the exact 2 s window only; empty
ASR, clipping, homophones, and disagreement stay unknown; trusted seed stays
fixed; no automatic positives. Completed identical inputs are a run-key no-op.
Candidate bundles land under `--work-dir/runs/<run_key>/candidate/` with status
`trained`, `evaluated`, `rejected`, `insufficient_evidence`, or `qualified`.
Scheduling (`--schedule`) refuses while `baseline.json` is blocked or trusted
seed/eval audio is missing.

Colocated checks: `python3 -m pytest scripts/test_wake_train.py -q`

## Deliberately not built yet

Clustering, dedup, representative sampling, and the coverage report are
specified but unimplemented. All four need a real score and similarity
distribution to calibrate against; choosing a similarity threshold before any
clips exist produces a pipeline tuned to nothing. Build them once the spool has a
few hundred labelled clips.

## Resolved: the satellite was dropping and misaligning inference windows

Fixed 2026-09-09. Three compounding defects, all in the inference path:

1. **Redundant embedding passes.** `WakeWordModel.predict()` is stateless: it
   recomputes mel over the full 2 s and runs the speech-embedding model 16 times
   per call, over audio 92% identical to the previous call. Now cached —
   `wake/streaming.py` reuses embeddings across the overlap and computes 2 per
   hop instead of 16.
2. **Window misalignment, which made the cache useless and predates it.**
   `WakeAudioBuffer.feed()` reset its counter to zero on emit, so windows
   advanced by the caller's chunk size rounded up to the hop — 4000 samples in
   production, or 25 mel frames. Not a whole number of stride-8 embedding steps,
   so measured reuse in production was exactly **zero**, and, far worse, the
   wake phrase landed at an arbitrary offset in every window. The buffer now
   emits on a fixed sample grid, so the advance is exactly 160 ms regardless of
   chunk size.
3. **ONNX Runtime threading.** Default settings ran the wake models at 53.6 ms
   p50 while burning 399% CPU; one non-spinning thread runs them at 39.2 ms p50
   for 100%. The graphs are far too small to parallelise. Scoped to the wake
   models via `single_threaded_ort()`.

Measured result on the satellite: p50 **232.8 ms → 48.0 ms** (4.85x), realtime
factor **1.45x → 0.30x**, process CPU **398% → 30%**, load average 8.5 → 2.5.
Scores are bit-identical to stateless `predict()` (max diff 0.000e+00 over 50
windows of real audio) — verified by `scripts/wake_bench.py`, which fails if they
ever diverge.

**Consequence for training: re-measure before retraining.** Every score ever
recorded from this satellite, including the 0.57-0.72 ceiling on genuine wakes
and the 126.8 hours of telemetry above, was produced under defect 2. The phrase
was never reliably centred in the window it was scored on. That ceiling may lift
on its own now. Retraining against numbers gathered from a misaligned pipeline
would bake the wrong operating point into the next model.
