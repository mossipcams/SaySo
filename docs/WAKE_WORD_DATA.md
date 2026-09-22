# SaySo wake-word data strategy

Decision record for how `sayso.onnx` gets its training data. Every future model
should be traceable to the rules here.

## Status: locked trainer is LiveKit generate-first

The next classifier is trained from `satellite/models/sayso.yaml` via
`scripts/wake_livekit_run.py` — **never skip generate.** Capture, miner spool,
and long-form corpus are eval / holdout / optional later overlay, not the
primary positive class. The batch snapshot path (`wake_corpus.py` /
`wake_train.py`) remains for labeling and a later mix-in.

The intended eval workflow (once trusted real audio exists) is:

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

Live config: **`satellite/config.yaml`** (Pi deploy:
`/etc/sayso-satellite/config.yaml`). Wake `threshold: 0.25`, `mine_dir:
/var/lib/sayso-satellite/wake-mining`, `mine_threshold: 0.12`, `mic_gain_db:
6.0`, `capture_rate: 16000`. EMEET Pulse sink volume **90%** is via `pactl`,
not yaml. Review and label with `scripts/wake_mine_report.py`.

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

## Primary training command (LiveKit generate-first)

Host-only dependencies live in `satellite/models/requirements-wake-train.txt`
(`livekit-wakeword[train,eval,export]==0.2.1`, `faster-whisper`). Do not
install them on the satellite runtime.

The canonical trainer for the next model is LiveKit's documented pipeline via
`satellite/models/sayso.yaml` and `scripts/wake_livekit_run.py`:

```bash
pip install -r satellite/models/requirements-wake-train.txt
python3 scripts/wake_livekit_run.py run \
  --config satellite/models/sayso.yaml \
  --work-dir /home/ubuntu/sayso-wakeword/runs/livekit-corpus-hn-v1 \
  --hard-negative-features /home/ubuntu/sayso-wakeword/runs/corpus-hn-v1/overlay.npy \
  --hard-negative-provenance /home/ubuntu/sayso-wakeword/runs/corpus-hn-v1/overlay.provenance.json
```

Stages: `setup` → `generate` (Piper TTS + adversarial negatives + backgrounds)
→ `augment` → **verified hard-negative overlay** → `train` → `export` → `eval`.
**Do not skip generate.** Target phrases are SaySo/Sayso spelling variants
only; `"say so"` homophones are `custom_negative_phrases`, never targets. Wavs
stay off git; use an isolated `--work-dir` (never `output-living2/`).

### Next retrain (2026-09-22): overlay corpus false-positive embeddings

Every activation on the replayed corpora is a false positive (no
device-addressed SaySo). Production frontend, deployed `5c5c3187`, threshold
0.42, +10 dB:

| Corpus | Hours | Clustered FPs | /hour | Max |
| --- | ---: | ---: | ---: | ---: |
| AMI far-field Mix Array1-01 | 8.67 | 42 | 4.84 | 0.596 |
| VOiCES `mc05-stu-far` + `tele` | 14.15 | 8 | 0.57 | 0.495 |
| Live Snowball TV/room 2026-09-21/22 | minutes | 13 labelled | — | 0.529 |

ACAV-only overlay (`d57c11c2`, top-5000) cut AMI to 2.08 /hour. ACAV's mined
tail topped out at 0.338, so random/hard ACAV does not cover far-TV. **Do not**
put AMI, VOiCES, or TV wavs into generate / `negative_train` / 
`custom_negative_phrases` — that memorizes those recordings. Overlay the
**(16, 96) embeddings** of hard production windows only.

Method:

1. Split at recording level. Overlay train AMI meetings and VOiCES rooms;
   hold out at least one AMI block and one VOiCES room (prefer `rm4`, which
   carried long-track FPs). Labelled Snowball FPs are overlay-only, not
   generate clips.
2. Mine embeddings from production 2 s / 160 ms windows with score ≥ 0.25
   (near-misses, not only 0.42 fires), diversity gap, provenance (path,
   `start_sample`, score, model sha256). Cap ~5000 corpus rows. Concat with
   ACAV top-5000 mined by `5c5c3187`.
3. `sayso.yaml` stays ACAV 2048 / `target_fp_per_hour` 0.05 /
   `max_negative_weight` 5000. Isolated work-dir
   `/home/ubuntu/sayso-wakeword/runs/livekit-corpus-hn-v1`. Do not empty
   `CUDA_VISIBLE_DEVICES`. Do not stop LFM2.
4. Qualify on stock LiveKit val, held-out AMI, held-out VOiCES room,
   `room-fp-challenge-20260922`, and `negative_tv_live_20260921`. Compare at
   0.42 and at matched recall vs `5c5c3187`.

Shipped 2026-09-22: Pi `/opt/sayso-satellite/models/sayso.onnx` is
`livekit-corpus-hn-v1` (`0a3260c8`). Live detect threshold **0.25** and
`mine_threshold` **0.12** are in `satellite/config.yaml` (qualified at **0.42**
on holdout). `5c5c3187` remains `sayso.onnx.bak-5c5c3187`. Clean holdout
(AMI ES2005* + VOiCES rm4, 5.43 h) was 0 clustered FPs vs 5 on `5c5c3187` and
2 on `d57c11c2`.

Real Snowball / miner wavs stay eval and optional embedding overlay — not the
primary positive class. Current Pi model is `livekit-corpus-hn-v1`
(`0a3260c8`) with live threshold **0.25** per `satellite/config.yaml`.

## Batch snapshot command (part 3 — secondary)

Do not install training deps on the satellite runtime.

Corpus workflow:

```bash
python3 scripts/wake_corpus.py ingest room.wav --corpus /path/to/wake-corpus --session-id room_a
python3 scripts/wake_corpus.py replay room_a --corpus /path/to/wake-corpus
python3 scripts/wake_mine_report.py /path/to/wake-corpus --unreviewed --play
python3 scripts/wake_mine_report.py /path/to/wake-corpus --label <event_id> negative
python3 scripts/wake_corpus.py split --corpus /path/to/wake-corpus --holdout room_a
python3 scripts/wake_corpus.py snapshot --corpus /path/to/wake-corpus
```

Training command:

```bash
pip install -r satellite/models/requirements-wake-train.txt
python3 scripts/wake_train.py \
  --spool /var/lib/sayso-satellite/wake-mining \
  --corpus /path/to/wake-corpus \
  --seed-dir satellite/models/data/seed \
  --work-dir satellite/models/output/wake_runs \
  --stub
```

The batch command runs ingest → select → snapshot → train → evaluate → save.
Conservative auto-labels admit negatives from the exact 2 s window only; empty
ASR, clipping, homophones, and disagreement stay unknown; trusted seed stays
fixed; no automatic positives. Completed identical inputs are a run-key no-op.
Candidate bundles land under `--work-dir/runs/<run_key>/candidate/` with status
`evaluated`, `rejected`, `insufficient_evidence`, or `qualified`.
Scheduling (`--schedule`) refuses while `baseline.json` is blocked or trusted
seed/eval audio is missing.

Colocated checks:

```bash
python3 -m pytest scripts/test_wake_livekit_run.py scripts/test_wake_train.py -q
```

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
