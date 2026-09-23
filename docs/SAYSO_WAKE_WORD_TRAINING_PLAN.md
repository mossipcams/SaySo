# SaySo wake-word training plan

This plan covers the SaySo wake classifier and satellite audio used for its
training and evaluation. LFM language/tool training is separate:
[SaySo model training plan](SAYSO_LFM_TRAINING_PLAN.md).

## Current training direction

The next wake model uses LiveKit's generate-first path in
`satellite/models/sayso.yaml`: setup → generate → augment → train → export →
evaluate, through `scripts/wake_livekit_run.py` in an isolated host work directory.
Real recordings and mined windows supply evaluation and a possible later overlay.
Preserve the shipped living2 bundle and independent holdouts. Operational details
are in [training/wake/README.md](../training/wake/README.md); data ownership,
source splits, and the secondary corpus path are in
[WAKE_TRAINING_DATA_ARCHITECTURE.md](WAKE_TRAINING_DATA_ARCHITECTURE.md).

## Capture and unattended mining plan

The three-part plan below was moved from `SAYSO_LFM_TRAINING_PLAN.md` on 2026-09-22.
Its recorded implementation status and bootstrap blockers were not revalidated
as part of that documentation split. It describes the capture/mining workstream;
the generate-first recipe above remains the next wake-model training path.


**Status:** parts 1–2 implemented; part 3 batch command implemented; labeling pilot and unattended schedule blocked on missing trusted audio.

**Goal:** capture useful satellite audio, clean the wake-training assets, then
run unattended training and evaluation. Save candidate bundles locally;
deployment remains separate.

**Scope:** LiveKit is the only wake provider. Reuse the
LVA audio feed, existing trainer/evaluator, one Python batch command, ordinary
files, SSH/rsync, and one host scheduler. Training and offline transcription
run on the host. Preserve HA ownership, voice-path validation/execution/state
verification, metric scoring, and the existing LFM smoke and locked 120-case
evals. Do not introduce a server, second microphone path, or continuous recorder.
Use colocated checks; do not edit files under `tests/`.

**Before part 1 — read-only prerequisites:**

- Inventory wake-specific roots and jobs on the satellite and training host:
  paths, hashes, sizes, recipes, dependency versions, source data/features,
  metrics, and each run's dependencies. Verify host access/compute and the
  actual deployed provider, model, and threshold. Identify and protect deployed
  and rollback bundles, unique inputs, and pinned datasets.
- Locate trusted training seed, calibration, and independent evaluation audio,
  including real confusions, positive distance/noise cases, and representative
  background recordings. All five local wake fixtures are currently absent;
  remote assets remain uninspected. If trusted data is missing, arrange one-time
  collection and independent labeling. Capture work can proceed, but the
  training experiment cannot start without seed data and a usable baseline.

Then execute **1 → 2 → 3**. Verification gates do not require recurring human
approval; after bootstrap, routine runs need no per-clip review.

### 1. Refactor satellite capture

**Deliverable:** complete, attributable capture records that reach the host
without degrading the voice path.

**Work:**

1. Extend the existing miner on processed 16 kHz mono PCM. Retain the exact
   scored two-second window, absolute sample range, session/capture ID,
   provider/model SHA-256, score/thresholds, processing settings, sampling
   reason, quality flags, and hashes. Keep the capture ID distinct from HA's
   canonical trace ID. Labels start unknown; threshold crossing is not proof
   of an accepted wake or user intent.
2. Collect detections, near-threshold events, and a small independent sample
   below threshold. Group overlapping windows before writing; preserve the
   actual triggering window. Cap events/audio duration per sampling class.
   Use a bounded collection pilot to set quotas from burst behavior, storage
   use, event diversity, and independent sample yield. Low-score captures can
   reveal missed wakes but cannot measure recall without independent labels.
3. Capture bounded context from the existing ring with explicit offsets.
   Snapshot pre-trigger audio before overwrite/rearm; finish post-trigger
   context by a fixed deadline without blocking capture, inference, or HA.
   Mark missing context and synthetic rearm padding; retain the scored window.
4. Write audio and essential metadata into a staging directory, then atomically
   rename it on the same filesystem. Published records are immutable. Store
   actual wake acceptance/suppression and later HA/STT outcomes in separate,
   atomically published records linked by capture ID; ingest them idempotently.
   Missing outcomes never prevent transfer or invalidate audio. Preserve the byte-exact STT tap;
   command recordings are not wake positives.
5. Move writes off capture/inference threads with a bounded stdlib queue.
   Reuse the STT writer's pattern while fixing its uncounted eviction and
   undrained-shutdown behavior in the new path. Count every loss; bound shutdown
   draining and exclude incomplete crash leftovers. Recording failures must
   leave wake detection and the voice pipeline operational.
6. Enforce spool byte/count caps. Transfer finalized records, verify host
   hashes, then delete only acknowledged IDs. At capacity, stop new collection
   and count drops instead of evicting unacknowledged data. Resume after drain
   without restart. Age retention applies to acknowledged local remnants and
   unpinned host-side unknown records; protect pinned datasets and run inputs.

**Files:** `satellite/sayso/wake/{mining,livekit,hook}.py`,
`satellite/sayso/{events,launcher,config}.py`, and
`satellite/sayso/wake/stt_capture.py` for correlation metadata;
`scripts/wake_mine_report.py`, `satellite/README.md`, and colocated checks.
Update `ARCHITECTURE.md` for bounded opt-in audio retention and offline transfer.

**Acceptance:** run the mining self-check and existing capture, provider,
launcher, STT-tap, handoff, and wake-gating checks. Cover sample/byte identity,
late outcomes, burst quotas, queue loss, partial writes, full spool, host outage,
restart/shutdown, and drain/resume. The host must ingest complete records with
traceable provenance and explicit loss accounting. On the satellite, compare
inference cadence/dropped windows with recording on/off and complete a real
wake → HA action → TTS interaction.

### 2. Clean up wake-training assets and establish the baseline

**Deliverable:** one active recipe, protected source data, disjoint datasets,
and a reproducible evaluation of the current model.

**Work:**

1. Use the prerequisite inventory to quarantine corrupt/truncated recordings,
   incomplete records, conflicting labels, and unverifiable provenance.
   Preserve unknown evidence; a poor model score is not a deletion criterion.
   Remove only verified duplicates with a retained copy or reproducible
   intermediates with protected inputs. Record each path, hash, and reason.
   Stop only verified obsolete wake-training jobs after preserving provenance.
   Exclude all LFM artifacts from cleanup.
2. Freeze source groups before augmentation. Overlapping recordings, synthetic
   variants, and known session/speaker lineage stay in one split. Exclude
   holdouts from training, calibration, background mixing, and derived features.
   Unknown lineage does not establish independence. Audit labels against the
   two-second classifier input: spelling or intent inferred from longer
   context cannot make acoustically identical “SaySo”/“say so” inputs separable.
   Retain valid contextual evaluation challenges and report unresolved ambiguity.
3. Recover or create the missing trusted recordings identified in preflight.
   Record provenance and independent background duration. Selected trigger
   clips and score telemetry cannot replace representative evaluation audio.
   Reconcile model README/config drift against the pinned trainer; retain one
   active LiveKit recipe and archive obsolete wake configs/run summaries.
4. Add strict mode to the existing evaluator: empty/missing/skipped required
   cases fail, model state resets per recording, and activation events use
   audio sample time and production cooldown. Report background duration and
   label timing results by their actual hardware. Preserve default developer
   behavior. On the corrected audio path, evaluate the current model at its
   deployed threshold and a threshold chosen only on calibration data. Freeze
   both baselines before training; calibration alone may fix the immediate issue.

**Files:** `satellite/models/{README.md,sayso-training.yaml,sayso-wake-data.yaml}`,
`.gitignore`, `scripts/wake_mine_report.py`,
`satellite/sayso/wake/{eval.py,test_eval.py}`,
`satellite/eval/{cases.json,manifest.json,run.py,test_run.py,README.md}`,
and inventoried wake-only artifact roots. Keep cleanup manifests with the
recordings outside Git.

**Acceptance:** verify protected hashes after cleanup, audio/metadata integrity,
label provenance, and split disjointness including augmentation sources.
Run the pinned trainer's config/data smoke check and strict nonempty baseline
evaluation across required categories, including positive distance/noise cases.
Trusted seed and baseline audio are required for part 3. Missing data is an
explicit bootstrap blocker with a collection requirement; final qualification
may await more background exposure.

### 3. Prove and schedule unattended training

**Deliverable:** one reproducible batch command that demonstrably improves
supported wake behavior, followed by unattended execution with explicit outcomes.

**Work:**

1. **Prove labeling feasibility.** Use a fixed host-side Faster Whisper model
   with timestamps and quality evidence; the existing benchmark's text-only
   wrapper is not a labeler. Against independent labeled examples representative
   of mined audio, measure genuine wakes mislabeled negative, accepted events/day,
   and coverage of actual false-wake categories. Set numerical error/yield gates
   before scoring and report denominators/uncertainty. Freeze the policy and
   validate it on a separate labeled slice.
2. **Constrain automatic labels.** Admit conservative negatives supported by the
   exact two-second input. Empty ASR, clipping, uncertain alignment, homophones,
   and disagreement remain unknown. ASR confidence is not ground truth; longer
   context and HA outcomes may exclude examples but cannot invent an acoustic
   distinction. Record teacher/policy versions, label source, and evidence.
   Keep trusted positive/negative seed fixed and cap weak-negative contribution.
   New automatic positive labels remain out of scope. If labeling errors or
   useful-category yield fail the pilot, leave scheduling disabled and record
   the feasibility failure. Do not add an unvalidated second AI judge.
3. **Build one batch command.** Add `scripts/wake_train.py`. Hash immutable
   snapshots of data, recipe, seed, teacher/policy, frontend, and dependencies.
   Map capture IDs deterministically to trainer filenames/splits and retain the
   mapping. LiveKit 0.2.1 augmentation reads `clip_######.wav`; feature extraction
   reads `clip_######_rN.wav`, so timestamp-named captures need an explicit adapter.
   Preserve an unaugmented feature path for exact real windows; apply synthetic
   augmentation separately. Assert source-to-feature counts and nonempty classes,
   including proof that a known real capture reaches training. Reuse pinned
   training/export APIs, cached seed/features, and host-only dependencies.
4. **Prove a real training run.** Run one bounded experiment. Select candidate
   thresholds only on calibration data; compare with both frozen baselines on
   held-out recordings. Report overall/per-category recall and false activations
   per independent background hour, with uncertainty. Require finite metrics,
   no overall/required-category recall regression, and improved recall or false activations against the
   recalibrated baseline. Freeze comparison rules before training. Do not tune
   on holdout failures or promise improved missed-wake recall from negatives alone.
5. **Verify voice behavior.** Replay the candidate's actual detection sample
   through production preroll/handoff using known command-onset annotations.
   Require first-word retention and no duplicate/omitted command samples.
   Canned transcripts are insufficient. Verify sustained cadence within the
   160 ms hop budget and latency on the target Pi, plus existing handoff/STT checks.
6. **Save and classify results.** Preserve every completed immutable candidate
   bundle: ONNX, selected threshold/provider, hashes, dataset/recipe versions,
   and comparison report. Record `trained`/`evaluated` progress separately from
   `rejected`, `insufficient_evidence`, or `qualified`. Qualification requires
   the quality/timing gates and the false-activation bound below. Reassess a
   saved candidate when new evaluation evidence arrives; do not retrain it merely
   to wait for evidence.
7. **Enable scheduling last.** After the labeling pilot and real experiment show
   useful non-regressing improvement, add one cron entry. Set cadence/new-event
   minimums from measured eligible yield. Use a process lock, compute/time budgets,
   and a run key covering the snapshot/config/environment/seed. Completed
   identical inputs are a no-op; transient failures have bounded retries and
   deterministic failures await changed inputs. Persist reasons for skips,
   failures, and repeated lack of useful data/improvement; mark stalled progress
   explicitly. No per-clip review or per-run approval enters the routine loop.

**Qualification bound:** retain the 0.02 false activations/hour target. Under a
Poisson assumption, zero events over the historical 28.38 hours yields a
one-sided 95% upper bound of about 0.106/hour; about 150 representative independent
zero-event hours are needed for 0.02/hour. Overlap, repeated playback, or selected
mined windows do not create that exposure. Insufficient or correlated evidence
cannot qualify a model, but need not block training or discard its output.

**Files:** new `scripts/wake_train.py` and one colocated runnable check;
`scripts/wake_mine_report.py`, `satellite/sayso/wake/eval.py` and colocated
eval/handoff checks, `satellite/eval/{run.py,test_run.py,README.md}`,
`satellite/models/{README.md,sayso-training.yaml}`, and one host-only requirements
file under `satellite/models/`. Keep satellite runtime dependencies separate.

**Acceptance:** exercise ingest → select → snapshot → train → evaluate → save
with a tiny fixture corpus/stub trainer. Repeated identical runs must neither
duplicate data nor retrain. Inject teacher/label failures, split/background
leakage, omitted real features, failed export, empty eval, worse candidates,
late detections, and interruption; none may qualify an invalid model or change
deployment. Then pass the labeling pilot and real comparison before verifying
consecutive scheduled cycles. A failed feasibility gate leaves part 3 explicitly
incomplete; successful capture/cleanup is not evidence that automation improves
the model.

Implementation references: [Faster Whisper](https://github.com/SYSTRAN/faster-whisper)
for timestamp/quality outputs and the [pinned LiveKit 0.2.1 package](https://pypi.org/project/livekit-wakeword/0.2.1/)
for trainer input/API behavior. Verify against pinned code rather than moving docs.
