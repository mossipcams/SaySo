# SaySo training plan

**Scope:** the stable design and constraints for `LFM2.5-230M-Base` supervised
fine-tuning, plus the three-part wake-word capture and training plan below.
Rules that outlive any one run belong here. For LFM, what ran, what it
scored, and which checkpoint is promoted belong in
[training/TRAINING_LOG.md](../training/TRAINING_LOG.md); commands and active
config belong in [training/README.md](../training/README.md).

SaySo trains `LiquidAI/LFM2.5-230M-Base` to select and emit Home Assistant tools in
the same OpenAI-compatible shape used by the integration. Home Assistant still
owns the entities, schemas, permissions, execution, and final voice pipeline.
llama.cpp only hosts inference.

Do not revive Instruct-checkpoint, FunctionGemma, or Axolotl full-SFT
plans. Checked-in `training/configs/lfm25-230m*.yml` and `functiongemma-270m*.yml`
Axolotl files are leftovers, not the training path.

## 1. Training contract

Use `schemas/sayso-tool-schema-v2.json` as the pinned training contract. It is a
snapshot of the Home Assistant Assist LLM API and defines the tool names,
descriptions, parameters, required fields, and constraints used to validate
examples. The flat `tools` list is the training source of truth; the same
artifact also groups tools by device-type tier (`query`, `generic`, `light`,
`fan`, `climate`, `media_player`, `vacuum`, `timer`) for catalog validation.
`schemas/sayso-tool-schema-v1.json` remains as a flat-only historical artifact
with an identical tool list and fingerprint.

Every expected tool call must pass the pinned schema before it enters a
dataset. When the Home Assistant contract changes, update the pinned artifact
and regenerate data. Do not hand-maintain a second tool catalog.

`ALLOWED_HASS_TOOLS` is a dataset-build gate for the pinned contract. It is not
the runtime capability boundary: runtime support follows the tools Home
Assistant supplies for that request.

## 2. Training examples

Store each example as structured `messages` plus OpenAI-style `tools`:

- tool entries use `{"type":"function","function":...}`;
- assistant tool calls contain the validated name and arguments;
- tool results represent the Home Assistant response before a final spoken
  answer when a follow-up is needed;
- `train_on_turn` is true only for assistant tool-call and final-response
  messages.

Labels are deterministic and schema-validated. Keep the expected call
authoritative; paraphrasing must never change the tool, arguments, or
call/no-call decision.

Do not train on:

- ChatML `<tool_call>` labels;
- eval case IDs or examples from `evals/cases/`;
- variants of the 38 recipe-lock golden utterances;
- unsupported tools or arguments;
- model-generated labels that have not passed SaySo schema validation.

Synthetic generation in `training/generators/pipeline.py` owns
utterance diversity; schema validation remains authoritative for every label.

Quotas are accounted on the supervision a row carries, not on its metadata.
`training/generators/coverage.py` classifies each accepted row by outcome
(action, state query, clarification, absence, unsupported), tool, domain and
targeting mode. A row satisfies an operation's positive quota only when it calls
that operation's tool on that capability's domain against a real target;
refusals fill a separate negative allowance, and offering a tool in the schema is
never coverage. Every retained supported tool and valid domain/action combination
must have positive rows, or generation fails. Tools deliberately outside the
declared coverage (`TRAINING_COVERAGE_EXCLUDED`) are also withheld from distractor
sampling, which is a dataset-scope decision and says nothing about which tools
Home Assistant supplies at runtime.

Generate actions only where the contract and the entity permit them. Operations
declare the entity features they need and Home Assistant's `supported_features`
supplies them, so a media player without power control never receives a
`HassTurnOn` label.

`training/generators/grounding.py` holds paired scenarios in which the request is
fixed and the entity graph moves — renamed, relocated, aliased, re-domained, or
differently capable targets, with individual and area targeting, genuine
ambiguity and presence/absence pairs. Expected behavior is derived from the
supplied graph and the current tool contract before any OHF wording or
paraphrasing is applied.

Both synthetic rendering paths share `training/generators/utterances.py`.
English phrasing uses a pinned OHF-Voice/intents subset through Hassil, with
literal home names and aliases bound to grammar slots. The adapter must preserve
all supplied settings, scopes and calls; unmatched combinations use an explicit
semantic fallback. Generation filters incomplete recognition fragments before
adding conversational framing. Numeric validation accepts digits and the number
words emitted by the existing STT transform, and rejects missing settings.
See `training/generators/OHF_SOURCE.md` for provenance and reproduction.

Corpus review must report actual upstream grammar sources, fallback usage,
multi-call/exclusion/alias coverage and label consistency, in addition to unique
request counts. Never regenerate a running job's pinned input in place.

## 3. Format and trainer

Keep two representations separate:

- The canonical SaySo dataset keeps OpenAI-compatible `function.arguments` as
  validated JSON strings.
- The TRL view may parse those strings into native objects only while applying
  the LFM2.5-Base `chat_template.jinja`. Do not put ChatML in the dataset or in
  the Home Assistant integration.

```text
canonical JSONL (OpenAI messages, string arguments)
        ↓ tokenizer.apply_chat_template + tools= (dict arguments)
ChatML + <|tool_call_start|>[HassTurnOn(name='…')]
        ↓ rsLoRA SFT on Base (TRL)
merged FP16 adapter
        ↓ llama.cpp convert + Q8_0
GGUF served with --jinja
        ↓ SaySo HTTP still sends OpenAI messages, not ChatML
```

Train from Base with TRL rsLoRA, not by continuing a previous merged checkpoint:

- rank 32, alpha 32, `use_rslora: true`, `all-linear`, FP16, no BF16, no Flash Attention
- microbatch 1, gradient accumulation 16
- learning rate `2e-4`, cosine, assistant-only loss
- For early-feedback runs, save every 250 optimizer steps; retain the first
  checkpoint and both epoch checkpoints needed for comparison.

Smoke on the longest rendered training row before a full run. Require finite
losses, a finite gradient at a positive learning rate, finite adapter tensors,
and nonzero LoRA B weights to demonstrate an actual optimizer update. A saved
adapter after a skipped FP16 step is insufficient. The full run still starts
from Base, never from the smoke adapter. Pascal GTX 1070 (8 GiB) may log an
allocator warning and still complete.

## 4. Data and eval

Locked gold, shadow, grounding, and the original 120 realistic cases live in
`evals/cases/`. Smoke (~24 promotion cases) is checkpoint selection;
promotion is the locked 120. Do not train on those utterances.
`evals.cases.excluded_train_utterances()` is the holdout set; the v3 generator
rejects any train utterance in that set (`quality_eval_overlap`), including
normalized prompts, instead of filtering contaminated rows out afterwards.
Generation is reproducible — the same seed yields a byte-identical dataset
across processes — so never seed generator randomness with builtin `hash()` on
a string.

Train each run from Base, not by continuing a previously merged checkpoint.
Evaluate the first 250-step checkpoint on the frozen suites while training
continues, using the same tokenizer/serving configuration as Base. Keep both
epoch evaluations; an early result is diagnostic and does not trigger automatic
promotion or stopping.

A run trains on one deterministic corpus. Do not blend corpora to make a set
larger: read the gold and shadow results first, then refine the cases the run
actually gets wrong and regenerate. Adding data before that evidence exists
hides which cases are weak. Corrective rows, when a refinement pass calls for
them, must use fresh homes, entities, and wording.

Shadow eval is 100–150 cases covering the same concepts as its gold set, with
different entities and phrasing. Promote only when gold and shadow both move the
right way. If only gold improves, the run is overfitting the benchmark.

Score generations with the apostrophe-safe production parser in
`custom_components/sayso/lfm_parse.py` (raw `/completion` text). llama.cpp
structured `tool_calls` truncates names such as `O'Malley's` and `Kids'`; that is
a serving bug, not a training label. Do not retrain to paper over it, and never
compare a score from one scorer against a score from another — record which
scorer produced each result.

`training/scripts/generate_balanced_test_data.py` builds the 2,500-example
held-out set. Do not train on those prompts.

`evals/cases/regressions.jsonl` (tag `grounding`) holds the entity-grounding
regressions, including Living Room + `media_player.living_room_tv` named "TV" →
`intent__HassTurnOn(name="TV", domain=["media_player"])` in the production prompt,
context and tool-schema format, plus held-out variations with different names,
ids, areas, distractors and presence/absence conditions. Check targets and
arguments, not tool selection alone. Its prompts belong to
`excluded_train_utterances()`; neither they nor near-duplicate scenario variants may
be trained on.

Home Assistant is authoritative for which entities exist, which are exposed to
Assist, and what each can do. `fetch_ha_home.py` records `exposure_source` on
every snapshot, and only `assist_exposure` is a valid input for the home-specific
recipe. That recipe enables real-home mixing at an explicit nonzero rate;
`--synthetic-only` is the deliberate override. Entity-frequency caps and the
per-capability holdout still apply, and the manifest records requested versus
achieved mixing, real-home rows (rows, never targets), per-entity target counts,
positive coverage and the negative distribution.

Which checkpoint is currently promoted, what each run scored, and which corpus
it used belong in [training/TRAINING_LOG.md](../training/TRAINING_LOG.md), not
here.

Promote a checkpoint only when it improves target behavior without regressing
STT, status, no-call, multi-action, light/fan, or lock polarity. Then export
GGUF and verify with llama.cpp `--jinja`. Freeze a promoted champion and do
not overwrite its GGUF, merged weights, or epoch checkpoint.

Promotion, deployment, and shipping are three separate decisions. A model may be
deployed without being promoted, and shipped without either — Run 013 step-2500
is all three states at once. Record which one applies; never infer promotion
from the fact that a model is serving or published. Ship with
`scripts/publish_model.sh`, which pins the artifact by sha256.

## 5. Runtime safety boundary

Training does not replace runtime controls. The SaySo integration must continue
to:

1. compile the tools supplied by Home Assistant;
2. treat model output as untrusted;
3. validate every call and every argument before execution;
4. validate every call in a batch before executing any call;
5. fail closed on ambiguity, unsupported tools, malformed output, and schema
   mismatch;
6. allow only the existing bounded correction path before execution;
7. execute through Home Assistant and use its result for the spoken response;
8. never retry an already executed action because of a later invalid response.

## 6. Non-goals

- model bake-offs (Instruct LFM, FunctionGemma, Alexa+)
- Axolotl full-parameter SFT
- fine-tuning on ChatML `<tool_call>` labels
- long autonomous chains
- replacing Home Assistant validation with model trust
- a SaySo server, broker, custom action protocol, or direct satellite-to-model
  connection

## Generator corrections vs locked eval categories

The canonical generator (`training/generators/`, recipe YAML under
`training/configs/generation/`) targets these promotion-suite failure classes.
This is a label/coverage intent map only — not a model accuracy claim.

| Generator correction | Intended eval categories |
|---|---|
| Status rows label `GetLiveContext`, not state-changing intents | `status` |
| Genuine ambiguity → clarify; context-resolved names → action | `ambiguity`, `aliases` |
| Exclusion rows call all non-forbidden targets | `exclusion` |
| Withheld-capability rows omit the tool from the offered catalog | `unavailable` |
| Distinct absence / unsupported / clarify families | `unavailable`, `ordinary` (refusal control) |
| `production_catalog(home)` for every row (no answer-first subsets) | `ordinary`, `climate`, `routines_vacuum`, `light_fan_settings`, `multi_action` |
| Area scenarios as a sibling family (subtracted from count, tagged `family="area"`) | area grounding in `ordinary`, `multi_action`, `exclusion` |
| Honest `GetDateTime` in catalog when exposed | general time queries (no dedicated promotion tag) |
| Contrast groups preserved in planning/split metadata | `status`, `ambiguity`, `unavailable`, `exclusion`, `multi_action` |

Allocation shares for accepted rows live in
`training/configs/generation/production.yaml` (see `docs/PLAN_GENERATOR_REFACTOR.md`).

### Family allocations and area rows

`GeneratorConfig.allocations` drives `FamilyTracker` in `planning.py`. Recipe-backed
runs (`recipe_path` set from `from_yaml`) always use this planner and never enter
`QuotaTracker`. Programmatic family runs pass `allocations=default_allocations()`
or explicit shares. Bare `GeneratorConfig()` with empty allocations still uses the
legacy tier/capability/operation `QuotaTracker` in `sampling.py` for colocated unit
tests only — not a production path.

Area scenarios are a **sibling family**, not a cross-cut inside primary families.
When `cross_cutting.area` is enabled, area minimums are subtracted from
`count`, the remaining slots are drawn from primary-family allocations, and each
accepted area row is tagged `family="area"`. An area row is not also counted as
`ordinary`, `status`, or `exclusion`. Production “ordinary 40%” is 40% of
`(count − area_total)`, not 40% of the full corpus.

### Cross-cutting rates (grounding and discrimination)

`grounding.rate` and `discrimination.rate` in the recipe are shares of accepted
rows. `enforce_rate_gate` in `rates.py` compares achieved share against the
reachable ceiling (`grounding_available_share` /
`discrimination_available_share`), not the raw recipe value when capacity is lower.

Production v3 sets `grounding.rate: 0.0` deliberately: promotion failures were
clarify/status/exclusion/unavailable, not grounding delivery. Re-enable only when
the catalogue and gate can support a nonzero share.

`discrimination.rate` is capped at the measured delivery ceiling
(`DISCRIMINATION_DELIVERY_CEILING = 0.02` in `scenarios/discrimination.py`).
The production recipe requests `0.02` so the gate does not silently accept a 4%
request against a 2% ceiling.

## Implementation map

| Concern | Location |
|---|---|
| Canonical generator CLI and recipes | `python -m generators.cli --config training/configs/generation/{production,smoke}.yaml`, `training/generators/{cli,config,pipeline,planning,rendering,validation,manifest}.py`, `training/configs/generation/` |
| Dataset build entry points | `training/generators/cli.py`, `training/scripts/generate_balanced_test_data.py` |
| Scenario facts and area families | `training/generators/scenarios/`, `training/generators/grounding.py` |
| Coverage accounting and audit | `training/generators/coverage.py`, `training/generators/planning.py`, `training/generators/audit.py` |
| Entity-grounding families | `training/generators/grounding.py`, `evals/cases/regressions.jsonl` |
| Real-home export and mixing | `training/scripts/fetch_ha_home.py`, `training/scripts/ha_websocket.py`, `training/generators/real_home.py` |
| Recipe-lock / quality / grounding eval | `evals/cases/regressions.jsonl` |
| Promotion and smoke | `evals/cases/realistic_v3.jsonl`, `evals/suites/` |
| LFM Python parse | `custom_components/sayso/lfm_parse.py` |
| LFM adapter | `training/adapters/lfm.py` |
| Schema validation | `training/adapters/schema.py` |
| TRL recipe (checked-in) | `training/configs/lfm25-230m-synthetic-v3-40k-trl.yml` |
| Evaluation | `evals/` |
| Pinned contract | `schemas/sayso-tool-schema-v2.json` (§1; v1 is a historical artifact) |
| Run history and scores | `training/TRAINING_LOG.md` |

Operational commands belong in `training/README.md`. Update this document only
when the training design or its safety boundary changes.

## Wake-word improvement: three-part implementation plan

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
