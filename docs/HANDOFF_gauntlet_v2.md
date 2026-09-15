# Handoff: Gauntlet v2 corpus (grounding fixed; still fails the pre-flight gate)

Status: **Do not start a v2 training run yet** — not because the generator is
broken, but because the corpus still measures 91.5% name-echoing. See
"Remaining gap".
Written: 2026-09-13. Updated: 2026-09-13 (grounding scaling fixed).
Worktree: `SaySo__worktrees/ajax-new-data-generation`, branch `ajax/new-data-generation`
Base commit: `8ef7be3 docs(training): record 40k grounded gauntlet v4 launch`

## TL;DR

Run 013 (the 40k grounded gauntlet v1 corpus) trained fine but did not teach
entity grounding. **The grounding-rate collapse is now fixed and verified flat
from 2k to 40k rows** (2.65% / 2.58% / 2.54%); a full 40k Gauntlet v2 corpus
builds with every gate on. The earlier diagnosis in this file — a quota-planner
starvation bug — was **wrong**: the ceiling was the 15-scenario grounding
catalogue times the duplicate limit (8), i.e. 120 rows at any corpus size.

What is still true: the corpus does not yet teach entity resolution. The new
pre-flight probe measures **91.5% of target rows naming the target verbatim**
(v1: 96.3%) against a <60% contract, because the discrimination category still
delivers ~1.5%. That is the remaining work.

## 1. Run 013 post-mortem (the reason for this work)

Run 013 = `SaySo-LFM2.5-230M-40k-grounded-gauntlet`, 40k rows, 2 epochs, 5000
steps, finished 2026-09-13 06:12 UTC. Evaluated at steps 2500 / 3250 / 5000.

| Suite | step-2500 | step-3250 | step-5000 | vs live champion |
|---|---|---|---|---|
| `recipe_lock` | 29/38 | 28/38 | 28/38 | tie (29/38) |
| `v3_gold` | 25/35 | 22/35 | 22/35 | +1 (24/35) |
| `v3_shadow` | 71/100 | 69/100 | 70/100 | -1 (72/100) |
| `realistic_20260908_v2` | 58/120 | 58/120 | 58/120 | +2 (56/120) |
| **Total** | **183/293** | 177/293 | 178/293 | 181/293 |

Verdict: **statistical tie with the model it replaced.** Epoch 2 added nothing
(2500 → 5000 moved -5 cases). Not promoted.

Failure analysis of `realistic_20260908_v2` (at step 2500, 58/120):

- `args_mismatch` 23, `missing_tool_call` 22, `unexpected_tool_call` 9,
  `tool_count_mismatch` 5, `tool_name_mismatch` 3.
- The model emits the **right tool and domain but the wrong entity**:
  `Morgan's Bedside Lamp` → `Bedroom Roller Blind`; `Sofa TV` → `Bedroom
  Curtains`; `Lounge Television` → `Laundry Television`.

### Hypotheses tested and rejected (all disproven)

| Hypothesis | Result |
|---|---|
| Eval scorer confused by aliases | rejected — 0/23 `args_mismatch` are alias/canonical mixups |
| Corpus lacks same-domain distractors | rejected — 96% of target rows have ≥2 same-domain competitors |
| Corpus inconsistent on ambiguous targets | rejected — 2/173 ambiguous utterances map to >1 target, both case variants |
| Held-out leakage | rejected — overlap audit 0/0/0/0 |
| Missing/champion baseline confusion | resolved — `CHAMPION.txt` exists in two places; `/srv/models/CHAMPION.txt` is current, `/srv/training-runs/CHAMPION.txt` is stale |

### Actual root cause

Full-corpus measurement (all 40,000 rows):

```
rows with a listed target      : 27,596
  utterance names entity verbatim : 26,583  (96.3%)
  utterance describes entity      :  1,013  ( 3.7%)
distinct utterances            : 28,399
duplicate utterances           : 11,601
```

**96.3% of training rows name the target entity verbatim in the user
utterance.** The corpus teaches "echo the name you were given"; the eval demands
"resolve a description to the right entity." That skill was never trained.

Manifest corroborates: `by_targeting: individual=32614 (82%)`, `area=1320 (3%)`.

## 2. Generator defects found and FIXED (local tree, 348 tests pass)

All verified by re-introducing the v1 behaviour and watching the new tests fail.

1. **`grounding_required` was a one-shot latch.**
   `pipeline.py`: `grounding_required=config.grounding_rate > 0 and not
   grounding_rows` — falsy the moment one grounding row lands, so forcing
   stopped immediately. Replaced with deficit-driven demand.

2. **Grounding slots were drawn at random.** `pick_variant` only returns a
   variant for 4 (capability, operation) pairs, so only ~4.3% of slots could
   ever host one. Added `_grounding_capable_slot` to select from capable pairs.

3. **Real-home branch preempted grounding.** Reordered so a real-home row wins
   only when it was actually selected (real-home mixing needs
   `media_players/turn_on`, which is also a grounding pair).

4. **`similar_name` robustness was dead code** — selected by
   `pick_robustness`, never handled in `scenarios.py`. Left inert; noted only.

5. **No gate asserted `achieved_rate`.** Added `_enforce_rate_gate` so a
   shortfall fails the build instead of shipping silently. v1 requested 3% and
   shipped 0.22% "successfully".

6. **Added entity-discrimination rows** (the category v1 lacked entirely):
   - `utterances.py`: `describe_target()` renders descriptions from tokens that
     **already exist in entity names** (`Ceiling`, `Pendant`, `Wall`, `Vanity`,
     `Nanoleaf`, `Lutron`). Nothing is invented — see the rejected approach below.
   - `pipeline.py`: `_entity_descriptors`, `_sibling_entities`,
     `_uniqueness_safe`, `_pick_discriminating_description`.
   - Uniqueness is structural: requires ≥1 same-domain same-area sibling; refuses
     any description containing the target's or a sibling's name/alias; refuses
     when a modifier also appears in a sibling name. A row that cannot be made
     unique is dropped, not shipped ambiguous.

7. **`validate.py`**: `missing_expected_target` now skipped for rows flagged
   `spec["discrimination"]`, since those legitimately do not name the target.

### Rejected approach (do not retry)

First attempt generated descriptions from invented positional phrases ("on the
left", "in the corner", "near the sofa"). **Entities carry no positional
attributes** — `home["entities"]` has `name`, `aliases`, `domain`, `kind`,
`device_class`, `area`, `floor`, `state`, `capabilities` and nothing else. Those
descriptions were fabricated and would have trained noise. Rewritten to use only
name-derived tokens.

## 3. FIXED: grounding rate collapsed with corpus size

### The diagnosis in the first draft of this file was wrong

It read as a quota-planner starvation bug — grounding-capable buckets are a fixed
~7.3% slice of the plan, ordinary rows drain them, `_grounding_capable_slot`
returns `None` 23% of the time at n=12,000. All of that is **measurably true and
is not the cause**. Five run-time scheduling fixes were attempted against it and
all five were reverted, which was the signal that the model was wrong.

Budgeting grounding capacity in `build_quota_targets` / `build_quota_plan` would
**not** have fixed it either. Do not spend time there.

### Actual root cause: an absolute ceiling, not a rate

Every grounding variant is one fixed scenario (fixed home, fixed request).
`DuplicateTracker` accepts any one scenario at most `near_duplicate_limit` (8)
times. The v1 catalogue held 15 scenarios:

```
15 scenarios x 8 rows = 120 grounding rows, at ANY corpus size
```

That is the whole defect. 49 rows at 2k reads as 2.4% and 136 rows at 40k reads
as 0.34% — the same constant, divided by n. Confirmed directly: at n=12,000, 936
of 1,399 forced grounding attempts died on `duplicate_semantic_id`, and the
per-family counts were pinned at exactly 8.

### The fix

`grounding.site_variants(index, area, elsewhere)` instantiates the entire
contrast set at one more area with fresh device names, and `_sites()` supplies 30
areas drawn from `homes._AREAS` (excluding every area the canonical set or the
eval set already speaks aloud). Catalogue: 15 -> **345 scenarios**, ceiling 2,760
rows. The canonical site is unchanged, so `required_training_variants()` still
returns the same 11 families and the "every contrast family must land" gate is
untouched.

Measured after, `grounding_rate=0.028`:

| n | achieved grounding | before |
|---|---|---|
| 2,000 | 2.65% | 2.35% |
| 12,000 | 2.58% | 1.12% |
| 40,000 | **2.54%** (1,016 rows) | 0.34% (136 rows) |

Two guards so this cannot ship silently again:

- `pipeline._grounding_capacity` — `run_generation` raises **before** generating
  when `count * grounding_rate` exceeds `len(training_variants()) *
  near_duplicate_limit`, naming the fix ("add sites in `grounding._sites`").
- `test_grounding_catalogue_scales_to_a_full_size_corpus` — asserts the same
  invariant in milliseconds, in total and per (capability, operation) pair.
  Verified to fail against the 15-variant catalogue.

`pipeline._GROUNDING_EFFECTIVE_CEILING` (the hardcoded 0.028) is **deleted**. The
rate gate's reachable ceiling is now derived: `min(capable-slot share, catalogue
capacity / count)`.

## 3b. Two further defects, found while validating the fix above

Validating the expanded catalogue meant checking all 345 scenarios against their
family's contrast contract, not just that they build. Both of these were found
that way, and both are now covered by
`test_every_generated_site_keeps_its_family_contrast`.

1. **The alias family targeted its own distractor (~42% of alias rows).**
   Pre-existing, not introduced by the expansion — the expansion multiplied it by
   30. `alias_family` lists the aliased entity first and a "Ceiling Light"
   distractor second; `pick_target` rotates on the scenario index, so whichever
   entity landed at the rotation offset became the target. Measured in the first
   v2 build: 27 of 64 alias rows fired at `Ceiling Light`, teaching nothing about
   aliases while being counted and labelled as alias grounding rows.

   Two-part fix, because there were two bugs: `alias_family` now passes
   `rng_index=0`, **and** `pipeline.generate_row` now honours `rng_index` at all.
   It never did — only `grounding.build_spec` (the eval path) did, so training and
   eval disagreed about what a variant meant. `supported_action_family` set
   `rng_index=0` and had been silently ignored in training the same way.
   Now 0 of 55 alias rows target the distractor.

2. **`_discrimination_available_share` was a 45% guess against 1.5% delivery.**
   It computed `slot share of all action pairs x 0.5` ≈ 0.45, so the rate gate
   effectively compared against the raw request. Delivery is 1.46-1.53%, the gate
   floor at `--discrimination-rate 0.02` was 1.50%, and an honest build therefore
   failed roughly half the time — one did, mid-validation. Replaced with
   `_DISCRIMINATION_DELIVERY_CEILING = 0.02`, measured and dated.

   Note for whoever hits this next: **do not fix a shortfall by lowering the
   requested rate.** The request drives the forcing loop, so asking for less
   delivers less. The rate gate and the request are not the same knob.

## 3a. Remaining gap: the corpus still teaches name-echoing

Gauntlet v2 **was generated** (see "Environment") and passes every build gate.
It still fails the pre-flight probe, which is the gate that predicts the training
outcome:

```
rows                     : 40,000
rows with a listed target: 32,643
  names target verbatim  : 29,829  (91.4%)   <- v1 was 96.3%; contract is <60%
  describes target       :  2,814  ( 8.6%)
    via discrimination   :    585
    via grounding        :    538
    via other            :  1,691
```

Fixing grounding moved this 96.3% -> 91.5%. It is not enough to justify GPU time:
the model can still answer 91.5% of target rows by echoing a name it was handed.

The binding constraint is the **discrimination** category, which delivers ~1.5%
however much is requested. Measured at n=4,000 with `discrimination_rate=0.35`:
1.98% delivered. From 4,109 forced attempts, 1,455 got an eligible slot, ~740
reached the description step, and 80 produced a usable description. The single
largest loss is **no same-domain same-area sibling to discriminate against**:
54% of scenarios (measured over 1,200) have none, so the row is correctly
dropped. Forcing `home_size >= 48` on discrimination rows lifts delivery 1.98% ->
3.4%, so this is home composition, not (as previously concluded) missing entity
attributes. Unstarted, and it is the next real piece of work.

### Other unfinished items

- **Docs not updated.** `training/README.md`, `TRAINING_LOG.md` untouched.
  `docs/CORPUS_v2.md` not written. Deferred on purpose: v2 is not cleared for
  training, so there is no run to log.
- `docs/PLAN_grounding_corpus_fix.md` ceiling section **corrected** with the
  measurements above.
- **Local vs remote drift resolved** — `sampling.py` re-synced to remote
  (both `23ec899c9e854ed0`). See "Environment" for current hashes.
- **Stale remote watcher** PID 147673 (from Sep 11, alive 2d 06h). Its loop is
  `while pgrep -f build_synthetic_dataset; do sleep 20; done`, i.e. it is waiting
  on *any* `build_synthetic_dataset` process — including a v2 build. When one
  runs, this watcher will fire, then try to `ls` and `tail`
  `/srv/datasets/sayso_v3_grounded_20260911_r2/generation.log` and run
  `/tmp/audit_r2.py`. That target directory **exists** (contains a 2026-09-11
  `generation.log`) but has no `.jsonl` outputs, so it is a half-finished run.
  **Kill PID 147673 before starting any build** to avoid it interleaving output
  with the v2 build and confusing the logs.

  ```bash
  SSHPASS="$TRAINING_BOX_PASSWORD" sshpass -e ssh "$TRAINING_BOX_USER@$TRAINING_BOX_HOST" 'kill 147673'
  ```

## 4. Independent defect recorded in GitHub

**#49** — `satellite: flush_preroll truncates the start of the command when
speech follows the wake word without a pause`
https://github.com/mossipcams/SaySo/issues/49

`satellite/sayso/wake/hook.py:69` calls
`flush_bytes(wake_skip_ms)` (default 500 ms, `config.py:45`), discarding the
first 500 ms after wake detection. When the command follows the wake word
without a pause, that window contains the start of the command. The
`buffer.py:107-112` fallback makes it worse: a buffer shorter than 500 ms returns
only the last 250 ms.

Evidence: five production traces, all with beheaded transcripts —
`So turn on the living.` / `So turn the video.` / `Say so.` — every one answered
with a canned refusal.

Already described in `docs/WAKE_WORD_DATA.md:25` and
`satellite/models/README.md:49`, but only as an obstacle to hard-negative mining;
the user-facing voice-path cost was not recorded as a defect.

**Relationship to #39** (step-5000 model refuses valid HA tools): verified
independently against the exact checkpoint in #39 (SHA
`931327525e63feaf0e0fdd9f7bf0b61abca77404380c56a59c49511745c181d0`) that a
**clean** utterance `Turn on the living room TV` still refuses. So #39 stands on
its own; #49 is a separate defect that compounds it. Cross-referenced in a
comment on #39.

**Not the cause:** the `device_registry.devices` deprecation warning in
`custom_components/sayso/routing.py:536`. It is a forward-compat warning (breaks
2027.9.0), it fires in the routing-hints path, and it is unrelated to audio
capture or model output. Worth fixing separately — one-line change.

## 5. Environment

Gauntlet v2 corpus (built locally, 2026-09-13, every gate on, exit 0):

```
training/datasets/sayso_v3_gauntlet_v2_20260913.jsonl          (gitignored)
training/datasets/sayso_v3_gauntlet_v2_20260913.manifest.json

training/.venv/bin/python training/generators/cli.py \
  --count 40000 --seed 20260913 \
  --grounding-rate 0.028 --discrimination-rate 0.02 \
  --real-home training/fixtures/real_home.json --real-home-rate 0.1 \
  --output training/datasets/sayso_v3_gauntlet_v2_20260913.jsonl
```

40,000 accepted in 44,579 attempts; grounding 987 rows (2.47%), discrimination
585 (1.46%), real-home 4,000 (10.0%).

```
sha256  c6e93dd8096bec8e8b58736a9d507844512142d40e3cf6aa8fcefedc38412ebd
```

Determinism verified the expensive way: two full 40k builds at seed 20260913 are
**byte-identical**. **Not cleared for training** — see 3a.

Deployed model (changed this session, at explicit user request, contradicting an
earlier "do not promote"):

```
port 8080 -> /srv/models/SaySo-LFM2.5-230M-40k-gauntlet-step2500-Q8_0.gguf
alias: v3-40k  |  CPU only (N_GPU_LAYERS=0)  |  systemd unit llama-server.service
```

Revert (one command):

```bash
SSHPASS="$TRAINING_BOX_PASSWORD" sshpass -e ssh "$TRAINING_BOX_USER@$TRAINING_BOX_HOST" \
  'sudo cp /etc/default/llama-server.bak-champion-20260913 /etc/default/llama-server \
   && sudo systemctl restart llama-server'
```

Note: run-013 step-2500 is a **statistical tie** with the champion it replaced
(±2 cases of 293). No measured improvement.

As of 2026-09-15 this same artifact is published as **SaySo Gauntlet v1**
(release `model-v1`, sha256 `229c805d…`) and is the default model the Home
Assistant integration downloads. Shipping it did not change its eval standing:
it remains a tie, not a promotion. See
[training/TRAINING_LOG.md](../training/TRAINING_LOG.md) "Shipped model".

Other facts worth keeping:

- `/srv/training-runs/CHAMPION.txt` is **stale** (points at a 2026-09-05 model
  that no longer exists on disk). `/srv/models/CHAMPION.txt` is current.
- `hassil` + `unicode-rbnf` were installed into `/srv/training-runs/.venv` to
  run evals. That venv is shared with training; it is no longer byte-identical to
  what Run 013 started with.
- Step-5000 checkpoint dir is `checkpoint-5000` (no `step-` prefix). A watcher I
  wrote initially used the wrong name — check paths against real dirs.
- Eval artifacts for Run 013: `/srv/training-runs/run013-eval/results/`.

### Source bundle hashes (local == remote, verified)

```
training/generators/pipeline.py                79b2592b3be87246
training/generators/sampling.py                23ec899c9e854ed0
training/generators/config.py                  8f19962d90d0074b
training/scripts/build_synthetic_dataset.py    be4f13f405d1d3b8
```

Remote bundle: `/srv/training-runs/sayso_40k_grounded_gauntlet_20260911-source`

## 6. Changed files in this worktree

Modified:

- `training/generators/config.py` — added `discrimination_rate`,
  `min_rate_achieved_fraction`; grounding default `0.03` → `0.028`; validation.
- `training/generators/pipeline.py` — rate gate, capable-slot selection,
  discrimination rows, helpers; `_grounding_capacity` + its pre-flight raise;
  `_GROUNDING_EFFECTIVE_CEILING` deleted in favour of a derived ceiling;
  `_DISCRIMINATION_YIELD` (0.5 guess) → `_DISCRIMINATION_DELIVERY_CEILING`
  (0.02 measured); `generate_row` honours a variant's `rng_index`.
- `training/generators/grounding.py` — `canonical_variants`, `site_variants`,
  `_sites`, `_catalogue` (cached): 15 scenarios → 345.
- `training/generators/sampling.py` — `take_slot`, `_key_gap`, `_best_keys`,
  `_slot_from`.
- `training/generators/utterances.py` — `describe_target`, `_DESCRIPTIONS`,
  `_BRAND_TOKENS`, `_STOP_TOKENS`.
- `training/generators/validate.py` — discrimination exemption.
- `training/generators/cli.py` — `--discrimination-rate`,
  `--allow-rate-shortfall`.
- `training/scripts/build_synthetic_dataset.py` — `--discrimination-rate`.

New:

- `training/generators/test_grounding_v2.py` — 12 regression tests.
- `training/scripts/audit_discrimination.py` — the pre-flight probe.
- `docs/PLAN_grounding_corpus_fix.md` — plan (ceiling section corrected).
- `docs/HANDOFF_gauntlet_v2.md` — this file.

Nothing is committed.

## 7. Verification status

`training/.venv/bin/python -m pytest training/generators training/tests
training/scripts -q` → **350 passed**, 6 warnings.

Each new test was validated against the broken code it guards, because a test
that passes on broken code is worthless:

- `test_grounding_reaches_its_ceiling_and_all_families` fails when the v1 latch
  and random-slot behaviour are re-introduced.
- `test_grounding_catalogue_scales_to_a_full_size_corpus` fails on the
  15-scenario catalogue: "holds 15 scenarios, capping grounding at 120 rows, but
  the default config asks for 1120".
- `test_every_generated_site_keeps_its_family_contrast` fails with `rng_index=0`
  removed: "ground_alias_s00: alias row targets 'Ceiling Light', not the aliased
  entity 'Dining Room Corner Lamp'".

Validated on the built 40k corpus, not just in unit tests:

| Check | Result |
|---|---|
| grounding label contracts (10 labels, act vs refuse) | 0 failures |
| alias rows targeting the distractor | 0 of 55 (was 27 of 64) |
| held-out eval prompts appearing in the corpus | 0 (guard confirmed live: 296 prompts loaded, not a silent `ImportError` fallback) |
| action rows whose target is absent from their own context/tools | 0 |
| determinism, two full 40k builds at one seed | byte-identical |

Note: `pytest` is **not** installed in `/srv/training-runs/.venv` on the remote,
so the suite only runs locally.

## 8. Recommended next steps, in order

1. ~~Budget grounding capacity in the quota planner~~ — **done differently, and
   the planner is not where it belonged.** See section 3.
2. ~~Correct `docs/PLAN_grounding_corpus_fix.md`~~ — **done.**
3. ~~Generate Gauntlet v2~~ — **done** (local, see "Environment").
4. ~~Promote the pre-flight probe~~ — **done**:
   `training/scripts/audit_discrimination.py`, exits 1 above `--max-verbatim`
   (default 0.60) so a build script can gate on it.
5. **Raise the discrimination rate** (section 3a). This is the blocking item for
   a v2 training run. Measured leads, cheapest first: force `home_size >= 48` on
   discrimination rows (1.98% -> 3.4%, one line in `generate_row`); then widen
   past the `individual` / single-call / value-free eligibility filters, which
   cost ~2/3 of forced attempts before a description is even attempted. Re-run
   the probe; the target is verbatim share < 60%.
   **Do not spend GPU time until that probe passes.**
6. **Update `training/README.md` and `TRAINING_LOG.md`**; write
   `docs/CORPUS_v2.md` — once there is a cleared corpus to describe.
7. **Kill stale remote watcher** PID 147673.
8. **Fix `routing.py:536`** deprecation (cheap, independent).
9. **Fix #49** (`wake_skip_ms` truncation) — separate from corpus work.

## 9. Honest caveats

- I misdiagnosed the live trace twice before landing: first as a model/entity
  failure, then as the cause of #39. Reading all traces and running the clean-input
  test is what settled it. Treat any claim here without a measurement behind it as
  unverified.
- I read the wrong `CHAMPION.txt` and reported a non-existent model as missing.
- I quoted an unreliable `ps %cpu` figure as fact (stale snapshot; 12 threads on
  6 cores). Do not trust `ps %cpu` for CPU-bound claims — use load average or
  `/proc/<pid>/stat` deltas.
- Everything measured at n≤6000 looked fine and the 40k run exposed the collapse.
  Do not extrapolate from small runs on this generator — and when a rate looks
  size-dependent, check first whether it is a **constant** divided by n. It was.
- The first diagnosis of that collapse (quota-planner starvation) was wrong, and
  five failed fix attempts against it were the evidence that it was wrong. Five
  reverts is a signal to re-measure the cause, not to try a sixth fix.
- Corpus generation is cheap (minutes); the failure mode is that it looks
  successful unless the rate gate is active. Keep `min_rate_achieved_fraction`
  at `0.75` — `0.9` tripped on ordinary variance (measured 2.4–2.8% spread).
