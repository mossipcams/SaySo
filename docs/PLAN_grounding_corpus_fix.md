# PLAN: Gauntlet v2 — grounding-rate fix and entity-discrimination rebalance

Status: PLAN (not yet implemented)
Author: agent, 2026-09-13
Scope: `training/generators/`, `training/scripts/build_synthetic_dataset.py`,
new dataset `sayso_40k_grounded_gauntlet_v2_<date>`, docs.

## Why

Run 013 (`SaySo-LFM2.5-230M-40k-grounded-gauntlet`, 40k rows, 2 epochs)
finished but the corpus did not teach the skill its namesake describes.

Measured on the frozen `realistic_20260908_v2` suite at step 2500 (best) and
5000 (final): 58/120, with `args_mismatch` dominant. Inspecting the failures
showed the model picks the **right tool and domain but the wrong entity**:

| expected | produced |
|---|---|
| `Living Room Ceiling Light` | `Sofa Reading Lamp` |
| `Morgan's Bedside Lamp` | `Bedroom Roller Blind` |
| `Sofa TV` | `Bedroom Curtains` |
| `Lounge Television` | `Laundry Television` |

So the failure is **entity binding**, not formatting.

### Hypotheses tested and rejected

All three data-integrity hypotheses were checked against the corpus and
**disproven**:

| hypothesis | result | evidence |
|---|---|---|
| eval scorer confused by aliases | rejected | 0 of 23 `args_mismatch` are alias/canonical mixups |
| corpus lacks same-domain distractors | rejected | 96% of target rows have >= 2 same-domain competitors; 61% have a same-area competitor |
| corpus inconsistent on ambiguous targets | rejected | 2 of 173 ambiguous utterances map to > 1 target, both case variants |
| held-out leakage | rejected | overlap audit 0/0/0/0 |

The corpus is well-formed, consistent, and distractor-rich.

### Root cause

The **utterance is lexically trivial**. Measured over the **full 40,000-row
corpus** (not a sample):

```
rows with a listed target      : 27,596
  utterance names entity verbatim : 26,583  (96.3%)
  utterance describes entity      :  1,013  ( 3.7%)

2+ same-domain competitors : 11,865  (43.0%)
has same-area peer         :  6,875  (24.9%)

distinct utterances        : 28,399
duplicate utterances       : 11,601   (29% of rows repeat a phrasing)
```

The distractor entities are present but decorative — the model can answer by
echoing a name it was just given, so it never learns to *resolve* a device.
Only **3.7%** of rows force description-to-entity resolution.

The manifest corroborates: `by_targeting: individual = 32614 (82%)`,
`area = 1320 (3%)`, `floor = 3055 (8%)`, `none = 3011 (8%)`.

Measured note: 3,258 target rows name an entity that is **not listed** in that
row's static context, and 9,146 rows have no target at all (refusals,
clarifications, status queries). Both are expected shapes, but v2 should keep
the `target_not_listed` count near zero since a target absent from context is
unlearnable.

This is a **design/config gap, not a silent generator failure**:

```jsonc
"grounding_rate": 0.03,        // only 3% grounding rows
"paraphrase_enabled": false,   // paraphrasing OFF
"paraphrase_variants": 0,      // zero variants
"ordinary_rate": 0.75,
```

### Two concrete generator defects

1. **Grounding under-delivery (14x).** Requested 3% (1200 rows), produced
   `rows: 88`, `achieved_rate: 0.0022`. The pipeline **did not fail** and the
   gate **passed** — `pipeline.py` only raises on *missing families*, never on
   a rate shortfall.

2. **`grounding_required` is a one-shot latch.** In `pipeline.py`:

   ```python
   grounding_required=config.grounding_rate > 0 and not grounding_rows,
   ```

   `not grounding_rows` is falsy the moment the first grounding row lands, so
   forcing stops immediately and the remaining budget reverts to the 3% random
   draw. Combined with `pick_variant` returning `None` when a
   `(capability, operation)` pair has no variant, most slots silently produce
   ordinary rows.

3. **No rate gate.** `report["grounding"]` records `achieved_rate` but nothing
   asserts it, so a 14x shortfall ships as a pass.

4. **`similar_name` robustness is dead code.** `pick_robustness` selects
   `"similar_name"` but no handler exists for it in `scenarios.py` (only
   `exclusion`, `multi_action`, `alias_distractor` are handled), so those rows
   silently degrade to ordinary. This is the natural hook for the new
discrimination category and should be implemented rather than left inert.

## Scope

### In scope

- Enforce the grounding quota in `pipeline.py` (fix the latch; assert rate).
- Add an explicit **entity-discrimination** target category to the generator:
  utterances that *describe* a device without naming it, with same-area
  same-domain peers present.
- Enable paraphrasing.
- Add rate/coverage gates so a short corpus cannot pass.
- Add a pre-flight discrimination probe usable before spending GPU time.
- Generate **Gauntlet v2** (40k rows) and its manifest.
- Update `TRAINING_LOG.md`, `training/README.md`, and add
  `docs/PLAN_grounding_corpus_fix.md` (this file) + a short
  `docs/CORPUS_v2.md` describing the v2 contract.

### Out of scope

- Retraining (separate decision; run after v2 is verified).
- Touching `ARCHITECTURE.md` — no ownership/trust-boundary change.
- Promoting or unpromoting any checkpoint.
- Changing the frozen eval suites (they are the fixed yardstick).

## Files to touch

| File | Change |
|---|---|
| `training/generators/pipeline.py` | fix grounding latch; add rate assertion; emit `by_targeting` discrimination stats |
| `training/generators/grounding.py` | add discrimination variants / helper if needed |
| `training/generators/config.py` | add `discrimination_rate`, bump `grounding_rate`, enable paraphrase defaults |
| `training/generators/cli.py` | expose new flags |
| `training/generators/utterances.py` | description-based phrasing (no entity name) |
| `training/generators/test_coverage_gates.py` | tests for the new gates |
| `training/scripts/build_synthetic_dataset.py` | accept new config; never hand-filter |
| `training/scripts/audit_discrimination.py` (new) | pre-flight probe: report `utterance_contains_exact_target_name` rate + competitor stats |
| `training/README.md`, `TRAINING_LOG.md`, `docs/CORPUS_v2.md` | docs |

## Verification

Ordered, cheapest first. **Step 1 gates everything after it.**

1. **Pre-flight discrimination probe** (no GPU). Run
   `audit_discrimination.py` on the current 40k corpus and on v2.
   - Baseline: ~95% of target rows name the target verbatim.
   - v2 target: **< 60%**, with >= 30% of rows using description-based
     reference and same-area peers.
   - If v2 does not clear this, do not retrain.
2. **Unit tests**: `python -m pytest training/generators training/tests -q`.
   New tests must fail on the old config (regression guard for the latch and
   the missing rate gate).
3. **Manifest gates**: `achieved_rate >= requested_rate * 0.9`,
   `by_targeting.discrimination >= discrimination_rate * 0.9`,
   `render_rows == accepted`, held-out overlap 0 against all four frozen
   suites, `uncovered_operations` empty.
4. **Determinism**: same seed -> identical corpus SHA-256 across two runs.
5. **No eval contamination**: v2 must not contain any eval suite utterance.
6. Only then: consider a 1-epoch retrain from Base (epoch 2 was shown to add
   nothing: 183/293 at 2500 vs 178/293 at 5000).

## Risks and notes

- Raising description-based rows makes rows genuinely **ambiguous** unless
  peers are constrained. The v2 contract must ensure each such row has exactly
  one valid target (or is explicitly an `ambiguity`/clarify row). Otherwise we
  teach guessing. **This is the main correctness risk of the change.**
- Changing `grounding_rate` upward changes corpus composition; existing
  recipe-lock and gold suites must stay untouched as the yardstick.
- Do not hand-edit the render output (`render_rows` must equal `accepted`).
- Keep `ALLOWED_HASS_TOOLS` as the pinned training contract.
- If the 95% figure shifts when measured over all 40k rows, record the true
  number in `docs/CORPUS_v2.md` before/after rather than the sampled estimate.

## Open question for the user

Which description-based targeting proportion is wanted? Default proposed:
**35%** (`discrimination_rate: 0.35`), leaving 65% named-target rows so the
model does not lose the ability to answer directly-named requests.

## Measured ceilings (2026-09-13, corrected)

An earlier revision of this section claimed a fixed "~2.8% effective" grounding
ceiling caused by capable-slot competition. **That was wrong**, and it was wrong
in a way worth recording: it was inferred from small runs, where it happens to
be true for a different reason.

### Grounding: the ceiling was the catalogue, not the scheduler (FIXED)

Every grounding variant is one fixed scenario (fixed home, fixed request), and
`DuplicateTracker` accepts any one scenario at most `near_duplicate_limit` (8)
times. So the catalogue size set an **absolute** row cap, independent of corpus
size:

```
15 variants x 8 = 120 rows, whatever n is
```

Measured before the fix: 49 rows at n=2k (2.4%), 135 at n=12k (1.1%), 136 at
n=40k (0.34%). The rate looked size-dependent only because a constant divided by
n does. At n=12k, 936 of 1399 forced grounding attempts were rejected
`duplicate_semantic_id`.

Fix: `grounding.site_variants` instantiates the whole contrast set at 30 further
areas, taking the catalogue to 345 scenarios (ceiling 2760 rows). Measured after:

| n | achieved grounding |
|---|---|
| 2,000 | 2.65% |
| 12,000 | 2.58% |
| 40,000 | **2.54%** |

Flat, as required. `pipeline._grounding_capacity` now raises **before** the build
when `count * grounding_rate` exceeds the catalogue ceiling, so the same class of
defect cannot ship silently again.

### Discrimination: ~1.5-2% delivered, and 35% is not reachable by tuning

Measured, n=4000, `discrimination_rate=0.35` requested → **1.98% delivered**
(0.34% with siblings guaranteed by forcing `home_size >= 48`: **3.4%**). The
compounding filters, from 4,109 forced attempts:

| Stage | Surviving |
|---|---|
| forced discrimination attempts | 4,109 |
| got a lights/covers/switches slot with room | 1,455 |
| reached the description step (individual, single call, value-free) | ~740 |
| a unique description could be rendered | 80 |

The dominant single loss is **no same-domain same-area sibling**: measured over
1,200 scenarios, 54% have none, so there is nothing to discriminate against and
the row is correctly dropped. Sibling presence is a function of home size, not of
missing entity attributes — the earlier "needs positional attributes" conclusion
is not what the numbers say. Raising it meaningfully is a home-composition
change, and is still unstarted.

Consequences recorded in code:

- `grounding_rate` default stays `0.028`; it is now genuinely reachable (2.54%
  delivered at 40k against a 2.8% request).
- `discrimination_rate` default stays `0.0`; a v2 build passes `0.02` and
  delivers ~1.5%. Requesting 0.35 still delivers ~2%.
- The rate gate compares against the reachable ceiling
  (`min(capable-slot share, catalogue capacity)`), not the raw request.
- `min_rate_achieved_fraction` is `0.75`, not `0.9`: measured run-to-run variance
  is 2.4-2.7%, and 0.9 tripped on ordinary variance.
