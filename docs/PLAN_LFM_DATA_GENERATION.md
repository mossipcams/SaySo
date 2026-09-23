# Generate the next LFM training corpus (full fine-tune, defect-driven)

Scope: generate the next synthetic corpus for **full-parameter** fine-tuning of
`LiquidAI/LFM2.5-230M-Base`, targeting the open model defects (#39, #94, #96,
#97). No training launch, host changes, runtime edits, wake-word changes, or
modifications under any `tests/` directory.

## Evidence gathered before writing this plan

Measured on this checkout with the pinned Base tokenizer and the recipe
generator (`configs/generation/smoke.yaml`, 120 rows), plus the three shipped
40k manifests in `training/datasets/`.

| Finding | Measurement |
|---|---|
| Real entity for the failing TV is named `TV`, area `Living Room` | `training/fixtures/real_home.json` → `media_player.living_room_tv` |
| Every synthetic entity name is area-prefixed | `homes.py::_random_entity_name` — all shapes emit `f"{area} {role}"` |
| `GetDateTime` positives in every shipped 40k corpus | **0** (`get_datetime_positive_min: 1`) |
| `clarify` rows in the shipped 40k corpora | 262/40,000 = **0.65%** |
| Shipped 40k corpora built from `production.yaml` | **none** — all used the capability-quota path, `allocations` empty |
| `count_row_tokens` on rows with tool calls | raises `UndefinedError` (92/120 rows) |
| `count_row_tokens` on rows without tool calls | returns **2** (`len(BatchEncoding)`, transformers >= 5) |
| Resulting manifest token stats | `{min: 0, p50: 0, max: 0}` — the budget gate is dead |
| True row length, measured correctly | min 2,663 / p50 5,401 / p95 6,508 / max 6,574 |

## Defect → data cause → fix

### #94 `MatchFailedError` on "turn off the living room TV" — root cause

The corpus teaches `name = "<Area> <Device>"` on 100% of rows. The user's real
home names that entity `TV` with `areas: Living Room`. The model emits
`name: "Living room TV"`, which exists nowhere, so HA's matcher fails. Same
mechanism explains #39's `Sofa TV` → `Sofia TV` and "big screen" misresolution.

**Fix:** `homes.py` grows a `bare_name_rate` so a share of entities are named
without the area prefix, carrying the area only in `areas`. The model must
learn to split "the living room TV" into `name` + `area` instead of
concatenating. Also raise real-home mixing off 0%.

### #97 noisy STT emits unavailable tools / invalid output

0.65% clarify supervision and no non-command supervision at all. The model was
never shown a junk transcript with a correct no-tool answer.

**Fix:** new `junk` family — garbled/non-command transcripts, zero tool calls,
one short "didn't catch that" reply. Allocated a real share.

### #39 14/27 standard tools refuse to fire

`min_positive_per_tool: 1` and `get_datetime_positive_min: 1` are vacuous
gates; `GetDateTime` actually shipped at zero.

**Fix:** real per-tool and per-operation floors in the recipe.

### #96 no-tool junk turns marked successful

Tracing/observability, not model data. Out of scope here; the `junk` family
gives it a labeled corpus to test against.

## Implementation

1. **Fix `count_row_tokens`** (`generators/validation.py`) — parse JSON-string
   tool-call arguments before templating, and take the length of `input_ids`
   rather than the `BatchEncoding`. This is the root-cause fix: the manifest,
   the budget gate, and full-FT memory planning all read it.
2. **`bare_name_rate`** in `generators/homes.py`, threaded through
   `GeneratorConfig` and the recipe.
3. **`junk` family** — `planning.py`, `scenarios/core.py`, `row_generation.py`,
   `rendering.py`, `coverage.py`, `gold.py`.
4. **New recipe** `configs/generation/full_sft_v4.yaml` — real allocations,
   real coverage floors, real-home mixing, junk + bare-name shares.
5. Build, audit, and record the corpus; update `training/README.md` and
   `training/TRAINING_LOG.md`.

Files: `training/generators/{validation,homes,config,planning,row_generation,rendering,coverage,gold}.py`,
`training/generators/scenarios/core.py`,
`training/configs/generation/full_sft_v4.yaml`,
ignored `training/datasets/` artifacts, `training/README.md`,
`training/TRAINING_LOG.md`, and this plan.

## Verification

- `python -m pytest training/tests -q` stays green.
- One runnable focused check for the token-count fix and the new family
  (`training/tests/`, new file only — no existing test modified).
- Fixed-seed smoke build; inspect junk rows and bare-name rows by hand.
- Full-corpus audit: exact allocations, per-tool positive floors incl.
  `GetDateTime`, junk share, bare-name share, real token min/p50/max with no
  truncation, held-out eval overlap, duplicate rate.
- Full fine-tune host/framework compatibility is a **documented launch gate**.
  Do not claim a training run or a model improvement from this work.
