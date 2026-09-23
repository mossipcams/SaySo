# v5 feasible mix — follow_up, correction, grounding on

Implement the mix that fits the current catalogues. Not the “every knob at 11”
40k recipe in `docs/PLAN_AGGRESSIVE_SFT_CORPUS.md`. No 40k generate. No
training launch. No eval/runtime edits. No HA entity rename/add. Do not cut
family shares to paper over the sampler.

`full_sft_v4.yaml` stays as the first-turn-only baseline (already has
`stt_log_rate: 0.10`). New recipe: `training/configs/generation/full_sft_v5.yaml`.

## Mix (primary allocations sum 1.00)

| Family | Share |
|---|---:|
| ordinary | 0.22 |
| follow_up | 0.13 |
| status | 0.10 |
| settings | 0.08 |
| clarify | 0.08 |
| correction | 0.06 |
| junk | 0.06 |
| multi_action | 0.06 |
| exclusion | 0.05 |
| aliases | 0.05 |
| unavailable | 0.04 |
| unsupported | 0.04 |
| absence | 0.03 |

Cross-cuts:

- `stt_log_rate: 0.10` (keep; independent of generic STT)
- `stt_noise_rate: 0.15` (keep; not applied to log-STT rows)
- `grounding.rate: 0.08` (measured: 562 variants × `near_duplicate_limit` 8 = 4496
  capacity; `grounding_available_share` ≈ 0.083 on 40k)
- `real_home.rate: 0.10` (not 0.20)
- `bare_name_rate: 0.35`
- `discrimination.rate: 0.015`
- `paraphrase_enabled: false`
- leftover real-home `family_mismatch` already retries synthetic in `pipeline.py`

If `verify_feasible` fails, grow identity (catalogue / near-dup), do not shrink
`follow_up`, `correction`, or `grounding.rate`.

## Families

### `follow_up`

Two-device area graph like `clarify`. First user is generic (“the light”). First
assistant is named “Did you mean A or B?” (`train_on_turn: true`). Second user
names one candidate. Second assistant is the gold action on that entity
(`train_on_turn: true`). Gold name is `entity["name"]`, never concatenated.

`gold_matches_family("follow_up")`: clarify (spec before extra turns) **or**
action (finished row / `classify_row`).

### `correction`

Pre-execution only (runtime `build_correction_messages` shape). Individual
action whose name does not already contain its area. User can say area + name.
First assistant emits concatenated `"{area} {name}"` with `train_on_turn: false`.
Tool payload matches `format_synthetic_validation_error` (`error.code`,
`message`, `allowed_tools`, `schema_fingerprint`) — copy the JSON shape in the
generator; do not import `custom_components`. Second assistant is the correct
call with `name=entity["name"]` and `train_on_turn: true`. No
post-`MatchFailedError` retry.

If the target name already includes the area, `family_mismatch` and retry.

## Files

- `training/generators/planning.py` — `PRIMARY_FAMILIES`, `FAMILY_ROBUSTNESS`,
  `_pick_capability_operation`
- `training/generators/scenarios/core.py` — `configure_family_scenario`,
  `_FAMILY_GRAPH_CONSTRAINTS`
- `training/generators/gold.py` — `gold_matches_family`
- `training/generators/rendering.py` — extra turns for both families
- `training/generators/row_generation.py` — wire follow_up rewrite / correction
  skip when name already contains area
- `training/configs/generation/full_sft_v5.yaml` — new
- `training/configs/generation/smoke.yaml` — small shares of both new families
  so the 120-row suite actually hits them (cut ordinary to keep sum ≈ 1.0)
- `training/tests/test_canonical_generator.py` — unit coverage for both shapes
- optional colocated tests under `training/tests/` if v5 recipe feasibility
  does not fit the canonical file

Do not edit `evals/`, `custom_components/`, or `full_sft_v4.yaml`.

## Verification

```bash
cd training && .venv/bin/python -c "
from pathlib import Path
from generators.config import GeneratorConfig
from generators.planning import verify_feasible, build_plan
from generators.grounding import training_variants
root = Path('.').resolve().parent
cfg = GeneratorConfig.from_yaml(root/'training/configs/generation/full_sft_v5.yaml', repo_root=root)
verify_feasible(
    cfg.count, cfg.allocations,
    near_duplicate_limit=cfg.near_duplicate_limit,
    area_required={},
    grounding_variants=len(training_variants()),
    grounding_rate=cfg.grounding_rate,
    real_home_path=cfg.real_home_path,
    real_home_rate=cfg.real_home_rate,
)
print('feasible', cfg.grounding_rate, cfg.allocations)
"
cd training && .venv/bin/python -m pytest \
  tests/test_canonical_generator.py \
  tests/test_stt_noise.py \
  tests/test_sampling.py \
  -q
```

Generate one follow_up row and one correction row (or a short smoke run) and
assert:

- follow_up: ≥2 user turns; first assistant has no tools and names both
  candidates; last trained tool call uses a candidate’s exact `name`
- correction: first tool-call assistant has `train_on_turn` false; second
  trained call uses gold `name` without concatenating area; tool error JSON
  has `error.code`

Do not write a 40k jsonl.
