# Make the v4 family mix feasible, then generate the corpus

Scope: the v4 generator already exists in this worktree (`junk`, `bare_name_rate`,
token gate, `full_sft_v4.yaml`). Do **not** relaunch training, edit runtime,
or touch `evals/`. Finish the corpus only after an 8k probe meets every family
quota instead of burning the attempt budget.

## What the 8k probe actually showed

`count=8000`, `max_attempts_multiplier=20` → **160k attempts**, then
`failed to meet allocation` with the shortfall on **unavailable** and
**unsupported**. That is infeasibility, not slowness.

Mechanism:

1. Easy families (`ordinary`, `status`, `settings`, …) fill first.
2. `balance_real_home` then drives remaining slots onto the **single** real home
   (`training/fixtures/real_home.json`, `home_id=real_home_train`) to hit the
   10% mix.
3. Those leftover slots are disproportionately `unavailable` / `unsupported`.
4. Pre-widening, those families were pinned to one or two
   `(capability, operation)` pairs (`climate/set_temperature`;
   `media_players/{turn_on,volume_set}`).
5. `semantic_id` hashes `home_id` + cap/op + target. One real thermostat is one
   identity. After `near_duplicate_limit` (8) accepts, every further real-home
   attempt is `duplicate_semantic_id`.
6. The loop keeps retrying until 160k and still misses the quota.

The working tree already widens the picker (`_WITHHOLDABLE_OPERATIONS` /
`_UNDERPOWERED_OPERATIONS` in `planning.py`). That is **unproven** at 8k, and
it does not stop the real-home balancer from funneling refusal slots onto one
registry.

## Required outcome

1. An 8k dry-run of the v4 mix completes with **exact** family allocations
   (including unavailable and unsupported) well under 160k attempts.
2. `verify_feasible` fails closed **before** the loop when a refusal family
   cannot fill under the recipe's real-home rate and near-duplicate limit.
3. Real-home mixing stays on families that teach this house's names
   (ordinary / status / settings / aliases / clarify). `unavailable`,
   `unsupported`, `junk`, and `datetime` stay synthetic: they teach catalog
   withholding and feature gaps, not "Thermostat".
4. Then generate the 40k `full_sft_v4` corpus, audit it, and record it in
   `training/README.md` + `training/TRAINING_LOG.md`.

## Files

- `training/generators/planning.py` — identity-capacity check in
  `verify_feasible`; keep the widened picker.
- `training/generators/pipeline.py` — do not select the real home for
  identity-poor families.
- `training/generators/scenarios/core.py` — put withheld tools (and, if still
  colliding, unsupported names) into `semantic_id`.
- `training/generators/row_generation.py` — only if family-mismatch still
  discards valid unsupported graphs.
- `training/configs/generation/full_sft_v4.yaml` — only if a share must move
  *after* the identity fix; do not paper over a broken sampler by shrinking
  unavailable/unsupported first.
- `training/tests/test_defect_v4_corpus.py` — runnable check that an 8k-shaped
  mix (or a scaled equivalent) fills unavailable/unsupported without exhausting
  attempts; assert real-home rows are not required of those families.
- `training/README.md`, `training/TRAINING_LOG.md` — after a successful 40k
  build.
- Ignored `training/datasets/sayso_full_sft_v4_20260922.*`.

## Verification

- `python -m pytest training/tests/test_defect_v4_corpus.py training/tests/test_canonical_generator.py -q`
- `python -m pytest training/tests -q` stays green.
- 8k probe: override `count=8000` on the v4 recipe (or an equivalent inline
  config), dry-run, confirm every family meets its quota and attempts stay
  well under `count * 20`.
- 40k: `cd training && python -m generators.cli --config configs/generation/full_sft_v4.yaml`
  then audit allocations, GetDateTime floor, junk share, bare-name share,
  real token min/p50/max, held-out overlap, duplicate rate.

No training launch. No claim that the model improved.
