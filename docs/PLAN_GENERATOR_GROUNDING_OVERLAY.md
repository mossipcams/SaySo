# Recipe grounding overlay (no slot steal)

The v5 40k run on 192.168.1.76 died at 7099/40000 after 800k attempts.
`family_mismatch` was 785401. Achieved families were exactly `follow_up` +
`multi_action`. Grounding.rate stays 0.08. Do not shrink family shares.

## Cause

On a recipe run, `want_grounding` stayed true while `grounding_missing` was
nonempty, then `next(plan.slots)` replaced the family slot with the first
remaining slot whose `(capability, operation)` is in `grounding_pairs()`.
That is a deterministic steal, not an overlay. After follow_up and
multi_action filled, the same next slot family-mismatched forever.

Quota-tracker runs (`grounding_capable_slot`) are unchanged. They are what
`tests/test_grounding_v2.py` covers.

## Change

- Overlay grounding only when the already-picked slot is a carrier family
  (`ordinary`, `aliases`, `settings`) whose `(cap, op)` is in `grounding_pairs()`
  and is not an area scenario.
- Recipe path must not overwrite the picked slot with `next(plan.slots)`.
- `want_grounding` is a deficit rate on carrier slots. Required catalogue
  families force overlay only on carriers (`bool(grounding_missing)` must not
  make every family attempt a grounding steal).
- Cache `grounding_pairs()` (`lru_cache` / `frozenset`) so each attempt does
  not rebuild the catalogue 40k times.

## Files

- `training/generators/grounding.py`
- `training/generators/pipeline.py`
- `training/tests/test_v5_feasible_mix.py` (recipe-path regression)
- this plan

## Verification

```bash
cd training && .venv/bin/python -m pytest \
  tests/test_v5_feasible_mix.py \
  tests/test_grounding_v2.py \
  tests/test_canonical_generator.py \
  tests/test_generator_pipeline.py \
  -q
```

Do not launch 40k generate or LlamaFactory train in this pass.
