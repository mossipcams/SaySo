# Area context training contract

## Scope

Make training prompts and the SaySo runtime prompt carry the same
`satellite_area`, `target_area`, and `target_area_source` fields. Generate
deterministic area-routing scenarios and fail closed when required area
categories or requested rates are absent.

## Files to touch

- `custom_components/sayso/area_context.py`: shared pure-Python contract,
  resolution, and prompt rendering.
- `custom_components/sayso/conversation.py`: derive runtime area context from
  the request, exposed entity catalog, and satellite/requesting device.
- `training/generators/context.py`: render the shared area contract.
- `training/generators/scenarios.py` and `training/generators/gold.py`: carry
  satellite and target area semantics into authoritative labels.
- `training/generators/grounding.py`: add area-focused contrast families.
- `training/generators/config.py`, `pipeline.py`, `stats.py`, and
  `training/scripts/build_synthetic_dataset.py`: configure the area mix,
  production recipe, manifest rates, and fail-closed gates.
- `training/generators/test_area_context.py`: colocated regression coverage.
- `training/configs/lfm25-230m-40k-grounded-gauntlet-trl.yml`: document the
  production recipe output contract if the existing recipe is superseded.

## Verification

- Run the area-context colocated tests and the generator tests.
- Run the existing runtime conversation/routing tests that cover system prompt
  and device-area behavior.
- Generate a deterministic small corpus and inspect its manifest for area
  source counts, full-catalog/real-home settings, and required-family gates.
- Check that identical requests change target when satellite area changes and
  that an explicit area overrides satellite area.

## Review fixes

- Keep area-context utterances grammatical and validate the generated default
  corpus with the existing linguistic suite.
- Make `satellite_area` authoritative throughout gold labels and context
  rendering, while retaining legacy fallback support for older fixtures.
- Wire the six fixed area families into the v3 production quota and report
  their delivery separately from source rates.
- Make source-category gates conditional on a positive requested area rate and
  count ambiguity only when the resolver actually reports ambiguity.
- Run generator tests, targeted corpus smoke checks, compilation, and diff
  validation before committing and opening the PR.
