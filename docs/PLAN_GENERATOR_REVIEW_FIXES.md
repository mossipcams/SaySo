# Generator review fixes

## Scope

- Make area resolution prefer canonical entity names over generic aliases.
- Make the ambiguous area scenario unambiguously ambiguous.
- Render the system prompt from the same resolved area context stored in metadata.
- Fail generation when requested real-home mixing falls below its configured gate.

## Files to touch

- `custom_components/sayso/area_context.py`
- `training/generators/context.py`
- `training/generators/grounding.py`
- `training/generators/labels.py`
- `training/generators/pipeline.py`
- Colocated generator tests for the changed contracts.

## Verification

- Run focused area, grounding, coverage, and real-home generator tests.
- Run a deterministic 1,000-row generation and assert prompt metadata agreement.
- Run Ruff and `git diff --check`.
