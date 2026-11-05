# Plan: guard the volume-step override import

## Problem

`custom_components/sayso/__init__.py` imports `async_register_volume_step` from
`.volume_step` at module top. `volume_step.py` imports Home Assistant internals
(`MediaSetVolumeRelativeHandler`, `intent.MatchTargetsConstraints`, ...). If a
future HA release renames or moves one, the ImportError prevents the whole
integration from loading.

## Scope

Fallback behaviour only: if the override cannot be imported or registered,
log the failure and let HA's default relative volume handler stay active.
Setup still succeeds.

## Files

- `custom_components/sayso/__init__.py`: drop the module-top import; add
  `logging` and `_LOGGER`; add `_async_register_volume_step(hass)` that imports
  and calls `async_register_volume_step` inside `try`/`except Exception`, logging
  with `_LOGGER.exception`; `async_setup` calls the wrapper and returns `True`.
- `tests/test_volume_step.py`: add one test that makes the `volume_step` module
  import fail (`sys.modules` entry set to `None`), then asserts setup succeeds,
  an ERROR log containing "Volume step fallback disabled" is emitted, and the
  INTENT_SET_VOLUME_RELATIVE handler is HA's `MediaSetVolumeRelativeHandler`
  (not `StepVolumeRelativeHandler`). Existing tests are not modified.
- `docs/PLAN_volume_step_guard.md`: this file.

Not touched: `volume_step.py`, any other test file, any other source file.

## Verification

- `.venv/bin/python -m pytest -q tests/test_volume_step.py` (expect 7 passing).
- Before/after comparison of `tests/test_init.py`, `tests/test_conversation.py`,
  `tests/test_schema.py` (some failures pre-exist with HA 2026.9.4).
- `.venv/bin/python -m ruff check .` and `.venv/bin/python -m ruff format --check .`.
- `git status --short` and `git diff --stat` show only the three allowed files.

## Out of scope

No commit, push, or branch change. No new dependencies. No code comments or
docstrings.
