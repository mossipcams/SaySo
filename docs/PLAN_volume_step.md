# Plan: relative volume for VOLUME_STEP-only media players (defect #34)

## Problem

"turn down the tv" produces `HassSetVolumeRelative {name: "TV", volume_step: "down"}`
(correct). Home Assistant's `MediaSetVolumeRelativeHandler`
(`homeassistant/components/media_player/intent.py`) matches targets with
`features=MediaPlayerEntityFeature.VOLUME_SET`. A Roku TV that supports only
`VOLUME_STEP` fails matching with `MatchFailedReason.FEATURE`, even though the
up/down branch only calls `async_volume_up` / `async_volume_down`.

## Scope

Fix lives in SaySo until HA fixes it upstream. Override the handler, keep HA's
behaviour for every case HA already handles.

Files (only these):

- `custom_components/sayso/volume_step.py` (new): `StepVolumeRelativeHandler`
  subclassing HA's `MediaSetVolumeRelativeHandler` (reuses its `slot_schema`,
  `intent_type`, `description`, `platforms`), plus `async_register_volume_step(hass)`.
- `custom_components/sayso/__init__.py`: call `async_register_volume_step(hass)`
  once from `async_setup`.
- `custom_components/sayso/manifest.json`: `"after_dependencies": ["media_player"]`
  so HA's own registration runs first and ours overwrites it.
- `tests/test_volume_step.py` (new).
- `docs/PLAN_volume_step.md` (this file).

Out of scope: upstream HA changes, evals, prompts, tool schema changes, existing
tests.

## Behaviour

`async_handle(intent_obj)`:

1. `return await super().async_handle(intent_obj)`.
2. On `MatchFailedError` whose `result.no_match_reason` is `FEATURE`, retry matching
   with the same name/area/floor/assistant and preferences, but
   `features=MediaPlayerEntityFeature.VOLUME_STEP`. Use `self.async_validate_slots`
   and build `MatchTargetsConstraints` / `MatchTargetsPreferences` as HA does.
3. If the retry finds no match, re-raise the original error.
4. Multi-target rule (same as HA): more than one match and no `name` slot keeps only
   `playing` states; none playing raises `MatchFailedError` with `STATE`.
5. Step: `"up"` or positive number → `async_volume_up()` once per target;
   `"down"` or negative number → `async_volume_down()` once; zero → no call.
6. `HomeAssistantError` → `intent.IntentHandleError`, as HA does.
7. Response with `async_set_states(matched states)`.

VOLUME_SET numeric changes still go through HA's `super()` path, unchanged.

## Verification

- `tests/test_volume_step.py`:
  - (a) VOLUME_STEP-only player: `down` → `async_volume_down` once; `up` → up once;
    numeric `-10` → one `async_volume_down`.
  - (b) VOLUME_SET player: numeric `+10` on volume 0.5 → `async_set_volume_level(0.6)`
    (HA path via `super()`).
  - (c) Player with neither feature → `MatchFailedError` (original, reason FEATURE).
  - (d) After `async_setup_component(hass, "sayso", ...)` the registered
    `HassSetVolumeRelative` handler is a `StepVolumeRelativeHandler` and its
    `slot_schema` is HA's (same object).
- Gate: `cd /Users/matt/Projects/SaySo__worktrees/volume-step && /Users/matt/Projects/SaySo/.venv/bin/python -m pytest -q tests/test_volume_step.py tests/test_init.py tests/test_conversation.py tests/test_tool_key_order.py tests/test_schema.py`
- Also run `ruff check` and `ruff format --check` on changed files, and the full
  `pytest tests --ignore=tests/test_realistic_v3.py`.

## Stop conditions

- Needs an edit outside the scope list.
- Needs changes to existing tests.
- Needs a network call or a Home Assistant contact.
