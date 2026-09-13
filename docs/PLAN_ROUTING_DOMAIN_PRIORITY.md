# Plan: prefer control domains in routing hints

## Problem

`identify_command_domain` in `custom_components/sayso/routing.py` returns a
domain hint only when the matched domains are exactly one. A real home exposes
non-control entities that share a friendly name with a control entity:

- `media_player.living_room_tv` and `remote.living_room_tv` are both named `TV`.
- `light.kitchen_light` and `update.kitchen_light` are both named `Kitchen light`.

The command then matches `{media_player, remote}` or `{light, update}`, the tick
count is not one, and the hint is dropped. The conversation agent receives the
full tool schema instead of a narrowed subset, and the served model performs
worse (measured in `docs/PLAN_LIVE_COMMAND_SMOKE.md`).

## Scope

- `custom_components/sayso/routing.py` — add a single, conservative tie-break.
- `tests/test_routing.py` — regression tests for the collisions.
- This plan doc.

No change to tool compilation, validation, execution, traces, or the public
signatures of `identify_command_domain`, `select_schema_for_domain`, or
`select_tools_for_domain`.

## Rule

Define `_CONTROL_DOMAINS` — the domains a Home Assistant Assist/SaySo tool can
act on. When a command matches more than one domain:

- If exactly one matched domain is a control domain, return it (non-control
  matches are satellite/meta signal and are ignored for the hint).
- Otherwise return `None` (two control domains stay ambiguous; no control domain
  stays unknown).

This is applied to both entity/domain-term matching and area/floor evidence so
the two paths stay consistent.

Ambiguity between two control domains (light vs fan, light vs switch) is
unchanged and still returns `None`, preserving the recipe-lock clarification
behavior.

## Files to touch

- `custom_components/sayso/routing.py`
- `tests/test_routing.py`

## TDD unit

One unit: control-domain preference in `identify_command_domain`.

New failing tests first:

1. `media_player.living_room_tv` + `remote.living_room_tv` (both `TV`) →
   `turn on the tv` returns `media_player`.
2. `light.kitchen_light` + `update.kitchen_light` (both `Kitchen light`) →
   `turn on kitchen light` returns `light`.
3. Two control domains (`light` + `fan`) still return `None`.
4. Only non-control domains (`update` + `remote`) still return `None`.
5. Area evidence with `media_player` + `remote` in one area returns
   `media_player`.

Then implement the helper and wire both call sites.

## Verification

1. `SAYSO_COMPAT_VENV_ROOT=/tmp /opt/homebrew/bin/python3 scripts/compat_matrix.py run-tests current`
   (pytest: `tests/test_routing.py` and the compat suite).
2. Live acceptance: re-send `Turn on the TV.`, `Turn off the living room TV.`,
   and `Turn on the TV in the living room.` through the SaySo agent and confirm
   the trace `context_completed` metadata now reports `domain_hint=media_player`
   with a narrowed tool count, and that the resolved tool/target is reported.
