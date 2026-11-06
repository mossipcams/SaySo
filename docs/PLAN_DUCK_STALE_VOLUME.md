# Plan: stop ducking from resetting TV volume

## Problem

Live HA (2026-11-05): `media_player.living_room_samsung_tv` (samsungtv, state
`on`) reports a stale `volume_level` of 0.19 that never changes. On every wake
`MediaDucker._duck_plan` picks the volume_set tier, ducks to 0.02, then restores
the stale 0.19, so voice/remote volume changes are undone on the next wake.
`docs/PLAN_DUCKING.md` intended TVs that report `on` to be muted.

## Scope

- `custom_components/sayso/ducking.py`: in `_duck_plan`, use the volume_set
  tier only when `state.state == MediaPlayerState.PLAYING`; otherwise fall
  through to the existing pause/mute tiers.
- `custom_components/sayso/test_ducking.py`: add one test: a player in `on`
  with a `volume_level` and VOLUME_SET|VOLUME_MUTE is muted then unmuted, and
  receives no volume_set calls. Existing tests unchanged.

## Verification

- `pytest custom_components/sayso/test_ducking.py`
- `ruff check .` and `ruff format --check .`
