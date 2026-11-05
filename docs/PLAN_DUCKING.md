# Plan: media player ducking on wake

## Goal

While an `assist_satellite` is `listening`, turn down the media players in the
satellite's area so STT hears the command, then put them back.

Lives in the integration (only HA can reach the TV), wired from `__init__.py`
next to `vad_sensitivity.py`. Not in the conversation agent: listening is over
by the time it runs.

## Behavior

- Trigger: `async_track_state_change_event` on every `assist_satellite.*`
  entity. Enter `listening` -> duck. Leave `listening` -> release.
- Targets: `media_player` entities whose area (entity area, else device area)
  equals the satellite's area, state not in off/standby/unavailable/unknown,
  excluding players on the satellite's own device (device ID match).
- Area lookup: extract one helper from `_satellite_area_name` in
  `conversation.py` (entity area, falling back to device area) and use it in
  both places. `_satellite_area_name` keeps its signature.
- Duck, tiered per player (first match wins):
  1. Has `volume_level` -> save original, `media_player.volume_set` to the
     configured duck level; restore the original.
  2. Supports `PAUSE` and state is `playing` (Roku apps) -> `media_pause`;
     restore with `media_play`.
  3. Supports `VOLUME_MUTE` and not already muted (Roku live TV / HDMI, which
     report `on` rather than `playing`) -> `volume_mute` true; restore false.
  4. Otherwise skip.
  All calls `blocking=False`.
- Overlap safety: one shared `{entity_id: original}` map plus a per-entity
  reference count. Record the original only when the entity is not already in
  the map. Each satellite remembers which entities it holds. Restore an entity
  when its count reaches zero.
- Stuck volume: a ~30 s timer per satellite releases its hold if it never
  leaves `listening`. Unload (`entry.async_on_unload`) cancels timers and
  restores everything.
- Config: `duck_volume` option (0.0-1.0, default 0.02) in the options flow.

## Roku note

Roku reports no `volume_level`, so its original level cannot be read back.
"Step to zero then up N" can't be undone. Roku pauses when playing, and
mutes otherwise.

## Files

Slice 1 (logic + test):
- `custom_components/sayso/ducking.py` (new)
- `custom_components/sayso/test_ducking.py` (new, colocated)
- `custom_components/sayso/conversation.py` (area helper extraction)
- `custom_components/sayso/const.py` (`CONF_DUCK_VOLUME`, `DEFAULT_DUCK_VOLUME`, timeout)

Slice 2 (wiring + option):
- `custom_components/sayso/__init__.py`
- `custom_components/sayso/config_flow.py`
- `custom_components/sayso/strings.json`
- `custom_components/sayso/translations/en.json`

## Verification

- `ruff check .` and `ruff format --check .`
- `pytest tests evals/tests --ignore=tests/test_realistic_v3.py`
- `pytest custom_components/sayso/test_ducking.py`
- Test cases: duck + restore; second wake while ducked keeps the true original;
  two satellites, restore only after both leave; timeout restore; unload
  restore; mute fallback; satellite's own player excluded; other-area players
  untouched.
