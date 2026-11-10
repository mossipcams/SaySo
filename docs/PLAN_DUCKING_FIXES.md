# Plan: ducking review fixes

Follow-up to PR #181 (and a139728). Branch `ajax/ducking-fixes` off
`origin/main`.

## Defects

1. Duck can raise volume: tier 1 always sets `duck_volume`. Duck to
   `min(level, duck_volume)`; skip the volume tier if `level <= duck_volume`
   (it's already quiet; don't touch it).
2. Restore races the command (user chose "restore first"): restore calls run
   `blocking=True` inside tracked tasks; `MediaDucker.async_wait_restores()`
   awaits pending restores with a ~2 s cap. The conversation agent awaits it
   before executing HA tools, so relative volume commands step from the real
   level. Duck calls stay `blocking=False`.
3. Tier 1 needs `MediaPlayerEntityFeature.VOLUME_SET`, not just a reported
   `volume_level`; otherwise fall through to pause/mute.
4. One ducker per config entry: keep a single shared ducker in
   `hass.data[DOMAIN]`, reference-counted by entries; teardown on the last
   entry unload.
5. HA stop while ducked: restore on `EVENT_HOMEASSISTANT_STOP`.
6. One failing service call drops later restores: per-call `try/except` with
   `_LOGGER.exception`, continue with the rest.

Skipped (minor/cleanup): pause/mute quick re-wake, routing.py area lookup,
`_counts` derivation, per-wake registry scan.

## Files

- `custom_components/sayso/ducking.py`
- `custom_components/sayso/test_ducking.py`
- `custom_components/sayso/__init__.py`
- `custom_components/sayso/conversation.py`

## Verification

- `ruff check .`, `ruff format --check .`
- `pytest custom_components/sayso/test_ducking.py tests/test_init.py tests/test_config_flow.py`
- Full gate compared against an origin/main baseline (local venv has known
  pre-existing failures).
- New tests: quieter player untouched; volume-without-VOLUME_SET falls
  through; two entries share one ducker; stop restores; failing call doesn't
  block other restores; `async_wait_restores` returns after restores land and
  respects the cap.
