# Refetch real-home snapshot

Scope: replace `training/fixtures/real_home.json` with the live Assist-exposed
home at `http://192.168.1.35:8123`. No HA renames, no extra entities, no
generator or recipe edits, no corpus generate, no training launch.

## Files

- `training/fixtures/real_home.json` (overwrite via `training/scripts/fetch_ha_home.py`)

## Verification

- Fetch exit 0 with `--require-entity media_player.living_room_tv`
- `exposure_source` is `assist_exposure`
- TV name `TV`, area `Living Room`; SaySo satellite media player absent
- `cd training && .venv/bin/python -m pytest tests/test_real_home.py tests/test_real_home_mixing.py tests/test_defect_v4_corpus.py -q`
