# Plan: eval tool catalog from HA 2026.11.1

Source of truth: `docs/HA_2026_11_TOOLS.md` (HA core tag 2026.11.1, read from a
sparse clone at `/tmp/ha-core-2026.11.1`, commit d8a0dc6). Production SaySo
passes HA's Assist API tools to the model unchanged; the eval must offer the
same list per home.

## Scope

1. Snapshot: `schemas/ha-2026.11.1-tools.json` with every Assist API tool HA
   2026.11.1 can offer: final tool name (`<domain>__<intent>` or custom name),
   description as HA declares it, parameters (types, enums, required) as HA
   declares them, and the offer condition.
2. Runner: `evals/runner.py` builds each home's catalog from the snapshot using
   HA's list-time rules:
   - always: `intent__HassTurnOn/Off`, `intent__HassCancelAllTimers`,
     `llm__GetDateTime`, `homeassistant__GetLiveContext`,
     `assist_satellite__HassBroadcast`
   - timer intents: offered, because every eval home's satellite stands in for
     the timer-capable satellite (production LVA/ESPHome satellite)
   - per exposed domain: media_player, climate, fan, light, vacuum, lawn_mower,
     humidifier, calendar, todo; `intent__HassSetPosition/StopMoving` for
     cover or valve
   - `media_player__search_media` and `media_player__play_media` only if an
     exposed media player lists both `search_media` and `play_media` in its
     `capabilities`
   - scripts: `script__<object_id>`
   - no `HassMediaSearchAndPlay`
3. Cases: shopping-list refusals become `todo__HassListAddItem` (with a todo
   entity in the home); script expectations get the `script__` prefix; add two
   cases each for search_media (first call), broadcast, and cover position.

Out of scope: training generator (separate step), scorer logic, prompt text.

## Slices

- Slice A (snapshot + runner + catalog test): `schemas/ha-2026.11.1-tools.json`,
  `evals/runner.py`, `evals/tests/test_ha_catalog.py`, plus any evals module that
  reads the v2 schema path. Existing tests that pin the old catalog: stop and
  report.
- Slice B (cases + homes): `evals/cases/*.jsonl`, `evals/homes/*.json`.
- Slice C: gate, branch `ajax/ha-tool-catalog` from `ajax/evaluation-set`,
  commit, push, PR with base `ajax/evaluation-set`.

## Verification

- `pytest evals/tests -q` green, including `test_case_tool_calls.py` (every
  expected call exists in the home's catalog) and the new catalog test.
- `ruff check .`, `ruff format --check .`, `pytest -q satellite/sayso`.
- `tests/` runs in CI (local Python is 3.12; HA needs 3.13.2).

## Results: Slice B

Changed expected calls (case id: old -> new):

- `realistic_eval_20260908_routines_vacuum_05`: `welcome_home` -> `script__welcome_home`
- `realistic_eval_20260908_routines_vacuum_10`: `evening_cleanup` -> `script__evening_cleanup`
- `v3_quality_gold_script_a`: `good_morning_script` -> `script__good_morning_script`
- `v3_quality_gold_script_b`: `leave_home_script` -> `script__leave_home_script`
- `v3_shadow_script_0060`: `conservatory_away_script` -> `script__conservatory_away_script`
- `v3_shadow_script_0061`: `solarium_away_script` -> `script__solarium_away_script`
- `v3_shadow_script_0062`: `studio_away_script` -> `script__studio_away_script`
- `v3_shadow_script_0063`: `terrace_away_script` -> `script__terrace_away_script`
- `v3_shadow_script_0064`: `balcony_away_script` -> `script__balcony_away_script`
- `realistic_eval_20260908_unavailable_02` ("add oat milk to Shopping List, please."): response `refusal`, no calls, `unavailable` capability block -> response `action`, `todo__HassListAddItem {"item": "oat milk", "name": "Shopping List"}`, `unavailable` null
- `v3_quality_gold_unsupported_todo` ("Add milk to the shopping list"): `refusal` -> `action`, `todo__HassListAddItem {"item": "milk", "name": "Shopping List"}`
- `v3_shadow_unsupported_0097` ("add batteries to the Cellar shopping list"): `refusal` -> `action`, `todo__HassListAddItem {"item": "batteries", "name": "Cellar Shopping List"}`

Category label now stale (category kept unchanged, to be revisited):
`realistic_eval_20260908_unavailable_02`, `v3_quality_gold_unsupported_todo`, `v3_shadow_unsupported_0097`
(all still `unsupported`/`unavailable`, but now expect an action).

New cases (appended to `evals/cases/regressions.jsonl`, tags `["regression", "ha-2026.11"]`):

| Case id | Utterance | Expected first call | Home |
| --- | --- | --- | --- |
| `ha2611_media_search_01` | Play Stranger Things on the Living Room TV | `media_player__search_media {"search_query": "Stranger Things", "player_name": "Living Room TV"}` | `ha2611_media_search_01` |
| `ha2611_media_search_02` | Search for Jazz Essentials on the Kitchen Speaker | `media_player__search_media {"search_query": "Jazz Essentials", "player_name": "Kitchen Speaker"}` | `ha2611_media_search_02` |
| `ha2611_broadcast_01` | Broadcast dinner is ready | `assist_satellite__HassBroadcast {"message": "dinner is ready"}` | `ha2611_broadcast_01` |
| `ha2611_broadcast_02` | Broadcast the bins need to go out tonight | `assist_satellite__HassBroadcast {"message": "the bins need to go out tonight"}` | `ha2611_broadcast_02` |
| `ha2611_cover_position_01` | Set the Living Room Blinds to 40 percent | `intent__HassSetPosition {"name": "Living Room Blinds", "position": 40}` | `ha2611_cover_position_01` |
| `ha2611_cover_position_02` | Set the Bedroom Curtains to 75 percent | `intent__HassSetPosition {"name": "Bedroom Curtains", "position": 75}` | `ha2611_cover_position_02` |

Homes created (`evals/homes/`):

- `ha2611_media_search_01.json`, `ha2611_media_search_02.json`: one media player with `search_media` and `play_media` capabilities.
- `ha2611_broadcast_01.json`, `ha2611_broadcast_02.json`: one light (the satellite area is the home area).
- `ha2611_cover_position_01.json`, `ha2611_cover_position_02.json`: one cover.

Homes changed (`evals/homes/`):

- `v3_gold_unsupported_todo.json`: added `todo.shopping_list` ("Shopping List", Sunroom).
- `v3_shadow_unsupported_0097.json`: added `todo.cellar_shopping_list` ("Cellar Shopping List", Cellar).

`realistic_eval_household_2.json` already exposes "Shopping List" (todo, Kitchen), so it is unchanged.

Runner: `HA_TOOL_NAMESPACES` keeps only the three bare names still used by prompt rewriting
(`GetLiveContext`, `HassTurnOff`, `HassTurnOn`). The `HassMediaSearchAndPlay` entry and the other
v2 bare-name entries are removed; the snapshot names are already namespaced.

`evals/tests/migration_map.json` is not changed: no test requires the new cases to appear in it.
