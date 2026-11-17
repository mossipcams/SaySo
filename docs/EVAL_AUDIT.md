# SaySo evaluation-set audit

## Correction (2026-11-17)

10 of the 12 GetLiveContext 'G' findings are false positives because `evals/scorer.py:_canonical` turns a string domain into a list, so Section 1 has 61 real case IDs, not 71.

- Date: 2026-11-17
- Branch: `ajax/evaluation-set` (worktree `SaySo__worktrees/ajax-evaluation-set`)
- Scope: `evals/` (cases, suites, homes, scorer, runner, outcomes, config) audited against `training/generators/`, `schemas/sayso-tool-schema-v2.json`, and `custom_components/sayso/`. Read-only except this file. No tests, evals, training, or integration code were changed; nothing was committed.
- Reproduce: `PYTHONPATH=/tmp/evalaudit/stub python3 /tmp/evalaudit/build_report.py` (scripts live outside the repo in `/tmp/evalaudit/`; see Appendix A).

## Caveats that bound these findings

1. **Generator provenance.** The case-producing generators cited in `provenance.source` (`training/evals/v3_quality.py`, `training/evals/recipe_lock.py`, `training/evals/grounding_eval.py`) are not in this worktree. They were removed in commit `98ae012` ("consolidate evaluation onto one production-contract pipeline") and remain only in git history. The cases therefore cannot be regenerated or diffed here; they are audited as frozen data.
2. **Training generator is in-repo, not the VM copy.** The generator that actually trained the model runs from `ajax-optimization/sayso-training` on the VM and has diverged from `training/` (per `user memory: sayso-training-runs-from-ajax-optimization`). Section 2 uses the in-repo registry and the manifest `training/datasets/sayso_v3_gauntlet_v2_20260913.manifest.json` (latest manifest, 40,000 rows). Training "would emit" statements describe the in-repo code only.
3. **`evals/archive/` is excluded** from scope. No suite (`evals/suites/*.yaml`) references it; it is referenced only by `evals/tests/test_migration.py` and `tests/test_realistic_v3.py`.
4. **Schema-level checks are not sufficient.** Every expected tool name was first checked against the v2 schema; 9 call names fail a bare schema lookup, but all 9 are per-home script entities (`training/generators/tools.py:script_tools`) and all 9 resolve in their own home (Appendix B). The real defects below are found by comparing against the training builders and home fixtures, not by schema validity alone.

## Summary

- **Cases audited:** 345 (120 `realistic_v3.jsonl`, 209 `regressions.jsonl`, 16 `context_lookalikes.jsonl`), 317 expected calls, 215 home fixtures. All home references resolve; suites and tags are consistent.
- **Wrong tool calls (Section 1): 71 case IDs.** Largest group: **24 expected actions on entities whose fixture `capabilities` lack the feature the registry requires** (e.g. pause on a TV that only reports `on`/`off`). The training gate (`capability_registry.py:entity_supports`) treats the operation as unsupported for that entity, so the trained answer is a refusal, not an action. Next: 12 `GetLiveContext` shapes training never emits, 11 timer args that contradict the utterance, 8 named-timer args outside the training name list, 8 media `device_class` expectations the fixture cannot supply, 7 lock `device_class` expectations training never emits, and 1 expected area absent from its household.
- **Trained but not evaluated (Section 2):** 18 of 59 trained capability-operation pairs have no expected call in eval; the blocked buttons, lawn-mower and todo groups are covered only by refusal cases (labelled refusal-only in Section 2.2). Training families `follow_up` (multi-turn), `correction`, `junk`, and `datetime` (`GetDateTime`) have **0** eval cases. Area-wide and floor-targeted actions have **0** eval cases (0 area-keyed action calls, 0 floor-keyed).
- **Evaluated but not trained (Section 3):** 12 `GetLiveContext` shapes, `device_class` expectations on media and locks, timer names outside the gold name list (3 cases), and categories with no matching family.
- **Structural (Section 4):** 329 of 345 cases cite generator/source paths absent from the tree; the runner duplicates the namespace table, catalog builder, and prompt; scoring is exact-match on arguments including `device_class`, so the 15 `device_class` expectations are unreachable from training output; clarification scoring requires pinned candidate substrings; and 10 utterance clusters (41 IDs) repeat across cases.

## 1. Wrong tool calls

Each row is a case whose expected call does not match (a) the feature set of its own home fixture, (b) the HA tool argument schema in `schemas/sayso-tool-schema-v2.json`, or (c) what the training builder in `training/generators/tools.py` emits for the same entity and operation. Values are the exact expected JSON from the case file. Counts are computed (Appendix B).

| Kind | Meaning | Rows |
|---|---|---|
| A-feature | Expected action on an entity whose fixture `capabilities` lack the feature the registry requires (`capability_registry.py:entity_supports`) | 24 |
| B-device_class-absent | Expected `device_class` the fixture entity does not carry; training emits none (`tools.py:_media_player_args`) | 8 |
| C-lock-device_class | Expected `device_class` on a lock; training `_domain_args` returns `{}` for `lock` | 7 |
| D-named-timer | Expected timer `name` (not an entity); scenario path never sets a name; gold path only from a fixed list (`gold.py:161-170`) | 8 |
| E-timer-args | Expected timer args contradict the utterance (name or area dropped or inconsistent) | 11 |
| F-area-absent | Expected `area` not defined in the case's household | 1 |
| G-live-context-shape | `GetLiveContext` with list `domain` or `name` only; training `build_query` emits `{name, domain: "<string>"}` | 12 |

Shapes checked and found consistent with the training builders (not flagged): `intent__HassTurnOn/Off` with `device_class: ["blind"|"garage"|"door"]` for covers, scenes with `domain: ["scene"]`, scripts as bare object IDs, and `HassClimateSetTemperature {name, temperature}`. These match `tools.py`.

| Kind | File | Case ID | Category | Expected tool | Expected args | Training would emit / reason | Note |
|---|---|---|---|---|---|---|---|
| B-device_class-absent | realistic_v3.jsonl | realistic_eval_20260908_media_01 | media | media_player__HassMediaUnpause | {"device_class": ["tv"], "domain": ["media_player"], "name": "Living Room TV"} | fixture entity has no device_class; training emits {"domain": ["media_player"], "name": "Living Room TV"} | device_class not derivable from home |
| B-device_class-absent | realistic_v3.jsonl | realistic_eval_20260908_media_02 | media | media_player__HassMediaPause | {"device_class": ["tv"], "domain": ["media_player"], "name": "Lounge Television"} | fixture entity has no device_class; training emits {"domain": ["media_player"], "name": "Lounge Television"} | device_class not derivable from home |
| B-device_class-absent | realistic_v3.jsonl | realistic_eval_20260908_media_03 | media | media_player__HassSetVolume | {"device_class": ["tv"], "domain": ["media_player"], "name": "Living Room Television", "volume_level": 30} | fixture entity has no device_class; training emits {"domain": ["media_player"], "name": "Living Room Television"} | device_class not derivable from home |
| B-device_class-absent | realistic_v3.jsonl | realistic_eval_20260908_media_04 | media | media_player__HassMediaPlayerMute | {"device_class": ["tv"], "domain": ["media_player"], "name": "Sofa TV"} | fixture entity has no device_class; training emits {"domain": ["media_player"], "name": "Sofa TV"} | device_class not derivable from home |
| B-device_class-absent | realistic_v3.jsonl | realistic_eval_20260908_media_06 | media | media_player__HassMediaUnpause | {"device_class": ["tv"], "domain": ["media_player"], "name": "Living Room TV"} | fixture entity has no device_class; training emits {"domain": ["media_player"], "name": "Living Room TV"} | device_class not derivable from home |
| B-device_class-absent | realistic_v3.jsonl | realistic_eval_20260908_media_07 | media | media_player__HassMediaPause | {"device_class": ["tv"], "domain": ["media_player"], "name": "Lounge Television"} | fixture entity has no device_class; training emits {"domain": ["media_player"], "name": "Lounge Television"} | device_class not derivable from home |
| B-device_class-absent | realistic_v3.jsonl | realistic_eval_20260908_media_08 | media | media_player__HassSetVolume | {"device_class": ["tv"], "domain": ["media_player"], "name": "Living Room Television", "volume_level": 55} | fixture entity has no device_class; training emits {"domain": ["media_player"], "name": "Living Room Television"} | device_class not derivable from home |
| B-device_class-absent | realistic_v3.jsonl | realistic_eval_20260908_media_09 | media | media_player__HassMediaPlayerMute | {"device_class": ["tv"], "domain": ["media_player"], "name": "Sofa TV"} | fixture entity has no device_class; training emits {"domain": ["media_player"], "name": "Sofa TV"} | device_class not derivable from home |
| C-lock-device_class | regressions.jsonl | recipe_lock_03_b | entity_identity | intent__HassTurnOn | {"device_class": ["door"], "name": "Patio Side Door Lock"} | training _domain_args returns {} for lock; emits {"name": "Patio Side Door Lock"} | exact-match miss on device_class |
| C-lock-device_class | regressions.jsonl | recipe_lock_03_d | entity_identity | intent__HassTurnOff | {"device_class": ["door"], "name": "Joe's Guest Room Door Lock"} | training _domain_args returns {} for lock; emits {"name": "Joe's Guest Room Door Lock"} | exact-match miss on device_class |
| C-lock-device_class | regressions.jsonl | recipe_lock_04_b | multi_action_exclusion | intent__HassTurnOn | {"device_class": ["door"], "name": "Patio Side Door Lock"} | training _domain_args returns {} for lock; emits {"name": "Patio Side Door Lock"} | exact-match miss on device_class |
| C-lock-device_class | regressions.jsonl | recipe_lock_04_d | multi_action_exclusion | intent__HassTurnOn | {"device_class": ["door"], "name": "Patio Side Door Lock"} | training _domain_args returns {} for lock; emits {"name": "Patio Side Door Lock"} | exact-match miss on device_class |
| C-lock-device_class | regressions.jsonl | recipe_lock_05_a | stt_corrupted | intent__HassTurnOff | {"device_class": ["door"], "name": "Basement South Door Lock"} | training _domain_args returns {} for lock; emits {"name": "Basement South Door Lock"} | exact-match miss on device_class |
| C-lock-device_class | regressions.jsonl | recipe_lock_05_d | stt_corrupted | intent__HassTurnOn | {"device_class": ["door"], "name": "Joe's Guest Room Door Lock"} | training _domain_args returns {} for lock; emits {"name": "Joe's Guest Room Door Lock"} | exact-match miss on device_class |
| C-lock-device_class | regressions.jsonl | recipe_lock_07_h | ambiguity | intent__HassTurnOn | {"device_class": ["door"], "name": "Kitchen Back Door Lock"} | training _domain_args returns {} for lock; emits {"name": "Kitchen Back Door Lock"} | exact-match miss on device_class |
| A-feature | regressions.jsonl | v3_quality_gold_media_play | media_play | media_player__HassMediaUnpause | {"device_class": ["tv"], "domain": ["media_player"], "name": "Home Theater TV"} | training gates media_players.play on features ['play']; entity capabilities ['on', 'off'] | refusal / no call (entity cannot perform op) |
| A-feature | regressions.jsonl | v3_quality_gold_media_pause | media_pause | media_player__HassMediaPause | {"device_class": ["tv"], "domain": ["media_player"], "name": "Rec Room TV"} | training gates media_players.pause on features ['pause']; entity capabilities ['on', 'off'] | refusal / no call (entity cannot perform op) |
| A-feature | regressions.jsonl | v3_quality_gold_media_volume | media_volume | media_player__HassSetVolume | {"device_class": ["tv"], "domain": ["media_player"], "name": "Family Room TV", "volume_level": 45} | training gates media_players.volume_set on features ['volume']; entity capabilities ['on', 'off'] | refusal / no call (entity cannot perform op) |
| A-feature | regressions.jsonl | v3_quality_gold_media_volume_rel | media_volume | media_player__HassSetVolumeRelative | {"name": "Guest Suite TV", "volume_step": "up"} | training gates media_players.volume_up on features ['volume_step']; entity capabilities ['on', 'off'] | refusal / no call (entity cannot perform op) |
| A-feature | regressions.jsonl | v3_quality_gold_media_mute | media_mute | media_player__HassMediaPlayerMute | {"device_class": ["tv"], "domain": ["media_player"], "name": "Home Theater TV"} | training gates media_players.mute on features ['mute']; entity capabilities ['on', 'off'] | refusal / no call (entity cannot perform op) |
| A-feature | regressions.jsonl | v3_shadow_media_play_0004 | media_play | media_player__HassMediaUnpause | {"device_class": ["tv"], "domain": ["media_player"], "name": "Conservatory Wall TV"} | training gates media_players.play on features ['play']; entity capabilities ['on', 'off'] | refusal / no call (entity cannot perform op) |
| A-feature | regressions.jsonl | v3_shadow_media_play_0005 | media_play | media_player__HassMediaUnpause | {"device_class": ["tv"], "domain": ["media_player"], "name": "Solarium Wall TV"} | training gates media_players.play on features ['play']; entity capabilities ['on', 'off'] | refusal / no call (entity cannot perform op) |
| A-feature | regressions.jsonl | v3_shadow_media_play_0006 | media_play | media_player__HassMediaUnpause | {"device_class": ["tv"], "domain": ["media_player"], "name": "Studio Wall TV"} | training gates media_players.play on features ['play']; entity capabilities ['on', 'off'] | refusal / no call (entity cannot perform op) |
| A-feature | regressions.jsonl | v3_shadow_media_play_0007 | media_play | media_player__HassMediaUnpause | {"device_class": ["tv"], "domain": ["media_player"], "name": "Terrace Wall TV"} | training gates media_players.play on features ['play']; entity capabilities ['on', 'off'] | refusal / no call (entity cannot perform op) |
| A-feature | regressions.jsonl | v3_shadow_media_pause_0008 | media_pause | media_player__HassMediaPause | {"device_class": ["tv"], "domain": ["media_player"], "name": "Studio Corner TV"} | training gates media_players.pause on features ['pause']; entity capabilities ['on', 'off'] | refusal / no call (entity cannot perform op) |
| A-feature | regressions.jsonl | v3_shadow_media_pause_0009 | media_pause | media_player__HassMediaPause | {"device_class": ["tv"], "domain": ["media_player"], "name": "Terrace Corner TV"} | training gates media_players.pause on features ['pause']; entity capabilities ['on', 'off'] | refusal / no call (entity cannot perform op) |
| A-feature | regressions.jsonl | v3_shadow_media_pause_0010 | media_pause | media_player__HassMediaPause | {"device_class": ["tv"], "domain": ["media_player"], "name": "Balcony Corner TV"} | training gates media_players.pause on features ['pause']; entity capabilities ['on', 'off'] | refusal / no call (entity cannot perform op) |
| A-feature | regressions.jsonl | v3_shadow_media_pause_0011 | media_pause | media_player__HassMediaPause | {"device_class": ["tv"], "domain": ["media_player"], "name": "Cellar Corner TV"} | training gates media_players.pause on features ['pause']; entity capabilities ['on', 'off'] | refusal / no call (entity cannot perform op) |
| A-feature | regressions.jsonl | v3_shadow_media_volume_0012 | media_volume | media_player__HassSetVolume | {"device_class": ["tv"], "domain": ["media_player"], "name": "Atrium Soundbar TV", "volume_level": 25} | training gates media_players.volume_set on features ['volume']; entity capabilities ['on', 'off'] | refusal / no call (entity cannot perform op) |
| A-feature | regressions.jsonl | v3_shadow_media_volume_0013 | media_volume | media_player__HassSetVolumeRelative | {"name": "Conservatory Soundbar TV", "volume_step": "down"} | training gates media_players.volume_up on features ['volume_step']; entity capabilities ['on', 'off'] | refusal / no call (entity cannot perform op) |
| A-feature | regressions.jsonl | v3_shadow_media_volume_0014 | media_volume | media_player__HassSetVolume | {"device_class": ["tv"], "domain": ["media_player"], "name": "Solarium Soundbar TV", "volume_level": 39} | training gates media_players.volume_set on features ['volume']; entity capabilities ['on', 'off'] | refusal / no call (entity cannot perform op) |
| A-feature | regressions.jsonl | v3_shadow_media_volume_0015 | media_volume | media_player__HassSetVolumeRelative | {"name": "Studio Soundbar TV", "volume_step": "down"} | training gates media_players.volume_up on features ['volume_step']; entity capabilities ['on', 'off'] | refusal / no call (entity cannot perform op) |
| A-feature | regressions.jsonl | v3_shadow_media_volume_0016 | media_volume | media_player__HassSetVolume | {"device_class": ["tv"], "domain": ["media_player"], "name": "Terrace Soundbar TV", "volume_level": 53} | training gates media_players.volume_set on features ['volume']; entity capabilities ['on', 'off'] | refusal / no call (entity cannot perform op) |
| A-feature | regressions.jsonl | v3_shadow_media_mute_0017 | media_mute | media_player__HassMediaPlayerMute | {"device_class": ["tv"], "domain": ["media_player"], "name": "Balcony Bedroom TV"} | training gates media_players.mute on features ['mute']; entity capabilities ['on', 'off'] | refusal / no call (entity cannot perform op) |
| A-feature | regressions.jsonl | v3_shadow_media_mute_0018 | media_mute | media_player__HassMediaPlayerMute | {"device_class": ["tv"], "domain": ["media_player"], "name": "Cellar Bedroom TV"} | training gates media_players.mute on features ['mute']; entity capabilities ['on', 'off'] | refusal / no call (entity cannot perform op) |
| A-feature | regressions.jsonl | v3_shadow_media_mute_0019 | media_mute | media_player__HassMediaPlayerMute | {"device_class": ["tv"], "domain": ["media_player"], "name": "Attic Bedroom TV"} | training gates media_players.mute on features ['mute']; entity capabilities ['on', 'off'] | refusal / no call (entity cannot perform op) |
| A-feature | regressions.jsonl | v3_shadow_media_mute_0020 | media_mute | media_player__HassMediaPlayerMute | {"device_class": ["tv"], "domain": ["media_player"], "name": "Porch Bedroom TV"} | training gates media_players.mute on features ['mute']; entity capabilities ['on', 'off'] | refusal / no call (entity cannot perform op) |
| A-feature | regressions.jsonl | v3_shadow_media_mute_0021 | media_mute | media_player__HassMediaPlayerMute | {"device_class": ["tv"], "domain": ["media_player"], "name": "Veranda Bedroom TV"} | training gates media_players.mute on features ['mute']; entity capabilities ['on', 'off'] | refusal / no call (entity cannot perform op) |
| A-feature | regressions.jsonl | sem_e997d4555dc4c991 | ambiguity | media_player__HassSetVolume | {"device_class": ["speaker"], "domain": ["media_player"], "name": "Porch Speaker", "volume_level": 64} | training gates media_players.volume_set on features ['volume']; entity capabilities ['on', 'off', 'play', 'pause'] | refusal / no call (entity cannot perform op) |
| D-named-timer | realistic_v3.jsonl | realistic_eval_20260908_timers_01 | timers | intent__HassStartTimer | {"minutes": 12, "name": "pasta"} | scenario path emits hours\|minutes\|seconds only with no name; gold path adds name from fixed list (gold.py:161-170) | name 'pasta' in gold name list |
| D-named-timer | realistic_v3.jsonl | realistic_eval_20260908_timers_02 | timers | intent__HassPauseTimer | {"name": "tea"} | scenario path emits {} with no name; gold path adds name from fixed list (gold.py:161-170) | name 'tea' in gold name list |
| D-named-timer | realistic_v3.jsonl | realistic_eval_20260908_timers_03 | timers | intent__HassTimerStatus | {"name": "laundry"} | scenario path emits {} with no name; gold path adds name from fixed list (gold.py:161-170) | name 'laundry' NOT in gold name list |
| D-named-timer | realistic_v3.jsonl | realistic_eval_20260908_timers_05 | timers | intent__HassStartTimer | {"minutes": 45, "name": "bread"} | scenario path emits hours\|minutes\|seconds only with no name; gold path adds name from fixed list (gold.py:161-170) | name 'bread' in gold name list |
| D-named-timer | realistic_v3.jsonl | realistic_eval_20260908_timers_06 | timers | intent__HassStartTimer | {"hours": 2, "name": "laundry"} | scenario path emits hours\|minutes\|seconds only with no name; gold path adds name from fixed list (gold.py:161-170) | name 'laundry' NOT in gold name list |
| D-named-timer | realistic_v3.jsonl | realistic_eval_20260908_timers_07 | timers | intent__HassPauseTimer | {"name": "pasta"} | scenario path emits {} with no name; gold path adds name from fixed list (gold.py:161-170) | name 'pasta' in gold name list |
| D-named-timer | realistic_v3.jsonl | realistic_eval_20260908_timers_08 | timers | intent__HassTimerStatus | {"name": "bread"} | scenario path emits {} with no name; gold path adds name from fixed list (gold.py:161-170) | name 'bread' in gold name list |
| D-named-timer | realistic_v3.jsonl | realistic_eval_20260908_timers_10 | timers | intent__HassStartTimer | {"name": "stretch", "seconds": 20} | scenario path emits hours\|minutes\|seconds only with no name; gold path adds name from fixed list (gold.py:161-170) | name 'stretch' NOT in gold name list |
| E-timer-args | regressions.jsonl | v3_shadow_timer_pause_0027 | timer_pause | intent__HassPauseTimer | {} | utterance names timer 'Terrace' but expected args are empty; training builder emits {} | expected args contradict utterance |
| E-timer-args | regressions.jsonl | v3_shadow_timer_pause_0028 | timer_pause | intent__HassPauseTimer | {} | utterance names timer 'Balcony' but expected args are empty; training builder emits {} | expected args contradict utterance |
| E-timer-args | regressions.jsonl | v3_shadow_timer_pause_0029 | timer_pause | intent__HassPauseTimer | {} | utterance names timer 'Cellar' but expected args are empty; training builder emits {} | expected args contradict utterance |
| E-timer-args | regressions.jsonl | v3_shadow_timer_pause_0030 | timer_pause | intent__HassPauseTimer | {} | utterance names timer 'Attic' but expected args are empty; training builder emits {} | expected args contradict utterance |
| E-timer-args | regressions.jsonl | v3_shadow_timer_status_0031 | timer_status | intent__HassTimerStatus | {} | utterance names timer 'Cellar' but expected args are empty; training builder emits {} | expected args contradict utterance |
| E-timer-args | regressions.jsonl | v3_shadow_timer_status_0032 | timer_status | intent__HassTimerStatus | {} | utterance names timer 'Attic' but expected args are empty; training builder emits {} | expected args contradict utterance |
| E-timer-args | regressions.jsonl | v3_shadow_timer_status_0033 | timer_status | intent__HassTimerStatus | {} | utterance names timer 'Porch' but expected args are empty; training builder emits {} | expected args contradict utterance |
| E-timer-args | regressions.jsonl | v3_shadow_timer_status_0034 | timer_status | intent__HassTimerStatus | {} | utterance names timer 'Veranda' but expected args are empty; training builder emits {} | expected args contradict utterance |
| E-timer-args | regressions.jsonl | v3_shadow_timer_cancel_0035 | timer_cancel | intent__HassCancelAllTimers | {} | utterance says 'in the Attic' but expected args are empty; training builder emits {} | expected args contradict utterance |
| E-timer-args | regressions.jsonl | v3_shadow_timer_cancel_0037 | timer_cancel | intent__HassCancelAllTimers | {} | utterance says 'in the Veranda' but expected args are empty; training builder emits {} | expected args contradict utterance |
| E-timer-args | regressions.jsonl | v3_shadow_timer_cancel_0039 | timer_cancel | intent__HassCancelAllTimers | {} | utterance says 'in the Annex' but expected args are empty; training builder emits {} | expected args contradict utterance |
| F-area-absent | regressions.jsonl | v3_shadow_timer_cancel_0038 | timer_cancel | intent__HassCancelAllTimers | {"area": "Courtyard"} | area 'Courtyard' not in household v3_shadow_timer_cancel_0038 | target cannot resolve |
| G-live-context-shape | regressions.jsonl | recipe_lock_06_a | status | homeassistant__GetLiveContext | {"name": "Patio South Blinds"} | training build_query emits {"domain": "cover", "name": "Patio South Blinds"} (domain as string) | list-domain or name-only shape never trained |
| G-live-context-shape | regressions.jsonl | recipe_lock_06_b | status | homeassistant__GetLiveContext | {"domain": ["fan"], "name": "Workshop West Fan"} | training build_query emits {"domain": "fan", "name": "Workshop West Fan"} (domain as string) | list-domain or name-only shape never trained |
| G-live-context-shape | regressions.jsonl | recipe_lock_06_c | status | homeassistant__GetLiveContext | {"name": "Joe's Guest Room Door Lock"} | training build_query emits {"domain": "lock", "name": "Joe's Guest Room Door Lock"} (domain as string) | list-domain or name-only shape never trained |
| G-live-context-shape | regressions.jsonl | recipe_lock_06_d | status | homeassistant__GetLiveContext | {"domain": ["light"], "name": "Kitchen North Light"} | training build_query emits {"domain": "light", "name": "Kitchen North Light"} (domain as string) | list-domain or name-only shape never trained |
| G-live-context-shape | regressions.jsonl | v3_quality_gold_status_media | status | homeassistant__GetLiveContext | {"domain": ["media_player"], "name": "Home Theater TV"} | training build_query emits {"domain": "media_player", "name": "Home Theater TV"} (domain as string) | list-domain or name-only shape never trained |
| G-live-context-shape | regressions.jsonl | v3_quality_gold_status_climate | status | homeassistant__GetLiveContext | {"domain": ["climate"], "name": "Great Room Thermostat"} | training build_query emits {"domain": "climate", "name": "Great Room Thermostat"} (domain as string) | list-domain or name-only shape never trained |
| G-live-context-shape | regressions.jsonl | v3_shadow_status_0085 | status | homeassistant__GetLiveContext | {"domain": ["media_player"], "name": "Atrium Lounge TV"} | training build_query emits {"domain": "media_player", "name": "Atrium Lounge TV"} (domain as string) | list-domain or name-only shape never trained |
| G-live-context-shape | regressions.jsonl | v3_shadow_status_0086 | status | homeassistant__GetLiveContext | {"domain": ["climate"], "name": "Conservatory Thermostat"} | training build_query emits {"domain": "climate", "name": "Conservatory Thermostat"} (domain as string) | list-domain or name-only shape never trained |
| G-live-context-shape | regressions.jsonl | v3_shadow_status_0087 | status | homeassistant__GetLiveContext | {"domain": ["media_player"], "name": "Solarium Lounge TV"} | training build_query emits {"domain": "media_player", "name": "Solarium Lounge TV"} (domain as string) | list-domain or name-only shape never trained |
| G-live-context-shape | regressions.jsonl | v3_shadow_status_0088 | status | homeassistant__GetLiveContext | {"domain": ["climate"], "name": "Studio Thermostat"} | training build_query emits {"domain": "climate", "name": "Studio Thermostat"} (domain as string) | list-domain or name-only shape never trained |
| G-live-context-shape | regressions.jsonl | v3_shadow_status_0089 | status | homeassistant__GetLiveContext | {"domain": ["media_player"], "name": "Terrace Lounge TV"} | training build_query emits {"domain": "media_player", "name": "Terrace Lounge TV"} (domain as string) | list-domain or name-only shape never trained |
| G-live-context-shape | context_lookalikes.jsonl | context_lookalikes_11 | status | homeassistant__GetLiveContext | {"domain": ["media_player"], "name": "TV"} | training build_query emits {"domain": "media_player", "name": "TV"} (domain as string) | list-domain or name-only shape never trained |

### 1.1 Notes on the table

- **A-feature (24 rows, 24 cases, 24 homes).** All are in `regressions.jsonl` and use the `v3_gold_media_*`, `v3_shadow_media_*`, and `grounding_porch_speaker_incapable` homes, which declare `capabilities`. The realistic homes have no `capabilities` key (5 of 5 realistic households), so the realistic media cases cannot be feature-checked at all. `sem_e997d4555dc4c991` is the clearest: the case expects `HassSetVolume` on a speaker whose fixture lists `on, off, play, pause` only.
- **B (8 cases, 4 households).** `realistic_v3.jsonl` media cases expect `"device_class": ["tv"]`, but the fixture TV entity has no `device_class` key at all. Training builds media args with `device_class` only when the entity carries one, so training output for these requests omits it. The scorer requires an exact argument match (`evals/scorer.py:_canonical`), so a correct model fails.
- **C (7 cases).** Lock expectations carry `device_class: ["door"]` while `tools.py:_domain_args` returns `{}` for `lock`. Training never emits it.
- **D (8 cases).** `realistic_v3.jsonl` timer cases name timers that are not entities (laundry, stretch, etc.). The gold path emits names only from `gold.py:161-170` (12 fixed names). Names outside that list: laundry, stretch (3 cases).
- **E (11 rows).** The same request shape gets different expectations. `realistic_v3` timer cases expect `name` on pause and status; `v3_shadow_timer_pause_0027..0030` and `v3_shadow_timer_status_0031..0034` expect `{}` for "Hold the Terrace countdown timer" and "How much time is left on the Cellar timer". `v3_shadow_timer_cancel_0035/0037/0039` ("in the Attic/Veranda/Annex") expect `{}`, while `..._0036` ("in the Porch") and `v3_quality_gold_timer_cancel_named` ("in the Sunroom") expect `area`. These cannot all be right.
- **F.** `v3_shadow_timer_cancel_0038` expects area `Courtyard`; household `v3_shadow_timer_cancel_0038` defines no such area.
- **G (12 cases).** Ten have `domain` as a list (`recipe_lock_06_b`, `06_d`, `v3_quality_gold_status_media`, `v3_quality_gold_status_climate`, `v3_shadow_status_0085..0089`, `context_lookalikes_11`). Two (`recipe_lock_06_a`, `06_c`) carry only `name`. Training's `build_query` always emits `domain` as a single string with `name`.
- **Script calls (not wrong).** 9 calls use bare script object IDs. All 9 resolve to a script entity in their own household, matching `tools.py:script_tools`. The schema lookup flags them only because they are per-home tools.
- **Semantic direction check.** A verb-to-tool rule check over all cases (lock/unlock, open/close, mute/unmute, pause/resume, timers) found no direction error. The rule check produced false positives (for example, "volume … percent" matched the brightness rule), and every hit was reviewed by hand. No direction errors remain.

## 2. Trained but not evaluated

### 2.1 Training families and robustness modes

Family counts are case counts from the mapped eval categories (both files). `0` means no eval case exercises the family. Training generator functions are cited by file and symbol.

| Training family / robustness | Generator (in-repo) | Eval cases | Eval categories mapped |
|---|---|---|---|
| ordinary | scenarios/core.py:build_scenario (robustness ordinary, pick_robustness rate 0.75) | 36 | ordinary, ordinary_on, ordinary_off, clean_direct |
| follow_up (multi-turn) | planning.py:_pick_capability_operation (follow_up/clarify branch); row_generation.py:_follow_up_reply | 0 | none |
| status | planning.py family status; scenarios/core.py query_state | 22 | status |
| settings | planning.py family settings; SETTINGS_OPERATIONS | 38 | light_fan_settings, light_brightness, fan_speed, climate_setpoint, media_volume |
| clarify | planning.py:_pick_capability_operation (clarify branch); gold.py:_ambiguous_gold | 62 | ambiguity |
| correction | correction.py:wrong_name_tool_call / validation_error_message | 0 | none |
| junk (no-call non-command) | row_generation.py:_junk_utterance/_junk_text | 0 | none |
| multi_action | scenarios/core.py (targeting multiple); area.py:_multi | 14 | multi_action, multi_action_exclusion |
| exclusion | scenarios/area.py:_exclusion; core.py exclusion robustness | 10 | exclusion |
| unavailable | scenarios/unavailable.py | 10 | unavailable |
| unsupported | capability_registry.py UNAVAILABLE ops; planning family unsupported | 14 | unsupported_no_action |
| absence | planning.py family absence (robustness ambiguity) | 3 | unavailable.kind device_absent/script_absent |
| aliases | scenarios/core.py:_usable_aliases (alias_distractor) | 11 | aliases, alias_distractor |
| datetime (GetDateTime) | row_generation.py:_datetime_utterance; planning.py:datetime_slot | 0 | llm__GetDateTime calls |
| stt_noise | stt_noise.py (stt_noise_rate 0.15) | 6 | stt_corrupted |
| discrimination / lookalike | scenarios/discrimination.py:pick_discriminating_description | 16 | context_lookalikes.jsonl (all) |

Other trained behaviours with no eval case:

- **Targeting modes.** `pick_targeting` (`scenarios/core.py:477`) and `gold.py` produce `area`, `floor`, and `multiple`. Eval has 0 area-keyed action calls outside timer-cancel and vacuum-clean-area, and 0 floor-keyed calls. `scenarios/area.py` builds the `area` family (`_area_wide`, `_multi`, `_exclusion`, `_unresolved`, `_conflict`); eval has no area-wide action case (see the count above).
- **Robustness modes.** `similar_name` and `large_home` (`capability_registry.py:DIFFICULTY_TAGS`) have no eval category or tag. `stt_noise` is covered by 6 `stt_corrupted` cases only. `alias_distractor` has 1 regression case.
- **Multi-turn.** `row_generation.py:_follow_up_reply` produces the user's reply to a clarification. The eval case schema is single-turn (`evals/cases.py:REQUIRED_CASE_KEYS` has no turns field), so the clarify-then-act path is untested.
- **Outcome mix.** Training manifest (`quota.achieved.outcome`, 40000 rows): action 28157 (70.4%), status 8704 (21.8%), clarify 302 (0.76%), unsupported+absence 2837 (7.1%). Eval (345 cases): action+status 292 (84.6%), clarification 22 (6.4%), refusal 31 (9.0%). Eval clarification share is 8.4x the manifest clarify share.
- **Rejected rows.** The 2026-09-13 manifest rejected 2050 rows for `missing_multiple_actions` (`rejection_reasons`). The `multi_action` rows (513) therefore understate the attempted multi-action share.

### 2.2 Capability-operation coverage

Training rows are capability-scoped positive plus negative rows from `quota.achieved.operation`. Eval coverage is computed by mapping each expected call to a (capability, operation) pair from its tool name, domain and device_class (`/tmp/evalaudit/build_report.py:infer`). Blocked operations (`control`, `set_position`, script `query_state`) can only appear as refusals, so their eval coverage is taken from utterance regexes.

| Capability.operation (training) | Training rows (quota.achieved.operation, pos+neg) | Eval coverage | Gap |
|---|---|---|---|
| climate.query_state | 963 | 4 calls / 4 cases | covered |
| climate.set_temperature | 964 | 16 calls / 16 cases | covered |
| climate.turn_off | 964 | 1 calls / 1 cases | covered |
| climate.turn_on | 964 | 0 calls / 0 cases | GAP |
| covers.close | 733 | 5 calls / 5 cases | covered |
| covers.open | 733 | 6 calls / 6 cases | covered |
| covers.query_state | 732 | 2 calls / 2 cases | covered |
| covers.set_position | 115 | 0 calls / 0 cases | GAP |
| fans.query_state | 674 | 2 calls / 2 cases | covered |
| fans.set_speed | 674 | 10 calls / 10 cases | covered |
| fans.turn_off | 675 | 11 calls / 11 cases | covered |
| fans.turn_on | 675 | 4 calls / 4 cases | covered |
| lights.query_state | 1413 | 4 calls / 4 cases | covered |
| lights.set_brightness | 1414 | 17 calls / 15 cases | covered |
| lights.set_color | 1413 | 6 calls / 4 cases | covered |
| lights.set_color_temperature | 1413 | 2 calls / 2 cases | covered |
| lights.turn_off | 1414 | 20 calls / 16 cases | covered |
| lights.turn_on | 1414 | 33 calls / 31 cases | covered |
| locks.lock | 644 | 8 calls / 8 cases | covered |
| locks.query_state | 644 | 2 calls / 2 cases | covered |
| locks.unlock | 644 | 3 calls / 3 cases | covered |
| media_players.mute | 415 | 8 calls / 8 cases | covered |
| media_players.next_track | 415 | 0 calls / 0 cases | GAP |
| media_players.pause | 415 | 7 calls / 7 cases | covered |
| media_players.play | 415 | 7 calls / 7 cases | covered |
| media_players.previous_track | 415 | 0 calls / 0 cases | GAP |
| media_players.query_state | 415 | 6 calls / 6 cases | covered |
| media_players.search_and_play | 415 | 0 calls / 0 cases | GAP |
| media_players.turn_off | 416 | 9 calls / 9 cases | covered |
| media_players.turn_on | 416 | 20 calls / 20 cases | covered |
| media_players.unmute | 415 | 0 calls / 0 cases | GAP |
| media_players.volume_down | 415 | 3 calls / 3 cases | covered |
| media_players.volume_set | 415 | 8 calls / 8 cases | covered |
| media_players.volume_up | 415 | 2 calls / 2 cases | covered |
| switches.query_state | 1156 | 1 calls / 1 cases | covered |
| switches.turn_off | 1156 | 6 calls / 6 cases | covered |
| switches.turn_on | 1157 | 7 calls / 7 cases | covered |
| timers.cancel | 482 | 0 calls / 0 cases | GAP |
| timers.cancel_all | 482 | 9 calls / 9 cases | covered |
| timers.decrease | 482 | 0 calls / 0 cases | GAP |
| timers.increase | 482 | 0 calls / 0 cases | GAP |
| timers.pause | 482 | 7 calls / 7 cases | covered |
| timers.start | 482 | 11 calls / 11 cases | covered |
| timers.status | 481 | 7 calls / 7 cases | covered |
| timers.unpause | 482 | 0 calls / 0 cases | GAP |
| scenes.activate | 1000 | 9 calls / 9 cases | covered |
| scenes.query_state | 1000 | 0 calls / 0 cases | GAP |
| scripts.query_state | 100 | 0 calls / 0 cases | GAP |
| scripts.run | 1900 | 9 calls / 9 cases | covered |
| vacuums.clean_area | 500 | 8 calls / 8 cases | covered |
| vacuums.query_state | 500 | 1 calls / 1 cases | covered |
| vacuums.return_home | 500 | 8 calls / 8 cases | covered |
| vacuums.start | 500 | 8 calls / 8 cases | covered |
| buttons.control | 33 | 0 calls; 2 refusal cases match 'button\|panic' | refusal-only |
| buttons.query_state | 635 | 0 calls; 2 refusal cases match 'button\|panic' | refusal-only |
| lawn_mowers.control | 33 | 0 calls; 3 refusal cases match 'lawn\|mow' | refusal-only |
| lawn_mowers.query_state | 633 | 0 calls; 3 refusal cases match 'lawn\|mow' | refusal-only |
| todo_lists.control | 33 | 0 calls; 3 refusal cases match 'shopping list\|to-do\|milk' | refusal-only |
| todo_lists.query_state | 633 | 0 calls; 3 refusal cases match 'shopping list\|to-do\|milk' | refusal-only |

Gaps (no expected call in eval): 18 of 59 pairs; refusal-only groups are counted separately. Eval-only (capability, operation) keys with no trained counterpart: none (41 distinct eval keys, all of which map to trained pairs).

## 3. Evaluated but not trained

- **GetLiveContext shapes** (12 calls, Section 1 kind G): list-valued `domain` (10) and `name`-only (2). Training never emits either.
- **device_class on media players and locks** (15 calls, Section 1 kinds B and C). The eval requires an argument the trained builders do not produce for those entities.
- **Named timers outside the training name list** (3 cases: laundry, stretch). Training's gold path draws from `GOLD_NAMES` (`gold.py:161-170`). The scenario path never names a timer. Named cases in eval are therefore either unreachable from training or only reachable via a closed list.
- **Named timers on pause/status** (`realistic_v3` timers_02, 03, 07, 08) expect a `name` argument that the scenario path never emits. The gold path adds one from its list, so the expected value is only partly reachable.
- **Area-scoped timer cancel.** `v3_quality_gold_timer_cancel_named` and `v3_shadow_timer_cancel_0036` expect `area`. Training emits `area` for cancel-all only through the gold path (`gold.py:163-166`).
- **Eval categories with no matching training family or op:** conversational, entity_identity. These need the VM generator or `training/evals/*.py` (removed) to confirm; in-repo training has no family of that name.
- **Refusal for ambiguity:** 9 cases in category `ambiguity` expect `refusal` (not clarify). Training has `absence -> ambiguity` robustness (`planning.py:FAMILY_ROBUSTNESS`), so refusal under `ambiguity` is trained only through `absence`; the category label is misleading (Section 4).
- **Multi-turn and correction.** No eval case exercises `follow_up` or `correction` (Section 2.1).

## 4. Structural problems


1. **Provenance points at missing generators.** 329 of 345 cases cite a source that does not exist in this tree:
   - `evals/cases/realistic_v3.json`: 120 cases
   - `training/evals/v3_quality.py:build_shadow_specs`: 100 cases
   - `training/evals/recipe_lock.py`: 38 cases
   - `training/evals/grounding_eval.py`: 36 cases
   - `training/evals/v3_quality.py:gold_specs`: 35 cases
   The `realistic_v3.json` path is also a stale name (the file is `realistic_v3.jsonl`). The `training/evals/` generators were deleted in `98ae012` and survive only in history. Cases cannot be regenerated or checked against their generator, so provenance is advisory only.
2. **Runner duplicates production-adjacent logic.** `evals/runner.py` copies `HA_TOOL_NAMESPACES` (lines ~78-106), the catalog builder (`_compile_catalog`, `production_catalog`, `_script_tools`), and `namespaced_tool_name`, instead of importing `training/generators/tools.py`. The copies match today (27/27 tools, identical values), but drift is silent.
3. **Eval prompt differs from the production prompt.** `evals/runner.py:45` `SAYSO_SYSTEM_PROMPT` matches `custom_components/sayso/const.py:78-79` (`DEFAULT_SYSTEM_PROMPT`). The runner additionally appends `DYNAMIC_CONTEXT_PROMPT` (`runner.py:48`, applied at line 207) and `DEVICE_CONTROL_TOOL_USAGE_PROMPT` (`runner.py:70`, applied at line 215). Neither text appears anywhere in `custom_components/`; both live only in `training/generators/context.py:20-23` and in the runner. Eval therefore measures a prompt the integration does not send.
4. **Exact-match argument scoring makes some expectations unreachable.** `evals/scorer.py:_canonical` normalises only `name` (to entity ID) and drops `domain` when it equals the entity domain. `device_class`, `volume_level`, `percentage`, `brightness` and `temperature` are compared exactly. The 15 `device_class` expectations (Section 1 B, C) therefore fail for any model trained on `training/generators/tools.py` output.
5. **Pinned numeric values not stated in the utterance.** realistic_eval_20260908_light_fan_settings_08 expects `percentage: 25` for "a quarter of its full speed"; the value is an inference, not stated.
6. **Clarification scoring requires pinned candidate text.** `scorer.py:classify` passes a clarification only if every candidate string appears verbatim (case-insensitive) in the reply. 10 of 22 clarification cases pin candidates. A correct clarification that paraphrases a candidate fails.
7. **Refusal and clarification detection are regex cues.** `scorer.py:_REFUSAL_CUES` and `_CLARIFICATION_CUES` decide `refusal`/`clarification` outcomes. A refusal sentence that contains a clarification cue and a "?" is scored as a clarification, and a clarification without "?" is scored as wrong response type. No case tests the cue behaviour itself.
8. **Category/response mismatches.** `ambiguity` expects `refusal` in 9 cases (for example `sem_681432e5293fbf31`, `v3_shadow_ambiguity_0092`). `unsupported_no_action` expects `clarification` in 2 cases (recipe_lock_08_clarify, v3_quality_gold_no_call_clarify). `v3_quality_gold_timer_start_named` (category `timer_start`, id says "named") has no name in its expected call.
9. **Duplicate utterances across cases.** 10 normalised utterance clusters cover 41 IDs (for example "turn on the tv" in 8 `sem_*`/`context` cases). Most are intentional household variants (`issue52_*`, `grounding_*`), so the same text has different verdicts by home. Training exclusion (`evals/cases.py:excluded_train_utterances`) is text-based, so it removes every variant's text from training even when only one variant was meant to be held out.
10. **Feature gating applies only to some homes.** Only the v3 and grounding homes carry `capabilities`; 5 of 5 realistic households do not, so the `realistic_v3` media cases are never feature-checked. All Section 1 A findings are therefore confined to `regressions.jsonl`, the only file whose homes declare `capabilities` for media players.
11. **Gates are uncalibrated and category-keyed.** `evals/config/gates.yaml:2` says "not yet calibrated against a trained checkpoint". Promotion gates list the 12 realistic categories, so regression categories are not gated. The context suite (16 cases) requires 100% pass. None of its cases appears in Section 1, so the findings above do not change that gate's inputs, but the 100% threshold has not been calibrated.
12. **Smoke suite description is stale.** `evals/suites/smoke.yaml` describes "area" coverage, but there is no area category; smoke has 1 timers case and 3 ambiguity cases of 24.
13. **Schema-level validator false positives.** A bare schema lookup flags 9 script calls as unknown tools (`evals/scorer.py:classify`, `UNKNOWN_TOOL` outcome). These are per-home script tools and do resolve; any future schema-only check would break them.
14. **Archive is not referenced by suites.** `evals/archive/` is referenced only by `evals/tests/test_migration.py` and `tests/test_realistic_v3.py`. No suite includes archived cases.
15. **Eval does not track the newer training families.** `training/configs/generation/production.yaml` allocates none of `follow_up`, `correction`, or `junk`; `full_sft_v4.yaml` adds `junk 0.06` only; `full_sft_v5.yaml` adds `follow_up 0.13`, `correction 0.06`, and `junk 0.06`. No eval family exists for any of them (Section 2.1).

## Appendix A. Commands and scripts

All scripts are in `/tmp/evalaudit/` (outside the repo). Read-only against the repo; writes only `docs/EVAL_AUDIT.md`.

| Script | Purpose |
|---|---|
| `/tmp/evalaudit/gen_loader.py` | Loads `training/generators/` without the package `__init__` (which needs the full training venv). |
| `/tmp/evalaudit/stub/yaml.py`, `/tmp/evalaudit/stub/jsonschema.py` | Import stubs. `yaml` is only used by `GeneratorConfig.from_yaml`; `jsonschema` only by validation. Neither is called by the shape/builder code. |
| `/tmp/evalaudit/check_calls.py` | Namespace parity (training vs runner vs schema) and per-call schema checks. |
| `/tmp/evalaudit/run_cases.py` | Loads all cases, counts categories, response types, and tool use. |
| `/tmp/evalaudit/train_shapes.py` | Calls `tools.build_call_for_operation` for every registry op to get training argument shapes. |
| `/tmp/evalaudit/home_check.py` | Checks names, domains, device classes, areas, scripts, forbidden entities and clarify candidates against each case's home. |
| `/tmp/evalaudit/vs_training.py` | Compares expected args with the trained builder output for the same entity. |
| `/tmp/evalaudit/features.py` | Checks expected actions against `capability_registry.entity_supports`. |
| `/tmp/evalaudit/suites.py` | Suite membership, tags, and archive references. |
| `/tmp/evalaudit/coverage.py` | Utterance checks and capability-operation coverage. |
| `/tmp/evalaudit/build_report.py` | Computes every number in this report and writes `docs/EVAL_AUDIT.md`. |

Reproduce the report: `PYTHONPATH=/tmp/evalaudit/stub python3 /tmp/evalaudit/build_report.py`.

## Appendix B. Computed counts

Output of `build_report.py` at generation time:

```text
cases_total=345 by_file={"context_lookalikes.jsonl": 16, "realistic_v3.jsonl": 120, "regressions.jsonl": 209}
expected_calls=317 schema_tools=27 namespaces_match=True
scripts_total=9 scripts_absent_from_home=0
homes_total=215 homes_with_capabilities=210 realistic_households_without_capabilities=5
feature_checked_actions=42 feature_violations=24 distinct_cases=24 homes=24
section1_kind_counts={"A-feature": 24, "B-device_class-absent": 8, "C-lock-device_class": 7, "D-named-timer": 8, "E-timer-args": 11, "F-area-absent": 1, "G-live-context-shape": 12} distinct_case_ids=71
named_timer_calls_realistic=8 named_timer_names_outside_gold_list=['laundry', 'stretch']
area_keyed_action_calls_excl_timer_cancel_and_vacuum=0 floor_keyed_calls=0
getdatetime_calls=0
suite_sizes smoke=24 promotion=120 context=16 smoke_not_in_promotion=0 promotion_tag_mismatch=0 smoke_tag_mismatch=0
provenance_missing_cases=329 by_source={"evals/cases/realistic_v3.json": 120, "training/evals/grounding_eval.py": 36, "training/evals/recipe_lock.py": 38, "training/evals/v3_quality.py:build_shadow_specs": 100, "training/evals/v3_quality.py:gold_specs": 35}
utterance_clusters_repeated=10 ids_involved=41
ambiguity_cases_expecting_refusal=9 unsupported_cases_expecting_clarification=2
clarification_cases=22 with_pinned_candidates=10
training_manifest rows=40000 outcome={"absence": 948, "action": 28157, "clarify": 302, "status": 8704, "unsupported": 1889}
training_families_by_difficulty={"alias_distractor": 1552, "ambiguity": 2441, "exclusion": 404, "large_home": 1528, "multi_action": 513, "ordinary": 30506, "similar_name": 1521, "unsupported": 1535}
ops_pairs=59 ops_gaps=18
```

Per-row data for Section 1 is the table above; Section 2.2 is generated from the same `infer` mapping.

### B.1 Notes on the computation

- The tool-name namespace check (`HA_TOOL_NAMESPACES`) and the schema check both pass for all 27 tools; the 9 script calls fail a bare lookup only because they are per-home script IDs.
- Entity lookup uses name or alias with dedupe on entity identity. An earlier pass counted an entity twice when its alias equalled its name, which hid 17 feature violations; the count above is after that fix.
- Training row counts are from the 2026-09-13 manifest only. The 2026-09-11 manifest has the same operation mix.

