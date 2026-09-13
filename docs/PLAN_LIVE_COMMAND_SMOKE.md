# Plan: live end-to-end command smoke test

## Goal

Prove SaySo's conversation agent actually drives real Home Assistant actions for
every command family it advertises, through the production `custom_components/sayso`
integration — not by calling llama.cpp directly.

## Scope

- New repo helper: `scripts/live_command_smoke.py` (stdlib only, same style as
  `training/scripts/fetch_ha_home.py`).
- Reads `HA_URL` and `HA_TOKEN` from the environment. Never logs the token.
- Discovers the SaySo conversation entity from `/api/states` when `SAYSO_AGENT`
  is unset (all `conversation.*` entities except `conversation.home_assistant`).
- Discovers real exposed entity names from `/api/states` to build utterances that
  target real devices, so a pass means a real action, not a no-op.
- Runs one turn per HA LLM tool family (the 27 tools in
  `schemas/sayso-tool-schema-v2.json`) via `POST /api/conversation/process`.
- After each turn, pulls the newest SaySo trace with
  `POST /api/services/sayso/list_traces?return_response` +
  `POST /api/services/sayso/get_trace?return_response` and reports the resolved
  tool, target, domain, success, and error stage/type.
- Skips tool families whose required domain is absent from the home, with an
  explicit SKIP reason.
- Also exercises the two SaySo services directly (`sayso.get_trace`,
  `sayso.list_traces`).

## Files to touch

- `scripts/live_command_smoke.py` (new)
- `docs/PLAN_LIVE_COMMAND_SMOKE.md` (this file)

No runtime integration changes. No test-framework changes.

## Command matrix (HA LLM tools)

Ordered so timer lifecycle runs start → status → mutate → cancel.

| Tool | Utterance | Requires |
|------|-----------|----------|
| GetDateTime | What time is it? | — |
| GetLiveContext | What is the state of {target}? | any controllable |
| HassTurnOn | Turn on {target} | light/switch/fan/media_player/cover/lock |
| HassTurnOff | Turn off {target} | same |
| HassLightSet | Set {target} to 50 percent | light |
| HassFanSetSpeed | Set {target} to 50 percent | fan |
| HassClimateSetTemperature | Set {target} to 70 degrees | climate |
| HassSetVolume | Set the volume to 30 percent | media_player |
| HassSetVolumeRelative | Turn the volume up | media_player |
| HassMediaNext | Next track | media_player |
| HassMediaPrevious | Previous track | media_player |
| HassMediaPause | Pause the music | media_player |
| HassMediaUnpause | Resume the music | media_player |
| HassMediaPlayerMute | Mute the TV | media_player |
| HassMediaPlayerUnmute | Unmute the TV | media_player |
| HassMediaSearchAndPlay | Play some jazz | media_player |
| HassVacuumStart | Start the vacuum | vacuum |
| HassVacuumReturnToBase | Send the vacuum back to its base | vacuum |
| HassVacuumCleanArea | Vacuum the kitchen | vacuum |
| HassStartTimer | Set a 5 minute timer | — |
| HassTimerStatus | How much time is left on my timer? | — |
| HassPauseTimer | Pause my timer | — |
| HassUnpauseTimer | Resume my timer | — |
| HassIncreaseTimer | Add 1 minute to my timer | — |
| HassDecreaseTimer | Remove 1 minute from my timer | — |
| HassCancelTimer | Cancel my timer | — |
| HassCancelAllTimers | Cancel all my timers | — |

## Pass / fail definition

- PASS: the turn returned speech, the trace recorded a tool call in the expected
  family, `summary.success is not False`, and no boundary error stage.
- FAIL: any execution error (`tool_execution_failed`), schema/argument boundary
  failure, empty response, or no tool call where a control command was expected.
- SKIP: required domain not present in the home.

Default is lenient on tool choice (any successful tool for query-ish commands);
`--strict` requires the exact tool name.

## TDD unit

This is a verification harness, not integration behavior. The testable unit is
the pure command-matrix + response-parsing logic: add a focused pytest module
that feeds recorded HA conversation/trace payloads into the parser and asserts
PASS/FAIL/SKIP classification, with no network. Live execution is the manual
verification step.

## Verification steps

1. `python scripts/live_command_smoke.py --dry-run` prints the full matrix and
   the discovered-agent logic without network access.
2. `python -m pytest tests/test_live_command_smoke.py -q` passes (parser unit).
3. With a real token:
   `HA_URL=http://192.168.1.35:8123 HA_TOKEN=... python scripts/live_command_smoke.py`
   runs the matrix and prints a PASS/FAIL/SKIP table plus the newest trace per
   command. `--token-file PATH` reads the secret from a file instead of the
   environment, and `--only HassTurnOn,HassTurnOff` runs a cautious subset first.
4. Inspect `--json-out` results file for the raw response text and resolved tool
   call per command.

## Result (2026-09-13, live run)

Served agent: `conversation.srv_models_lfm2_5_230m_q8_0_gguf_192_168_1_140_8080`
(model `v3-40k`, LFM2.5-230M-Q8_0 @ 192.168.1.140:8080), Home Assistant 2026.9.2.

**1 PASS / 23 FAIL / 3 SKIP.** The only pass was `HassMediaPlayerUnmute`
("Unmute the TV"), which resolved `media_player.living_room_tv` and executed.

Findings:

- The integration pipeline works end to end: schema compile, tool validation,
  HA execution, error mapping, and `sayso.get_trace`/`list_traces` all function.
  Failures surface as trace `error_stage`/`error_type` (`ha_action_failed`,
  `schema_mismatch`, `iteration_limit`).
- Root cause is the served model, not the wiring. Recorded in
  `training/TRAINING_LOG.md` (Run 008): `v3-40k` ep1 scores 9/38 on the
  recipe-lock gold set and memorized its 2,124-name training vocabulary instead
  of learning to copy names, so a real home's unseen names cause refusals or
  garbled slugs.
- Routing hint regression: exposed `update.kitchen_light` and `update.nightstand`
  share friendly names with the lights, so `identify_command_domain` matches two
  domains and returns `None`, sending all 25 tools instead of a narrowed subset.
- HA 2026.9 namespaces Assist tools (`intent__HassTurnOff`); the pinned training
  contract is HA 2026.8.3 plain names. Runtime schemas keep the namespaced form
  (commit `40fd013`), an out-of-distribution input for the trained model.
- Safety: "Set the kitchen light to 50 percent" produced a `GetDateTime` call and
  the spoken reply "Done." with no action taken (false confirmation).
- Traces intentionally store no raw model text or tool arguments (README), so
  the resolved tool/target and the spoken response are the available evidence.

