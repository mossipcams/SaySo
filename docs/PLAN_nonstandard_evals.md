# Plan: nonstandard voice command evals

## Scope

Add a held-out eval set for the 24-family nonstandard command list (2026-10-20).
Generator work is planned separately in ajax-optimization
`sayso-training/docs/PLAN_NONSTANDARD_GENERATOR.md`.

The runner is single-turn today: `render_case` sends `[system, user]`. Ten of the
families depend on earlier turns or tool results: timer ambiguity, failure, and
corrections; contextual references; target corrections; restore; relative
brightness; and conditionals. Production already sends the HA chat log as
history (`transcript.chat_log_to_messages`), so the eval needs the same shape.

## Changes

1. `evals/cases.py`: accept two optional case keys. Each is a list of
   OpenAI-format chat messages (`user`, `assistant` with optional `tool_calls`,
   `tool`).
   - `history`: messages before the utterance.
   - `after_utterance`: the assistant tool call and tool result after the
     utterance, used to score the model's next turn (for example, a reply
     after a timer error).
   If a key is present, validate it as a list of objects with a known `role`.
   Cases without these keys render exactly as before.
2. `evals/runner.py`: `render_case` builds
   `[system, *history, user(utterance), *after_utterance]`. Area context still
   uses `case.utterance`.
3. `evals/tests/test_history.py`: rendering order, unchanged rendering for
   existing cases, and rejection of a malformed history.
4. Data:
   - `evals/cases/nonstandard_v1.jsonl` (54 cases, tag `nonstandard`, plus
     `p0`/`p1`/`p2` and `nonstandard_family_NN`). Family 4 cases were converted
     from absolute satellite volume to relative satellite steps (see Labelling
     conventions).
   - `evals/homes/nonstandard_home_1.json` (Kitchen satellite with its own
     `Kitchen Satellite` speaker, one TV). Each entity has an `entity_id`, used as
     the `id` in intent success results.
   - `evals/homes/nonstandard_home_2.json` (Living Room satellite, two TVs).
   - Add the case file to `_CASE_FILES`.

No suite or gate is added. Run the set as a diagnostic with
`python -m evals.cli run --tag nonstandard ...`. Promotion and smoke suites are
unchanged.

## Labelling conventions

- Volume by amount: `HassSetVolumeRelative.volume_step` integer (`10`, `-5`) for
  the TV, or direction `"up"` / `"down"` for either device.
- Satellite self-volume is relative only (decided 2026-10-20). Users say "turn it
  down" or "speak louder", never a number, so the satellite is labelled only with
  `HassSetVolumeRelative` `volume_step` `"up"` / `"down"`. Hedged magnitudes
  ("a little") still map to the direction. No absolute `HassSetVolume` is labelled
  on the satellite. TV-vs-satellite disambiguation (family 5) is kept.
- "Make it 15 minutes" right after starting a 10-minute timer: `HassIncreaseTimer`
  `minutes: 5`.
- Target correction ("No, I meant the bedroom TV"): act on the new target only.
  Do not undo the earlier action unless the user asks.
- Relative brightness without a known current value: read first
  (`GetLiveContext`, scored as `status`). When a current value is in history, the
  expected brightness follows `brightness_pct` (the percent, for example `20`),
  not the raw `brightness` (0–255).
- Unsupported families (remote navigation, app switching, cover position):
  `refusal` until an executable tool exists, then relabel.
- Tool results follow the shapes verified against a live Home Assistant 2026.10.2
  (verified 2026-10-21):
  - Intent success: `{"speech": {}, "response_type": "action_done", "data":
    {"success": [{"name": <entity name>, "type": "entity", "id": <entity_id>}],
    "failed": []}}`, with no speech text and no `targets` key. Timer intents have
    no entity, so `success` is `[]`.
  - GetLiveContext: `{"result": "Live Context: An overview of the areas and the
    devices in this smart home:\n<yaml>"}`, with no `success` key. Light entries
    carry `brightness` (string, 0–255) and `brightness_pct` (string,
    `round(brightness / 255 * 100)`, minimum 1 when brightness is above 0).
    Media player `volume_level` stays a float on a 0–1 scale.
  - Intent errors: `{"error": ..., "error_text": ...}`, for example
    `TimerNotFoundError` / `Timer not found` and `MultipleTimersMatchedError` /
    `Multiple timers matched`.

Family 19 (assistant speech controls) is not included. "Say that again" is a
plain spoken reply with no matching response type, and "stop talking" is
handled by the satellite.

## Verification

- `python -m pytest evals/tests -q`. `evals/tests/test_nonstandard_fixtures.py`
  checks the verified tool-result shapes, the satellite relative-only rule, and
  the brightness labels.
- Every new case parses, and every expected or history call passes
  `tool_contract.check_tool_calls` against `production_catalog(home)`.
- An ideal response for each case scores `passed`.
- Existing promotion and smoke suite hashes are unchanged.
