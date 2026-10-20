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
     `p0`/`p1`/`p2` and `nonstandard_family_NN`).
   - `evals/homes/nonstandard_home_1.json` (Kitchen satellite with its own
     `Kitchen Satellite` speaker, one TV).
   - `evals/homes/nonstandard_home_2.json` (Living Room satellite, two TVs).
   - Add the case file to `_CASE_FILES`.

No suite or gate is added. Run the set as a diagnostic with
`python -m evals.cli run --tag nonstandard ...`. Promotion and smoke suites are
unchanged.

## Labelling conventions

- Volume by amount: `HassSetVolumeRelative.volume_step` integer (`10`, `-5`).
- "Make it 15 minutes" right after starting a 10-minute timer: `HassIncreaseTimer`
  `minutes: 5`.
- Target correction ("No, I meant the bedroom TV"): act on the new target only.
  Do not undo the earlier action unless the user asks.
- Relative brightness without a known current value: read first
  (`GetLiveContext`, scored as `status`).
- Unsupported families (remote navigation, app switching, cover position):
  `refusal` until an executable tool exists, then relabel.
- Tool results follow HA shapes: `IntentResponse.as_dict()` for intents;
  `{"success": true, "result": "Live Context: ..."}` for GetLiveContext, with raw
  attribute scales (`brightness` 0–255, `volume_level` 0–1); and
  `{"error": ..., "error_text": ...}` for intent errors. Verify these against a
  live trace before relabelling.

Family 19 (assistant speech controls) is not included. "Say that again" is a
plain spoken reply with no matching response type, and "stop talking" is
handled by the satellite.

## Verification

- `python -m pytest evals/tests -q`
- Every new case parses, and every expected or history call passes
  `tool_contract.check_tool_calls` against `production_catalog(home)`.
- An ideal response for each case scores `passed`.
- Existing promotion and smoke suite hashes are unchanged.
