# 0002. Post-model path shared by production and evals

## Status

Accepted.

## Context

The eval runner scored the raw first completion. Correction retry, contract and argument checks, area drop, grounding and clarify, held-call replay, the "Done." shortcut, and follow-up iterations ran only in production. The promotion suite therefore measured a different pipeline from the one users hit.

Other gaps had the same effect. The eval sent tools as a separate parameter while production had moved them into the system message in 1.17.0. The eval homes had no context-only entities, so a wrong domain such as `sensor` or a target on a person entity was invisible to the gate. Clarifications were accepted on wording alone and never checked against their candidates. The original promotion suite was edge-heavy, with 40 of 120 cases covering exclusion, ambiguity, unavailable, and multi-action, and it used template phrasing.

## Decision

- One HA-free loop lives in `core/post_model.py` (`run_turn`). It owns held replay, correction, contract and argument checks, area drop, grounding and clarify, the "Done." shortcut, and follow-up iterations. `conversation.py` is an adapter. It supplies the complete, validate, and execute callables and maps the outcome back to the same boundary codes, phases, and speech as before.
- Evals run the same loop through the endpoint adapter with a fake executor. Four decisions were accepted on 2026-11-19:
  1. Gate on the final outcome. Report first-pass results beside it.
  2. Treat every eval case as a voice turn, with the satellite area taken from its home fixture.
  3. Validate with the contract check only in eval, because voluptuous is HA-only. Report the gap.
  4. Keep the clarification cue regex. Grounding clarifies use the `grounding_clarify` source flag.
- The eval prompt matches production. Tools are inlined before the Area context block and no separate `tools` key is sent.
- Scorer rules: a clarification must name every candidate. A redundant but correct `domain` is accepted, and a wrong one still fails. Alias and script-equivalent calls match. A scalar `domain` matches a list. Status, satellite-area, and relative-volume rules follow Home Assistant.
- Context gate: `evals/homes/live_home_1.json` is a redacted live home with its context-only entities. The zero-tolerance flags `wrong_domain` and `context_entity_targeted` run in the smoke and promotion gates and in a `context` suite that must pass 100%.
- Promotion suite: exclusion is removed. It has 78 reviewed cases plus 452 generated cases, 530 in total. The generator is seeded and deterministic, uses no LLM, and rejects any expected call that does not validate or resolve to exactly one entity.
- Case integrity: wrong expected calls were corrected, `device_class` was removed from expectations, per-case capabilities were added, and a duplicate case was removed.
- Earlier turns reach the model only after SaySo has asked a clarifying question (`core/history.py`). Otherwise the model repeated the last action on acknowledgements like "Thank you."
- PR #132 satellite review notes (`58e89fa`) are recorded here for completeness. They add a drift guard on the raw wake ring and one function for ring capacity. They are not part of the post-model path.

Rejected:

- Scoring the raw first completion.
- Sending the whole chat history to the model.
- Gating on first-pass results alone.
- Clarification grading by regex alone.
- LLM-written or LLM-judged eval rows.

## Consequences

- Scores measure the final outcome, so they move. Fewer `wrong_target_area` and `action_on_ambiguity` flags are expected. First-pass numbers stay in the output for comparison.
- Eval validation is contract-only. Voluptuous argument errors are not seen by eval. The count of calls that only voluptuous would reject is reported when nonzero.
- Measured at the start of batch 5 with the in-memory stub: smoke 16/24, promotion 441/530. These are stub numbers, not model accuracy.
- At the time of the scorer fixes, `pytest evals/tests` went from 17 failed and 402 passed to 419 passed.
- Promotion scores are not comparable with the v7, v11, or v12 history.
- Not built: the households 6–10 growth in `docs/PLAN_EVAL_SCALE.md`, and the phase 3 real-usage pull in `docs/PLAN_EVAL_NATURAL_LANGUAGE.md`. Both plans remain.

## Shipped in

- `99fa376` feat(evals): match the production prompt and gate on real failing commands
- `d531eaa` fix(evals): prompt the model with the embedded runtime tool layout
- `df49822` feat(evals): gate on Home Assistant context look-alikes
- `2f9276c` fix(evals): correct expected tool calls and fixtures
- `c7703c5` fix(evals): accept alias and script-equivalent calls; fix contradictory cases
- `41efaec` fix(evals): grade clarifications by their candidates and accept redundant domains
- `24dcd08` evals: natural promotion suite with 530 reviewed and generated cases
- `b48303e` refactor: move the correction-message builders into core
- `7ebc407` refactor: move the held-call helpers into core
- `f257c2c` refactor: move the pure tool-contract types into core
- `82f026a` refactor: run the post-model loop through core.post_model
- `1f58f42` feat: report post-model outcome flags from run_turn
- `b2b2c96` feat: score evals on the real post-model path
- `33e1169` refactor: move the HA-free modules into core/ and enforce the boundary
- `5d6a1de` fix(integration): send earlier turns to the model only after a clarifying question
- `58e89fa` fix(satellite): address PR 132 review notes
