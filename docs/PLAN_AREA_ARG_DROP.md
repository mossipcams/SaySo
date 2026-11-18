# Plan: drop unspoken areas from named-device calls

## Problem

`area_context.py:84` renders `target_area: <satellite area>` whenever the
utterance names no area, including commands for a named device in
another room ("Mute the TV" from the Kitchen satellite). If the model
copies that hint into the call (`name=Sofa TV, area=Kitchen`):

- grounding accepts it, because `is_grounded` returns on the name match
  and never checks `area` (`grounding.py:231-238`);
- Home Assistant finds no "Sofa TV" in the Kitchen and the turn fails
  with "not found"/"action failed"; there is no retry without the area.

Home Assistant itself only uses the satellite area to break ties
(`docs/HA_2026_11_TOOLS.md:192`), so an area-only call must keep its
area. Changing the prompt would move the model off its training
distribution (area block layout and values), so the fix goes in
pre-execution validation instead.

## Rule

Before execution, for each validated call:

- the call has a string `name` that resolves (case-insensitive, by
  entity name or alias) to exactly one grounding entity, and
- the call has a string `area` that is not that entity's area, and
- that area was not spoken in the utterance (area name or alias phrase;
  the satellite area does not count as spoken)

then remove `area` from the call's arguments. Every other call is left
unchanged, including area-only calls, calls whose name is ambiguous or
unknown, calls whose area matches the entity's area, and calls whose
area the user said out loud (those still fail or clarify as today).

Record dropped areas in the parse span metadata so traces show it.

## Scope

- `custom_components/sayso/grounding.py`: one pure function implementing
  the rule over `GroundingHome`.
- `custom_components/sayso/conversation.py`: apply it in the existing
  voice-turn grounding block in `_async_run_tool_calls`, before
  `is_grounded`, so the executed call uses the normalized arguments.
- `custom_components/sayso/test_grounding.py`: unit tests.
- New colocated `custom_components/sayso/test_area_arg_drop.py` only if
  an execution-level test fits the existing conversation test helpers;
  otherwise unit tests only, reported.

Not touched: `area_context.py`, prompt text, training, evals, `tests/`,
`ARCHITECTURE.md` (no ownership or boundary change).

## Verification

- Unit: satellite area dropped for a unique named device elsewhere;
  kept for area-only calls; kept when equal to the entity area; kept
  when the user said that area; kept when the name is ambiguous or
  unknown; aliases resolve.
- Execution-level (if feasible): "Mute the TV" from a Kitchen satellite
  with model call `name=Sofa TV, area=Kitchen` reaches HA without
  `area`.
- CI gate: `ruff check .`, `ruff format --check .`,
  `pytest tests evals/tests --ignore=tests/test_realistic_v3.py`,
  `pytest -q custom_components/sayso`, `pytest -q satellite/sayso`.
