# Named clarify: “Did you mean X or Y?”

Scope: when gold is genuine ambiguity (`kind=no_action`, `response=clarify`)
and two or more supporting devices match, the assistant text must name those
devices. No new family, no allocation change, no multi-turn history, no eval
edits, no 40k generate, no training launch.

## Why

v4 already requests `clarify: 0.07`. Those rows currently render
“Which device did you mean?” and never say the candidates. Area-only
`duplicate_name_clarification` already names rooms; family clarify does not.
`similar_name` robustness is listed in `pick_robustness` and has no handler.

## Files

- `training/generators/gold.py` — attach `candidates` on ambiguous clarify gold
- `training/generators/rendering.py` — `Did you mean the {A} or the {B}?`
- `training/tests/test_canonical_generator.py` — no tool calls; names in the reply

## Out of scope

- Second-turn “the kitchen one” follow-ups (training-plan 6% history bucket)
- Wiring `similar_name` robustness
- Raising `clarify` share, real-home growth, HA aliases

## Verification

```bash
cd training && .venv/bin/python -m pytest \
  tests/test_canonical_generator.py::test_clarify_family_never_labels_action \
  tests/test_canonical_generator.py::test_clarify_assistant_names_candidates \
  -q
```
