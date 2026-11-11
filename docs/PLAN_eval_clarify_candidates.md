# Plan: grade clarifications by their candidates, accept redundant domains

## Problem
- `evals/scorer.py:classify` passes any reply that `is_clarification()` accepts. Cases list the devices the question must offer (`expected.clarification.candidates` in `evals/cases/realistic_v3.jsonl`), but `evals/runner.py` builds `Expectation` without them, so they are never checked.
  - On promotion, v11's 99/120 includes 7 clarifications that offer the wrong devices. Example: "switch the bedside lamp off in the bedroom" expects Jamie's / Robin's Bedside Lamp and passed with "Did you mean the Bedroom Desk Fan or the Bedroom Window Shade?".
- `_canonical` compares arguments exactly, so a call that adds a correct `domain` fails. Example: `HassTurnOn(name="Front Entrance Lock", domain=["lock"])` when the case expects only `name`, and the entity's domain is `lock`.

## Change
1. `evals/outcomes.py`: add `clarification_candidates: tuple[str, ...] = ()` to `Expectation`.
2. `evals/runner.py`: fill it from `expected["clarification"]["candidates"]` when present.
3. `evals/scorer.py`:
   - In `classify`, a clarification passes only if `is_clarification(text)` holds and, when candidates are given, every candidate name appears in the reply (casefold substring). Otherwise return `Outcome.WRONG_RESPONSE_TYPE`, as today for a non-question.
   - In `_canonical`, drop a `domain` argument when the call names a single home entity (via the label index) and the domain list is exactly that entity's domain. Apply this to both expected and actual calls. A wrong domain still differs and fails.
4. New test `evals/tests/test_clarification_candidates.py`:
   - wrong-candidate question fails;
   - right-candidate question passes;
   - no candidates keeps today's behaviour;
   - redundant correct domain passes;
   - wrong domain fails.

## Out of scope
Case files, suites, gates, and every existing test file. If an existing test fails because it pins the old behaviour, stop and report; don't edit it.

## Verify
- `pytest evals/tests -q`
- `pytest tests evals/tests --ignore=tests/test_realistic_v3.py -q`
