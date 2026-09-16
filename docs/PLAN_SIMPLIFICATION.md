# SaySo Simplification Plan

Goal: dramatic LOC reduction, god files split, duplicated helpers unified,
if-chains replaced by tables, without deleting tests, weakening assertions, or
removing any behaviour AGENTS.md says to keep.

## Non-negotiables

- No test deleted, no assertion weakened. `tests/` untouched.
- Pre-execution tool validation, fail-closed safety/ambiguity/capability
  barriers, HA tool execution, verification, metric scoring and the offline
  eval path all survive unchanged in behaviour.
- The 38 recipe-lock golden cases and the shadow eval keep producing
  byte-identical JSONL. Verified by snapshotting generated output before and
  after each change.
- Test suite must end at or better than baseline.

## Baseline (measured)

| | lines |
|---|---|
| all `.py` (excl. `.venv`, `.git`, caches) | 47,156 |
| non-test | 27,831 |
| test | 19,325 |

Suite baseline: `tests` + `satellite/sayso` -> 6 failed, 434 passed, 3 errors.
`training` -> 9 failed, 382 passed, 1 error. All pre-existing.

## Phase 1 - Home Assistant integration (`custom_components/sayso`, 6,619)

1. Split `conversation.py` (1,100, one 385-line method).
   - `conversation.py` keeps the entity and the turn loop.
   - `boundary.py` gets batch prevalidation, boundary diagnostics mapping,
     correction-message building, failure->code mapping.
   - `messages.py` gets chat-log serialisation, error results, area prompt.
   - The two near-identical ~55-line correction blocks collapse into one
     helper; the five repeated `except (SaySoTimeoutError, ...)` tuples become
     one module constant.
   - Delete unreferenced `_batch_validation_failure_code`,
     `_validate_tool_call_batch`.
2. `routing.py`: area and floor matching are the same function twice; so are
   the three `_*_by_id` helpers and the two `_resolve_preferred_*_id`. Unify.
3. `client.py`: one transport wrapper instead of three copies of the same
   `try/except TimeoutError/ServerTimeoutError/ClientError`; one JSON-envelope
   reader instead of two; share tool-call parsing with `inference.py`.
4. `config_flow.py`: `_options_schema` (144 lines of repeated
   `vol.Optional(...)` blocks) becomes a field table.
5. `schema.py`: one compile path instead of `compile_tool` and
   `_build_compiled_tools_from_source` doing the same normalise/canonicalise
   dance.

## Phase 2 - training corpus tooling (`training`, 20,700)

6. New `training/evals/specs.py` holding the spec vocabulary duplicated between
   `recipe_lock.py`, `v3_quality.py` and `build_synthetic_dataset.py`
   (`_entity`, `_home`, `_action`, `_no_action`, `_status`, `_turn_on`,
   `_turn_off`, `_light_set`, `_spec`, `score_quality_gold`,
   `assert_quality_eval_contract`).
7. The locked rows are data, not code. `locked_specs` (388 lines) and
   `gold_specs` (372 lines) move to JSON beside their modules, read by a small
   loader. Output verified byte-identical.
8. `build_synthetic_dataset.py` (2,045) is two unrelated pipelines in one file.
   Split the legacy LLM verbaliser/judge path out; drop its local copies of
   `render_example`, `validate_utterance`, `validate_spec`, `_final_text`,
   `expand_utterance` in favour of the `generators/` originals.
9. `generators/tools.py`: 30 near-identical `build_*` functions become one
   table-driven builder.
10. `generators/capability_registry.py`: `_*_ops` builders become a data table.

## Phase 3 - satellite and scripts

11. Shared `_percentile` / `read_wav` instead of three copies.
12. `scripts/generate_sayso_tool_schema.py` re-uses `schema.canonicalize_schema`
    instead of reimplementing it.
13. `wake/streaming.py` and `wake/mining.py` `demo()` blocks trimmed to the
    runnable check they are.

## Verification per phase

- `.venv/bin/python -m pytest -q` (tests + satellite)
- `cd training && ../.venv/bin/python -m pytest -q tests evals scripts generators`
- Golden-output diff for every generated eval/dataset artifact touched.
