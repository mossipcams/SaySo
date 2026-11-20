# Plan: Fix 13, evals run the real post-model path

Parent: `docs/PLAN_review_fixes.md` Fix 13 (Batch 2, after Fix 14, which is
already on this branch: `sayso_contract.py` is an importlib loader, lines 1-39).
Scope of this plan: no code changes. Line numbers are against the current
tree. The review's cites were off by one in places; corrected cites are below.

## Goal

`evals/runner.py` today scores the raw first completion, so the 530-case
promotion suite and the 24-case smoke suite never exercise SaySo's post-model
logic. The goal is one HA-free function in `custom_components/sayso/core/`
that runs the post-model loop (correction retry, contract and argument
validation, area drop, grounding and clarify, held-call replay, the "Done."
shortcut, follow-up iterations). Production calls it with real Home Assistant
seams. The evals call it with the llama.cpp HTTP adapter and a fake executor.
Scores are then measured on what SaySo would actually say and do.

Non-goals: no new eval cases, no new gates, no change to the prompt layout the
evals send, no change to `ARCHITECTURE.md` (ownership and invariants do not
change; core stays inside the integration), no grammar work (Fix 10).

## Current flow

Production, `custom_components/sayso/conversation.py`:

| # | Step | Lines | HA dependency |
|---|---|---|---|
| 1 | Create `_Turn`, open SAYSO_REQUEST span | 406-428 | `conversation.ConversationInput`, `TraceContext` (`tracing.py:104`) |
| 2 | Provide LLM data, compile tools | 434-447 | `chat_log.async_provide_llm_data`, `chat_log.llm_api` (`llm.Tool`), `compile_llm_tools` (`schema.py:365`) |
| 3 | Domain hint, area context, exposed domains | 455-480 | `hass` registries (`ar`, `er`, `dr`), `hass.states`, `async_should_expose` |
| 4 | Messages property: `model_history` then `apply_area_context` | 129-134, 481 | `chat_log.content` via `transcript.py:29` (imports HA, `transcript.py:6`) |
| 5 | Held-call replay (before the model) | 487-518 | `self._held` keyed by `conversation_id`; `time.monotonic` (490); `_is_voice_turn` (290-291); `_resolve_held_option` (294-317) |
| 6 | First completion via `_Turn.complete` | 520-523, 213-245 | `engine.async_chat_completion`; trace INFERENCE span |
| 7 | Model error mapping | 185-211, 522-523 | `diagnostics` (`BoundaryFailureCode`, `record_boundary`) |
| 8 | No tool calls: text or empty | 525-533 | `chat_log.async_add_assistant_content_without_tools` (154) |
| 9 | Tool loop entry: `llm_api is None` | 543-548 | `chat_log.llm_api` |
| 10 | Available names, compiled tools, `tool_map` | 550-554 | `build_tool_availability_names`, `build_tool_map` (`schema.py:258-262`) |
| 11 | Well-formed batch, unknown or unavailable tool | 565-572 | `boundary.py:118` (imports HA, `boundary.py:6`) |
| 12 | Contract check on compiled dicts | 574-576 | `_contract_failures` (824-847): pure, uses `core/tool_contract.py:59` |
| 13 | Argument validation (voluptuous) | 579-580 | `validate_arguments` (`boundary.py:127-152`) calls `validate_tool_arguments` (`schema.py:324-346`) on `llm.Tool.parameters` |
| 14 | Correction retry | 582-620 | `build_correction_messages` (`transcript.py:61`); `turn.complete(correction=True)`; `correction_used` (612) |
| 15 | Voice-only grounding setup | 560, 622-627 | `_grounding_home` (320-355) reads `hass` registries and states |
| 16 | Area drop | 628-640 | pure: `drop_unspoken_area` (`core/grounding.py:133`) |
| 17 | Grounding and clarify, held store | 641-653 | pure: `is_grounded` (`core/grounding.py:246`), `clarify` (`:275`); held store (646-652) |
| 18 | HA_ACTION span open | 655-657 | `action_metadata` (`boundary.py:184`) uses `llm.Tool.name` |
| 19 | Execute | 658-660, 247-287 | `async_wait_for_restores` (255, `ducking.py:313`); `chat_log.async_add_assistant_content` (272) runs the HA tools |
| 20 | Result parsing | 273-284 | `tool_result_failed`, `tool_result_data`, `first_target` (`boundary.py:155-175`) |
| 21 | Span summary | 661-662 | `apply_action_summary` (`boundary.py:205`) |
| 22 | Batch failure speech | 664-677, 737-809 | `execution_failure_speech` reads `ar` and `hass.states` (797-805) |
| 23 | "Done." shortcut | 679-683 | `_is_action_tool` (850-854) needs `llm.ActionTool` / `llm.IntentTool` |
| 24 | Follow-up completion | 685-691 | `turn.messages` again (from `chat_log`) |
| 25 | Follow-up tool calls, iteration cap | 693-702 | `max_tool_iterations` (`runtime`); `ERROR_TOOL_ITERATION_LIMIT` (`const.py:98`) |
| 26 | Empty follow-up, or text | 704-710 | `ERROR_EMPTY_RESPONSE` (`const.py:92`) |

Evals, `evals/runner.py`: `render_case` (284-308) builds messages, compiled
tools, `exposed_domains`, and area context. `evaluate` (324-349) builds the
prompt with `tools_before_area_messages` (326-334; tools are sent only if the
layout is unchanged), calls `adapter.complete` once, calls
`read_completion` (`evals/outcomes.py:96-117`, which runs the contract check
for `violations`), then `score` (`evals/scorer.py:384`). Nothing executes,
nothing retries, no follow-up call, no grounding (`render_case` never makes a
turn voice; `_is_voice_turn` needs `satellite_id` or `device_id`,
`conversation.py:290-291`).

Existing seams and HA-free pieces already in `core/`: `ToolCall` and
`ChatCompletionResult` (`core/completion.py:16-33`, re-exported by
`client.py:21-22`); `check_tool_call(s)` and `tools_by_name`
(`core/tool_contract.py:59-115`); grounding (`core/grounding.py`, imports only
`vocabulary`); `apply_area_context` (`core/area_context.py:109`);
`model_history` (`core/history.py:20`); `GroundingHome` and `GroundingEntity`
(`core/grounding.py:44-59`); the exception classes (`core/exceptions.py`).

HA-bound pieces the core must not import:

- `schema.py:9-10` (voluptuous and HA): holds `ToolArgumentFailureCode`,
  `ToolArgumentValidationError` (`schema.py:246-256`) and
  `format_synthetic_validation_error` (`schema.py:349-362`). Those are pure
  and move to core (slice 1).
- `transcript.py:6`: holds `_tool_call_message` (16-26) and
  `build_correction_messages` (61-100), which are pure and move to core
  (slice 2). `chat_log_to_messages` (29-58) stays in the adapter.
- `boundary.py:6`: holds `is_well_formed_batch` (118-124), pure; moves to core
  (slice 1). `validation_failure_code` (110-115) maps to diagnostics enums and
  stays.
- `const.py:3` (`homeassistant.const`): holds `ERROR_*` (89-99) and
  `HELD_CALL_TTL` (100). The strings move to `core/vocabulary.py` and `const.py`
  re-exports them. The TTL value is passed in as data.
- `tracing.py:13-14` and `diagnostics.py:9-11`: `Stage` (`tracing.py:21`),
  `ErrorType` (`tracing.py:43`), `BoundaryFailureCode` and `BoundaryPhase`
  (`diagnostics.py:22`, `:32`). `Stage` and `ErrorType` move to a core module
  (slice 4). The boundary enums stay, and the adapter maps the core's phase
  and failure values onto them.
- `conversation.py:93-104` and `:294-317` (YES, NO, FILLER sets and
  `_resolve_held_option`): pure, move to core (slice 3).

Corrected cites from the review: `_async_answer` at 430 (not 431);
`_async_run_tool_calls` at 535 (not 536); correction 582-620; area drop 630;
grounding and clarify 641-654; "Done." 682-683; held replay 487-518;
`_Turn.execute` at 247 (not 248).

## Target design

### Core function

New module `custom_components/sayso/core/post_model.py`. No `homeassistant`
import. Its imports are `core.*` only, plus `asyncio` if needed (it isn't).

```python
async def run_turn(
    turn: TurnInput,
    *,
    complete: Callable[[list[dict[str, Any]]], Awaitable[ChatCompletionResult]],
    validate: Callable[[list[ToolCall]], ValidationResult],
    execute: Callable[[list[ToolCall]], Awaitable[ExecutionResult]],
    recorder: Recorder,
) -> TurnOutcome: ...
```

`complete` takes the rendered messages. The adapter closes over the tools and
the `correction` flag, so the correction call passes `correction=True` through
a second callable or a keyword: `complete(messages, *, correction: bool)`.
The adapter's `complete` owns the tools choice (`complete_schema.tools`,
which is the same object as `active_schema`, see `conversation.py:465`).

`validate` is added beyond the spec, and the reason is required. Production
validation has two layers: the pure contract check (`tool_contract.py`) and the
voluptuous check against `llm.Tool.parameters` (`boundary.py:127`). The second
layer needs HA. The core runs the contract check itself, then calls `validate`
only for calls that pass it. This matches `conversation.py:574-580`. The
eval passes a `validate` that accepts all calls, so eval gets the contract layer
and misses the voluptuous layer. That gap is documented under Risks.

`execute` takes validated calls and returns the data the core needs:

```python
@dataclass(frozen=True, slots=True)
class ExecutionResult:
    failed: bool
    ha_error: str | None
    ha_error_text: str | None
    target: str | None
    results: tuple[tuple[str, dict[str, Any]], ...]  # (call id, tool data)
```

The `results` field lets the core build the follow-up tool messages itself,
with the same shape as `chat_log_to_messages` (`transcript.py:50-56`:
`json.dumps(tool_result_data(item))`, `content: ""` for assistant tool-call
messages). Production still writes to `chat_log` inside `execute`, so HA's
history is unchanged.

Input data:

```python
@dataclass(frozen=True, slots=True)
class TurnInput:
    user_text: str
    messages: list[dict[str, Any]]           # raw history, not rendered
    render: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
    tools: tuple[dict[str, Any], ...]         # compiled function dicts
    available_names: frozenset[str]
    action_tools: frozenset[str]              # _is_action_tool equivalent
    exposed_domains: frozenset[str]
    home: GroundingHome | None                # None means not a voice turn
    held: HeldState | None
    now: float
    held_ttl: float
    max_tool_iterations: int
    confirmed: bool = False
```

`render` is a pure function. Production passes
`lambda m: apply_area_context(model_history(m), area)`. This matches
`_Turn.messages` (`conversation.py:129-134`) and is applied to every outgoing
list, including the follow-up and correction lists. Keeping `messages` raw
means the core appends to raw history, then renders. `model_history` gives the
same result on raw-plus-appended messages, because the appended messages come
after the last user message, which is the anchor (`core/history.py:20-30`).

`HeldState` is `_Held` (`conversation.py:358-363`) moved to core with the same
fields: `calls`, `index`, `options`, `expires`. `conversation.py` keeps
`_Held = HeldState` so `tests/test_grounding_flow.py:25` still imports it.

Outcome:

```python
@dataclass(frozen=True, slots=True)
class TurnOutcome:
    speech: str | None
    failure: Failure | None
    held: HeldState | None
    executed: tuple[ToolCall, ...]
    first: ChatCompletionResult | None        # first raw completion, for diagnostics
    correction_used: bool
    clarified: bool
    done_shortcut: bool
    area_dropped: bool
    held_replayed: bool

@dataclass(frozen=True, slots=True)
class Failure:
    kind: FailureKind      # INVALID_ARGUMENTS, SCHEMA_MISMATCH, UNAVAILABLE_TOOL,
                           # TOOL_EXECUTION_FAILED, ITERATION_LIMIT, EMPTY_CORRECTION,
                           # EMPTY_RESPONSE, INVALID_MODEL_OUTPUT, MODEL_FAILED, TIMEOUT
    phase: Phase           # INITIAL, CORRECTION, EXECUTION, FOLLOW_UP (core's own enum)
    validated: tuple[tuple[ToolCall, dict[str, Any]], ...] = ()
    ha_error: str | None = None
    ha_error_text: str | None = None
```

The core does not produce `execution_failure_speech`, which needs HA
registries. For `TOOL_EXECUTION_FAILED` the core returns `speech=None` and
the failure carries `validated`, `ha_error`, and `ha_error_text`, and the
adapter builds the speech. The eval uses a fixed placeholder.

`Recorder` is a protocol with `open(stage: str, **metadata) -> span`,
`close(span, *, success: bool)`, and `span.metadata: dict`. `TraceContext`
satisfies it structurally. The core opens only TOOL_PARSE spans and writes
their metadata (`area_dropped`, `clarify`, `calls`). The INFERENCE span is
opened by the adapter inside `complete`, and the HA_ACTION span is opened by
the adapter inside `execute`. Both keep their current timing, metadata, and
`success` values (`conversation.py:221-244`, `:655-662`).

### Adapter, what stays in `conversation.py`

- Context stage (434-485): LLM data, compile, domain hint, area context,
  exposed domains. These are HA reads and stay as they are.
- Building `TurnInput`: `messages = _chat_log_to_messages(chat_log.content)`
  (no `model_history`, no area context; the core renders); `available_names`
  from `build_tool_availability_names` (`schema.py:262`); `action_tools` from
  `_is_action_tool` over `tool_map` (`conversation.py:850-854`); `home` from
  `_grounding_home` when `_is_voice_turn` (`conversation.py:622-627`), else
  `None`; `held` from `self._held.pop(conversation_id)` (487); `now` from
  `time.monotonic()`; `held_ttl = HELD_CALL_TTL`.
- `complete` callable: wraps `_Turn.complete` (213-245), so the INFERENCE span
  and error mapping stay. Maps `SaySoError` to the core: the core catches it and
  returns a `Failure` with `phase`, and the adapter maps it through
  `turn.model_failure` (185-211) so speech and `record_boundary` stay the same.
- `validate` callable: `validate_arguments` (`boundary.py:127`) against
  `tool_map`, with the `dict(arguments)` behavior kept (`boundary.py:142`).
- `execute` callable: the body of `_Turn.execute` (247-287) and the HA_ACTION
  span, `apply_action_summary`, and `tool_result_failed` (`boundary.py:159`).
  It returns `ExecutionResult`.
- Outcome to response mapping: `TurnOutcome.speech` goes to `turn.text`
  (152-160), and `failure` goes to `turn.boundary_failure` (162-183) with the
  same codes, phases, and `correction_used` as today. `TOOL_EXECUTION_FAILED`
  uses `execution_failure_speech` (737-809) and the `ha_error` fields. The
  mapping table is in slice 4's verify step.
- `_async_answer` keeps the `llm_api is None` check (543-548) and the voice
  guard. Held replay moves into the core (`TurnInput.held` plus
  `now`/`held_ttl`), so `_async_answer` no longer contains the replay branch.

### Behavior the core must keep

- Repairable test (582-592): correction only when the batch has no valid calls,
  no correction yet, no executed tool, and a schema exists.
- Correction result with no tool calls: `EMPTY_CORRECTION` (613-618).
- Grounding runs only when `home is not None` and `confirmed` is false
  (560, 622). Area drop runs before grounding (628-640).
- Clarify returns `clar.speech` and stores held options only when
  `clar.options` is non-empty (646-653).
- "Done." after a single action-tool call that succeeded, without a follow-up
  model call (682-683).
- Iteration: `iteration >= max_tool_iterations` returns `ITERATION_LIMIT`
  (694-699). The counter starts at 1 (555).
- Held replay: cancel words return `ERROR_CANCELLED` (493-500); a resolved
  option replays with `confirmed=True` (502-518).

## Eval integration

`evals/runner.py` gets a sync `evaluate` that wraps the async core with
`asyncio.run` per case, or one loop for the whole list. `evaluate` is
currently called from sync code (`evals/cli.py`, `evals/tests/test_adapters.py`),
so the sync signature stays. Calling it from a running loop is not supported;
this is a test-only risk.

For each case, `render_case` gives the messages, compiled tools,
`exposed_domains`, and `area` context. The runner builds:

- `render`: `tools_before_area_messages(m, tools)` (`runner.py:328`), so eval
  keeps its prompt layout. The `sent_tools` rule (runner.py:332-334: send tools
  only when the layout did not change) is reproduced in the `complete` closure,
  which records whether `render` changed the list.
- `complete`: `adapter.complete(rendered, sent_tools)` then
  `completion.parse_completion_result(raw)` (`evals/outcomes.py:103`). Parse
  errors raise `SaySoInvalidResponseError`, so the core returns
  `Failure(INVALID_MODEL_OUTPUT)`. Transport errors (`runner.py:336-340`) become
  a `SaySoError` subclass, and the closure sets the `transport_error` flag.
- `validate`: returns every call as validated, with the arguments unchanged.
  The contract check has already run in the core.
- `execute`: a fake executor returning `ExecutionResult(failed=False, ...,
  results=((id, {"success": []}),))` for each call. The fake does not check
  entities. Eval does not test HA behavior.
- `recorder`: a null recorder.
- `action_tools`: names not in `QUERY_TOOLS` (`evals/outcomes.py:13-15`), and
  also not `todo__TodoGetItems`, which `schema._is_query_tool` treats as a query
  (`schema.py:120-131`), but `QUERY_TOOLS` does not list. Slice 5 adds a test
  that compares this set to `_is_query_tool` over HA-built tools.
- `home`: a `GroundingHome` built from the case's home dict (`load_home`,
  `cases.py:160-170`; entities, `home_areas` at `runner.py:135`, and
  `satellite_area`). Decision for the parent: every eval case is a voice turn,
  because the cases model satellite commands. Current grounding behavior is
  never exercised by eval otherwise.
- `held`: `None`. Eval cases are single-turn. The case schema has no
  follow-up utterance (`cases.py:25-39`), so held replay is not exercised by the
  eval. Held replay stays covered by `tests/test_grounding_flow.py`.

What the scorer gets:

- `Turn.text` = `TurnOutcome.speech` (final text, including grounding clarify
  speech and "Done.").
- `Turn.calls` = `TurnOutcome.executed` (calls the core would run). A
  clarified or held turn has no calls.
- `Turn.violations` = contract violations from the final validation pass only.
- `Turn.raw` = the raw output of the first completion, unchanged.
- A new `first_pass` field on `CaseResult` (or a second `Outcome` in
  `as_dict`) holds `classify(first-pass Turn)`, the current number. This is the
  before-and-after comparison. The gate uses the final outcome.
- New flags (`evals/scorer.py:384-451`): `correction_used`, `area_dropped`,
  `grounding_clarify`, `done_shortcut`. They are informational and do not
  change `passed`.

Scorer rules that change or need a decision:

- `classify` (`scorer.py:307-355`) needs no logic change. It reads
  `turn.calls`, `turn.violations`, and `turn.text`, and those now mean the final
  outcome.
- `_CLARIFICATION_CUES` (`scorer.py:18-21`) matches "did you mean" and similar
  phrases. It is used only by `is_clarification` (`scorer.py:38-40`), and that
  is used only by `is_refusal` (`scorer.py:47`). The grounding template
  `ERROR_DID_YOU_MEAN` ("Did you mean the {name}?", `core/vocabulary.py`) already
  matches the cue, so the regex can stay. For grounding clarify, the scorer
  should accept the `grounding_clarify` source flag rather than rely on the
  regex. This is a recommendation for the parent to confirm, not a required
  change. `is_refusal` behavior must stay the same for model text, because
  `evals/tests/test_scorer_ha_equivalence.py:379-384` pins it.
- `score` flags `wrong_target_area` (`scorer.py:407-413`) and
  `action_on_ambiguity` (`scorer.py:391-394`) become less frequent, because
  area-dropped and clarified calls are not executed. Those are expected score
  shifts and are reported as such.

Existing case expectations map as follows. `response_type` and `calls` keep
their meaning: `action` and `status` expect executed calls, `clarification`
expects text naming the candidates (`_names_every_candidate`,
`scorer.py:296-304`), and `refusal` expects no calls with refusal text.
`clarification.candidates` are read as before. No `cases/*.jsonl` edits.

## Slices

Each slice is one commit (not made in this plan). Each slice runs the full CI
gate in Verification, and each touches at most four files. Slice 0 is a
baseline run and writes no repo file.

0. Baseline (no repo file). Run the stub adapter over smoke and promotion on the
   current HEAD and save the summary and per-case outcomes to
   `evals/results/fix13_before_*.json` (gitignored: `.gitignore:46`). Record the
   CI pytest count (2366 passed, 1 skipped at c837b4d).
1a. Pure constants. Files: `core/vocabulary.py` (add the `ERROR_*` strings
   from `const.py:89-99`), `const.py` (re-export them). Behavior-preserving.
1b. Contract types. Files: `core/tool_contract.py` (add
   `ToolArgumentFailureCode`, `ToolArgumentValidationError`,
   `format_synthetic_validation_error`, `is_well_formed_batch`, and a
   `contract_failures` helper that mirrors `_contract_failures` at
   `conversation.py:824-847`), `schema.py` and `boundary.py` (import from
   core, same names re-exported). Behavior-preserving. Check first that no
   test imports `YES_ANSWER` (none found).
2. Correction messages. Files: `core/correction.py` (new: `_tool_call_message`,
   `build_correction_messages`), `transcript.py` (import from core; keep
   `chat_log_to_messages`), `tests/test_correction_messages.py` (new). The test
   builds the same messages through the old and new functions for a fixed
   fixture and asserts equality. Existing `tests/test_schema_identity.py:225`
   correction test must pass unchanged.
3. Held helpers. Files: `core/held.py` (new: `HeldState`, `YES_ANSWER`,
   `NO_ANSWER`, `FILLER`, `resolve_held_option`, `is_cancel`, `is_expired`
   taking `now`), `conversation.py` (import; `_Held = HeldState`; the sets and
   `_resolve_held_option` move out). `tests/test_grounding_flow.py` must pass
   unchanged, including the expired-held test that monkeypatches
   `time.monotonic` (`:279-287`), because the adapter still passes
   `time.monotonic()` at call time.
4. The loop. Files: `core/post_model.py` (new: `TurnInput`, `TurnOutcome`,
   `Failure`, `ExecutionResult`, `ValidationResult`, `Recorder`, `run_turn`),
   `core/trace_stages.py` (new: `Stage` and `ErrorType` StrEnums moved from
   `tracing.py:21` and `:43`), `tracing.py` (re-export), `conversation.py`
   (replace `_async_run_tool_calls` and the held branch in `_async_answer` with
   the adapter, using the outcome mapping). This is the riskiest slice. Its
   verify step includes the mapping table check below. No test edits.
5. Eval switch. Files: `evals/runner.py` (`evaluate` calls `run_turn` with the
   callables above), `evals/outcomes.py` (`first_pass` field on `CaseResult`,
   `Turn` unchanged in shape), `evals/scorer.py` (flags, `first_pass` outcome in
   `as_dict`), `evals/tests/test_post_model_eval.py` (new: stub-based tests for
   correction recovery, area drop, grounding clarify, "Done.", follow-up after a
   status query, transport error flag, and the `action_tools` parity check).
   `evals/tests/test_adapters.py` and `test_scorer_*` must pass unchanged.

No slice changes `ARCHITECTURE.md` (no ownership, runtime communication, or
invariant change). Revisit only if the core is later given a runtime role.

## Risks

- Trace and span parity: TOOL_PARSE and INFERENCE spans, `correction`,
  `area_dropped`, `clarify`, and HA_ACTION `target` metadata must match.
  `tests/test_tracing.py` covers spans and is the guard. The `Recorder` protocol
  keeps trace timing where it is.
- Held-call TTL and time source: production uses `time.monotonic()`
  (`conversation.py:490`, `:651`) and `HELD_CALL_TTL = 300` (`const.py:100`). The
  core takes `now` and `held_ttl` as data, and `expires` is set by the core
  as `now + held_ttl`. `tests/test_grounding_flow.py:279-287` monkeypatches
  `time.monotonic` and must still pass. Eval never sets `held`, so the TTL is not
  exercised there.
- Tool schema shapes: production validates against HA `llm.Tool` (voluptuous
  parameters, `ActionTool` classes). The core sees compiled dicts
  (`CompiledToolSchema.tools`). The contract check is already dict-based, so
  pre-execution gating matches today's eval gating. The voluptuous layer is not
  run in eval. Calls that the voluptuous layer rejects but the contract accepts
  will appear as executed in eval but not in production. Report the count of
  such calls in the eval run if it is nonzero. `action_tools` is derived by name
  in eval, so slice 5's parity test is required.
- Follow-up message parity: production rebuilds follow-up messages from
  `chat_log` (`conversation.py:688` via `turn.messages`), using the HA tool
  name (`conversation.py:262`, equal to the compiled `function.name`,
  `core/tool_schema.py:150`) and raw arguments (`boundary.py:142`, so no
  normalization). Slice 2 and slice 4 must show that core-built messages equal
  `chat_log_to_messages` output for the same turn.
- Eval score shifts: expected for correction (INVALID_ARGUMENTS to pass),
  area drop (fewer `wrong_target_area`), and grounding clarify (fewer
  `action_on_ambiguity`, possible new `MISSING_TOOL_CALL` if grounding is too
  strict). Report before and after per category, with a per-case transition
  table (before outcome to after outcome) for smoke and promotion, and keep the
  first-pass numbers next to the final numbers. Do not change a gate to hide a
  drop. Report any drop as a finding.
- Latency and call count: follow-up calls add model requests for non-action
  turns (status and query tools, and multi-call batches), so the real llama.cpp
  run takes longer. Not measured in this plan.
- Sync eval wrapper: `asyncio.run` per case inside `evaluate` fails if a caller
  already runs a loop. Current callers are sync. Keep it that way.
- Test edits: none are planned. `tests/test_grounding_flow.py:25` (`_Held`),
  `:205` (`agent._held`), and `tests/test_realistic_v3.py:24` (imports
  `is_clarification`, which is kept) depend on current names. Keep the names as
  aliases. Changing a test needs the user's approval (CLAUDE.md test rules).
- Fix 1 and Fix 2 landed in PR #208, which this branch stacks on.

## Verification

Environment for every command:
`export PATH=/Users/matt/Projects/SaySo__worktrees/ajax-codebase/.venv/bin:$PATH`

Per slice:

- `ruff check .`
- `ruff format --check .`
- `pytest tests evals/tests --ignore=tests/test_realistic_v3.py -q`
- `pytest -q satellite/sayso`
- `python -m evals.generate_realistic --check` (AGENTS.md; the generator is not
  touched, so this is a regression guard)

Slice-specific:

- Slice 2: `pytest tests/test_correction_messages.py tests/test_schema_identity.py -q`.
- Slice 3: `pytest tests/test_grounding_flow.py -q`, unchanged.
- Slice 4: `pytest tests/test_conversation.py tests/test_grounding_flow.py
  tests/test_schema_identity.py tests/test_tracing.py tests/test_area_arg_drop.py
  tests/test_tool_result_error.py tests/test_failure_speech.py
  tests/test_diagnostics.py -q`, unchanged. Plus a mapping check: each
  `Failure.kind` maps to the same `boundary_failure` code, phase, and
  `correction_used` value as the pre-slice code (table written in the slice
  notes, verified by the existing tests above).
- Slice 5, before and after on the stub adapter:
  - Baseline from slice 0, then the same command after slice 5:
    `python -c` that imports `evals.cases.select_cases`,
    `evals.adapters.in_memory.InMemoryAdapter`, and `evals.runner.evaluate` and
    runs smoke (`suite="smoke"`) and promotion (`suite="promotion"`). The stub
    returns the case's expected calls on the first call and a plain "Done." on
    any call whose last message has role `tool`. This checks the pipeline and
    not the model. Compare summaries and per-case transitions.
  - `pytest evals/tests -q` (includes `test_adapters.py` and `test_cli.py`).
  - `python -m evals.cli run --suite smoke --adapter endpoint --server ...` is
    not run here. It needs the llm VM, which is out of scope.

Real-model numbers come from the llm VM, not this plan. The stub comparison
shows that the pipeline runs and what changed in scoring. It does not show
model accuracy.

Acceptance for the whole Fix 13 batch: CI gate green; before and after report
attached to the PR body; no changes to `evals/cases/`, `evals/suites/`, the
gates, or any existing test assertion.

## Decisions (accepted by the parent, Nov 19, 2026)

1. Score the final outcome, and keep first-pass as a secondary column. The
   recommendation is to gate on final and report first-pass beside it.
2. Treat every eval case as a voice turn, with `satellite_area` from the home
   fixture.
3. Accept contract-only validation in eval (voluptuous is HA-only), and report
   the gap.
4. Keep `_CLARIFICATION_CUES` unchanged and use the `grounding_clarify` source
   flag for grounding clarify.
