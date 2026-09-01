# SaySo architecture

Status: target architecture

SaySo is a Home Assistant conversation agent for a local voice assistant. Home
Assistant owns the voice session, device state, permissions, tool execution,
and response delivery. SaySo supplies the conversation behavior and the tuned
model that turns natural language into a typed, safe control plan.

The complete path is:

```text
Home Assistant voice satellite
        ↓
Replaceable wake-word engine
        ↓
Home Assistant Assist pipeline
        ↓
SaySo ConversationEntity
        ↓
Tuned SaySo model
        ↓
Validated HA tool calls
        ↓
Home Assistant state and response
```

## Design goals

- Make a Mac or other supported device behave like a temporary Home Assistant
  voice satellite.
- Keep wake-word detection replaceable without changing conversation logic.
- Use the native Assist pipeline for audio capture, turn handling, speech
  recognition, conversation context, and response playback.
- Tune SaySo for the small, typed smart-home control vocabulary that it must
  handle well.
- Prevent model output from becoming an arbitrary Home Assistant service call.
- Make every mutating action explainable, permissioned, and state-verified.
- Keep the first physical-device demo local and small: one home, one pipeline,
  and one dependable control path.

## Architectural invariants

1. Home Assistant is the source of truth for entities, areas, capabilities,
   permissions, and live state.
2. The satellite and wake-word engine never choose or execute Home Assistant
   services.
3. The Assist pipeline owns the voice turn. SaySo receives a conversation
   input, not an independent audio protocol.
4. The model emits a strict semantic `ControlPlan`. It does not emit raw
   Home Assistant service calls or use entity IDs as semantic targets.
5. Deterministic code parses, resolves, validates, and authorizes a plan before
   creating any HA tool call.
6. Ambiguous, unsupported, unresolved, hidden, incapable, malformed, or unsafe
   requests produce clarification or no-action and cannot mutate state.
7. Home Assistant remains the final execution boundary. It checks the tool
   call again, performs the service call, and verifies the resulting state.
8. Queries and clarifications do not invoke mutating tools.
9. The model is replaceable behind one narrow adapter; changing the model does
   not change safety or execution rules.

## Component boundaries

| Component | Owns | Must not own |
|---|---|---|
| Voice satellite | Microphone/speaker transport and Assist-compatible audio | Intent parsing, entity selection, HA service calls |
| Wake-word engine | Detecting the configured wake phrase and emitting a wake event | Transcription, conversation decisions, device control |
| Assist pipeline | Audio turn lifecycle, VAD/endpointing, STT, conversation ID, context, and response playback | SaySo-specific planning or direct model tool execution |
| SaySo `ConversationEntity` | Conversation handling, candidate context, model invocation, plan validation, and response policy | Owning authoritative HA state or bypassing HA permissions |
| Tuned SaySo model | Mapping natural language and supplied context to a typed plan | Inventing tools, choosing arbitrary entity IDs, or executing actions |
| Validation/tool adapter | Plan parsing, target resolution, safety/capability checks, and semantic-to-HA tool mapping | Speech recognition, free-form model repair, or hidden side effects |
| Home Assistant | Registries, exposure, permissions, tools/services, state changes, and verification | Depending on model output being safe |
| Evaluation and telemetry | Stage timings, outcomes, failure attribution, and regression checks | Live actuation by default |

The ConversationEntity is the only SaySo integration boundary exposed to Assist.
The model and validator may be in-process or reached through a local adapter,
but neither receives authority to call Home Assistant directly.

## Runtime sequence

### 1. Wake and capture

The satellite listens through a replaceable wake-word engine. A successful
detection starts one Assist voice turn. The engine reports only a wake event;
it does not decide whether the spoken request is safe or supported.

The satellite sends the turn to Home Assistant using the selected Assist
pipeline. Assist handles endpointing and audio transport according to the
pipeline configuration.

### 2. Assist processing

The Assist pipeline transcribes the turn, attaches the conversation and
satellite context, and routes the text to SaySo's `ConversationEntity`. The
entity receives the transcript, conversation ID, originating satellite/area,
and the Home Assistant context needed for the turn.

STT errors, empty turns, timeouts, and pipeline transport failures end before
model execution and become a normal Assist error response.

### 3. SaySo planning

The ConversationEntity builds a bounded model input from:

- the current transcript;
- the Assist conversation context and recent referents;
- exposed Home Assistant entities and their semantic names;
- relevant areas, floors, aliases, capabilities, and current state; and
- the supported action vocabulary and ControlPlan schema.

The tuned model returns one typed `ControlPlan`:

```json
{
  "outcome": "action",
  "intent": "turn_on",
  "targets": ["corner lamp"],
  "scope": {"area": "living room"}
}
```

The exact schema is defined in code and is versioned with the model adapter.
The model may request an action, query, clarification, unsupported result, or
no-action. It may use semantic names and aliases, but it may not invent a
Home Assistant entity ID, domain, service, or arbitrary tool name.

Invalid or extra model output is rejected. There is no generic agent loop and
no speculative JSON repair.

### 4. Deterministic validation

The validator is the trust boundary between model output and Home Assistant.
For an action plan it:

1. validates the schema and supported intent;
2. resolves names, aliases, pronouns, areas, floors, inclusions, and
   exclusions against Home Assistant's exposed graph;
3. applies the ambiguity rule and asks for clarification when plausible
   targets cannot be distinguished;
4. checks domain, capability, value/range, and requested-state compatibility;
5. rejects hidden, disallowed, unknown, empty, or mixed-invalid target sets;
6. maps the approved semantic operation to a bounded HA tool call; and
7. records the plan, selected targets, validation result, and reason.

Validation is atomic for a single plan: a request is not partially executed
because one requested target is invalid. Explicit multi-target operations are
allowed only when every target passes the same checks and the operation's
fan-out limit.

Queries follow a read-only path and return current Home Assistant state. A
clarification stores only the minimum conversation reference needed for the
next turn and expires it rather than guessing after context is stale.

### 5. HA tool execution and verification

Only validated tool calls cross into Home Assistant:

```text
validated tool call
  → HA exposure and permission check
  → domain/capability/entity check
  → Home Assistant service/tool execution
  → state_changed observation or bounded timeout
  → completed, unchanged, or failed result
  → Assist response
```

Home Assistant repeats the boundary checks because the graph or permissions
may have changed since planning. A service exception is a failed action. A
missing expected state change is not reported as success. The ConversationEntity
turns the result into a concise spoken response, earcon, clarification, or
error through Assist.

## State and authority

| State | Authority | SaySo access |
|---|---|---|
| Entity and device registry | Home Assistant | Read-only context for planning |
| Exposure and permissions | Home Assistant configuration | Read-only during planning; enforced again at execution |
| Current device state | Home Assistant state machine | Read-only query and verification input |
| Conversation session | Assist conversation context | Bounded referents keyed by conversation ID if needed |
| ControlPlan | SaySo model output after strict parsing | Never treated as authorization |
| Tool call | SaySo validator output, rechecked by HA | No direct service authority |
| Physical state change | Home Assistant and the device | Observed by verification |

SaySo does not maintain an authoritative shadow graph. If cached planning
context is stale or unavailable, it fails closed or asks for clarification.

## Replaceable interfaces

The following interfaces are intentionally narrow:

- `WakeWordEngine`: emits wake events and exposes lifecycle/configuration.
- `AssistPipeline`: accepts a voice turn and routes it to a conversation agent.
- `ConversationEntity`: accepts Assist conversation input and returns an Assist
  conversation result.
- `SaySoModel`: accepts normalized transcript/context and returns typed model
  output.
- `HaToolAdapter`: exposes only the validated semantic tools available to
  SaySo and returns execution/verification results.

Wake-word engines and model runtimes can be swapped independently. The
ControlPlan, validator, HA tool contract, and evaluation cases remain stable
across those substitutions.

## Security and failure behavior

All external input is validated at its boundary. Model output is untrusted
input. Credentials and local transport security follow Home Assistant's
configuration; a public-network deployment requires stronger identity and
transport controls before exposure.

| Failure | Behavior |
|---|---|
| Wake engine unavailable | Assist remains usable through another configured trigger or reports unavailable |
| STT/Assist failure | No model call; concise error response |
| Invalid model output | No-action; record a parse/schema failure |
| Ambiguous target | Clarification; no tool call |
| Unsupported or incapable request | No-action or clarification; no tool call |
| Hidden/disallowed entity | Rejection at validation or HA execution; no mutation |
| HA service failure | Failed response; never claim success |
| Verification timeout/unchanged state | Failed or unconfirmed response |
| Restart or lost connection | Rebuild context from HA; fail closed until ready |

## Observability and evaluation

Telemetry must identify a conversation turn without storing raw audio by
default. At minimum record correlation ID, pipeline/model revisions, outcome,
selected target IDs, validation reason, execution result, and stage timings.

Measure the path by stage so aggregate success cannot hide a dangerous action:

```text
wake → Assist/VAD → STT → retrieve → plan → parse/validate
     → resolve → authorize → HA tool → verify → response
```

The evaluation harness uses authored cases and dry-run execution by default.
It must separately score ControlPlan accuracy, candidate recall, exact target
resolution, clarification behavior, false execution, wrong-device execution,
query/follow-up accuracy, and cold/warm latency. Live execution requires an
explicit mode and entity allowlist.

The first physical-device gate is a complete successful turn:

```text
wake → capture → Assist/STT → SaySo ConversationEntity → tuned model
→ validated HA tool call → state verification → spoken response
```

Model bake-offs, larger corpora, generalized multi-satellite support,
fine-tuning hooks, streaming optimizations, and polished diagnostics follow
only after this path is reliable.

## Implementation map

The codebase should converge on these ownership boundaries:

| Concern | Home Assistant boundary | SaySo boundary |
|---|---|---|
| Voice satellite and wake word | Satellite/Assist configuration | None beyond conversation input |
| Conversation agent | ConversationEntity registration and response type | Conversation orchestration |
| Model | Local runtime adapter | Tuned model and prompt/schema |
| Resolution and safety | HA exposure/capability checks | Semantic resolution and plan validation |
| Execution | HA tools/services and state verification | Validated tool-call request/result handling |
| Evaluation | Optional live allowlisted executor | Dry-run controller, metrics, and ledger |

The primary production path is the Assist-to-ConversationEntity path above.
Direct text or audio HTTP entry points, independent satellite protocols, and
server-side HA service clients are compatibility or evaluation surfaces, not
alternate authorities.

See [EVALUATION_PLAN.md](EVALUATION_PLAN.md) for the evaluation contract and
measurement gates.
