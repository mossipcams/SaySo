# SaySo Evaluation and Benchmark Plan

Status: Approved planning baseline  
Companion: [Architecture](ARCHITECTURE.md)

## Objective

Determine whether SaySo is more accurate and faster than Home-LLM for routine
smart-home control and whether measured end-to-end behavior can outperform
Alexa+ on the same household workload.

The harness must separate failures in:

1. Speech recognition
2. Entity retrieval
3. ControlPlan generation
4. Deterministic resolution
5. Safety validation
6. Home Assistant execution
7. Physical state verification

This prevents an aggregate score from hiding a dangerous wrong-device path.

## Fixed test environment

- A versioned synthetic Home Graph used by every model.
- An optional redacted snapshot of the real evaluation home.
- Fixed origin satellite and area per case.
- Identical candidate entities and ControlPlan schema where technically possible.
- Deterministic decoding with recorded seed and generation settings.
- The same Mac, power mode, model residency policy, and warmup procedure.
- Pinned model IDs, revisions, quantization, runtime, dependency versions, and
  Home-LLM revision in every report.
- Dry-run execution by default; live execution requires both `--execute` and an
  explicit entity allowlist.

## Dataset design

Initial target: 320–420 reviewed cases, scaling past 1,000 after the harness is
stable.

| Category | Initial count | Required examples |
|---|---:|---|
| Simple control | 40 | on, off, toggle, brightness, thermostat |
| Room-relative control | 35 | current room, named room, implicit room |
| Multiple devices | 30 | groups, whole room, mixed current states |
| Exceptions and negation | 35 | all except, do not touch, leave on |
| Pronouns and follow-ups | 60–80 | it, them, back on, expired reference |
| State queries | 35 | single state, any/all group query |
| Ambiguity | 35 | equally plausible names and aliases |
| Unsupported/no action | 35 | unsupported intent, unsafe approximation |
| Casual speech | 25 | fillers, corrections, household phrasing |
| ASR-corrupted speech | 25 | plausible substitutions and dropped words |
| Aliases | 20 | entity, room, device, script, and scene aliases |
| Floor-relative control | 20 | upstairs/downstairs with exclusions |

Variants are deterministic and tied to an authored expected result. Runtime
model-generated test data is not counted as reviewed coverage.

Train data is a separate pipeline ([TUNING_PLAN.md](TUNING_PLAN.md)). Eval
`case_id`s never enter SFT. Home-LLM's generator is allowed as a synthetic
source for that pipeline; Home-LLM tool-call labels are not.

## Expected result per case

```json
{
  "case_id": "exceptions-001",
  "category": "exceptions",
  "origin_area": "living_room",
  "turns": ["Turn all the lights off in here except the lamp"],
  "expected_control_plan": {},
  "expected_candidate_entities": [],
  "expected_resolved_entities": [],
  "expected_outcome": "valid_action",
  "execution_allowed": true
}
```

Comparisons canonicalize optional defaults and unordered entity sets. They do
not weaken semantic requirements or treat a merely executable action as
correct.

## Primary metrics

### Exact ControlPlan accuracy

Percentage of cases whose canonical ControlPlan exactly matches the expected
intent, domain, scope, state/value/mode, include/exclude semantics, and outcome.

Report parser/schema failures separately from semantically wrong plans.

### Candidate retrieval recall

Percentage of actionable cases where every required target is present in the
candidate set before model inference. Also report candidate-set size and top-k
recall.

### Exact target resolution

Percentage of actionable cases where the resolved entity-ID set exactly equals
the expected set.

### Wrong-device rate

```text
action cases where at least one unintended entity was targeted
---------------------------------------------------------------
                  all executed action cases
```

Also report unintended entity count. This is a severe failure and cannot be
hidden by partial target correctness.

### False-execution rate

Percentage of ambiguity, unsupported, invalid-plan, unresolved-reference, or
permission-denied cases that produce any Home Assistant action.

### Clarification precision and recall

- Precision: requested clarifications that were actually required.
- Recall: ambiguous cases that correctly requested clarification.

### Query and follow-up accuracy

Exact normalized answer accuracy for state queries and exact plan/target
accuracy for follow-up turns.

### Latency

Record at minimum:

- wake latency
- VAD/endpoint latency
- STT latency
- entity retrieval latency
- prompt construction latency
- model prefill latency when available
- model decode latency
- prompt and completion token counts
- ControlPlan parse/validation latency
- resolution latency
- HA request/accept latency
- HA state-verification latency
- total EOS-to-HA-action latency

Report cold start separately. Primary latency statistics use a resident, warm
model and include median, p95, sample count, failures, and hardware metadata.

## Implementation tasks

Each task is a 5–15 minute TDD unit and follows the workflow in the parent plan.

| Task | Failing test first | Minimal implementation | Verification |
|---|---|---|---|
| 39. Evaluation schema | Incomplete expected outcomes load | JSONL case schema with home, origin, turns, expected plan/targets/outcome | Invalid rows fail fast |
| 40. Core corpus | Category/count check fails | 120 authored simple, room, multi-device, scene, script, climate, and query cases | Expected outputs and counts pass |
| 41. Safety corpus | Ambiguity/unsupported coverage fails | 100 ambiguity, pronoun, negation, exclusion, unsupported, and no-action cases | Every safety category has positive and negative cases |
| 42. Language-noise corpus | Casual/alias/ASR coverage fails | 100–200 deterministic variants tied to authored outcomes | Total reaches 320–420 reviewed cases |
| 43. Follow-up corpus | Multi-turn coverage fails | 60–80 paired turns with expiry and referent changes | Follow-up assertions pass |
| 44. Metric scorer | Deliberately wrong records score incorrectly | Canonical plan, retrieval, target, safety, clarification, query, and follow-up metrics | Hand-calculated fixture matches |
| 45. Benchmark runner | Model error aborts the run | Seeded, resumable JSONL runner with warmup and timings | Interrupted run resumes without duplicates |
| 46. LFM benchmark | Report lacks quantization/token data | LFM run configuration and warm repeated measurements | Reproducible report is generated |
| 47. Home-FunctionGemma benchmark | Second model requires controller changes | Model-specific prompt/parser adapter with common candidates and ControlPlan | Both models use the same scorer |
| 48. Live safety boundary | Evaluation can actuate accidentally | Dry-run default plus execution flag and allowlist | No action without both safeguards |
| 49. Home-LLM baseline import | Service-call output cannot compare | Optional JSONL adapter normalizing actions and targets | Same corpus yields comparable metrics |
| 50. Alexa+ trial import | Manual observations cannot enter reports | Randomized command sheet and CSV observation importer | Sample observations generate metric tables |
| 51. Statistical report | Report hides sample sizes or cold starts | Median/p95, distributions, category failures, and environment metadata | Golden report assertion passes |

## LFM2.5 versus Home-FunctionGemma

Run two views when technically possible:

1. Same-runtime comparison: identical runtime, quantization class, schema,
   candidate set, prompt budget, hardware, and generation settings.
2. Best-local-runtime comparison: each model uses its best supported local Mac
   runtime, with runtime differences displayed rather than hidden.

Both models receive the same semantic information. Model-specific chat or
tool-call syntax may differ, but both adapters must return the same typed
ControlPlan contract.

Required outputs:

- Accuracy and safety metrics by category
- Parser/schema failure rate
- Candidate retrieval held constant
- Prompt/completion tokens
- Cold and warm latency
- Peak resident memory when available
- Full failure ledger keyed by case ID

## Home-LLM comparison

Home-LLM remains an evaluation-only dependency and never enters SaySo's
production control path.

For the fairest end-to-end comparison:

- Pin the Home-LLM integration and model revisions.
- Expose the same Home Assistant entities and aliases.
- Run the same text and audio cases.
- Normalize its service-call outputs to action, targets, and values.
- Score observed target state, not wording or response style.
- Include its Home Assistant/Assist routing time in end-to-end latency.
- Report unsupported or malformed tool calls as failures, not exclusions.

## Alexa+ comparison

Alexa+ is a black box, so SaySo internal telemetry cannot establish that it is
faster. Use a paired household trial:

1. Randomize the same routine command set across SaySo and Alexa+.
2. Replay or speak from a fixed position and level.
3. Start timing at observed end of speech.
4. Stop timing at Home Assistant/device state change or an external observable
   physical-state signal.
5. Record wrong device, no action, clarification, timeout, and response text.
6. Repeat enough trials to distinguish stable behavior from network variance.
7. Keep cloud/network conditions and device routes in the report.

Report SaySo as “capable of outperforming Alexa+” only until paired trials show
an actual accuracy and latency advantage.

## Statistical limits

The initial 300–500 cases are enough to compare models and discover common
failures, but they cannot confidently prove a wrong-device rate below 0.5%.

With zero observed wrong-device failures, roughly 600 independent action trials
are required merely to place the one-sided 95% upper bound near 0.5%. Prefer at
least 1,000 varied action trials before making a strong safety-rate claim.

Always publish:

- numerator and denominator
- confidence interval
- excluded and failed cases
- number of repeated trials
- dataset revision
- model/runtime revisions
- warm/cold status
- hardware and power configuration

## Decision gates

### Gate A — Retrieval viability

- Required-target top-k recall is high enough that model comparison is useful.
- Candidate sets remain small enough for low prompt and prefill latency.

### Gate B — Model viability

- LFM2.5 and Home-FunctionGemma complete the full corpus.
- Invalid outputs always become no-action.
- No model is advanced solely on generic benchmark performance.

### Gate C — Deterministic safety

- Resolver target accuracy is measured independently from model accuracy.
- Ambiguous and unsupported cases have zero execution in dry-run validation.
- Live execution is allowlisted and state-verified.

### Gate D — Voice viability

- STT-corrupted and recorded-audio cases preserve acceptable action accuracy.
- Median and p95 EOS-to-action are measured with a resident model.

### Gate E — Product claim

- SaySo beats the selected Home-LLM baseline on wrong-device and false-execution
  rates without unacceptable latency regression.
- Alexa+ claims are based on paired observed trials, not architectural inference.

## Performance targets

- Simple command accuracy: greater than 99%
- Wrong-device execution: less than 0.5%
- Median EOS-to-HA command: less than 1 second
- p95 EOS-to-HA command: less than 2 seconds

These remain optimization targets until the sample size and observed results
support them.
