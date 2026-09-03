# SaySo training plan

**Status:** authoritative design for SaySo supervised fine-tuning (SFT).

This document is the **single source of truth** for how SaySo trains a Home
Assistant tool-calling model. Operational scripts under `training/` implement
this plan incrementally; when code and this document disagree, update the code
to match this plan.

Related but separate: [SCHEMA_LOCK_PLAN.md](SCHEMA_LOCK_PLAN.md) covers the
**runtime** llama.cpp tool-schema compiler and reference artifact lock. It does
not define training data design.

## Design principle: schema-conditioned function calling

SaySo must **not** train by memorizing a flat list of Home Assistant tool names.
At runtime, Home Assistant supplies the exact tool menu for each request; the
model must:

1. Read the provided tool schemas (name, description, parameters, required
   fields, enums, ranges, result shape).
2. Select the correct tool, produce valid arguments, issue independent parallel
   calls when appropriate, or make **no call** when the request is ambiguous,
   unsupported, or already satisfied.

This matches SaySo architecture (Home Assistant is authoritative for tools and
validation) and FunctionGemma strengths (reliable single and parallel tool
calls; not long multi-step chains).

Training teaches **conditional selection over changing menus**, not rote recall
of a fixed catalog.

## 1. Training contract from Home Assistant

Generate the training contract from a **pinned real Home Assistant 2026.8.3
Assist API** snapshot. For each tool exposed by that API, capture:

| Field | Source |
|-------|--------|
| Tool name | HA `Tool` name |
| Description | HA tool description |
| Parameters | JSON Schema from `voluptuous_openapi.convert()` |
| Required fields | Schema `required` array |
| Enums and ranges | Schema constraints (enums, min/max, patterns) |
| Result shape | Documented or observed tool-result structure |

**Pin** the snapshot as a versioned artifact (aligned with
`schemas/sayso-tool-schema-v1.json` once the runtime schema lock completes).
**Diff** the contract when Home Assistant or the Assist API changes; regenerate
training data against the new pin rather than editing tool definitions by hand.

**Do not** manually maintain Home-LLM tool definitions or copy Assist dumps
from third-party projects. Home Assistant installations vary available tools per
request, assistant, and exposure; training must reflect schema-conditioned
behavior, not a static Home-LLM subset.

## 2. Changing tool menus per example

Each training example includes:

- The **correct** tool (and arguments).
- **2–6 plausible distractor tools** drawn from the same contract (same domain,
  similar names, or commonly confused alternatives).
- Realistic **household context**: entities, areas, floors, aliases, satellite
  areas, and exposure-relevant names.
- The **expected tool call** (name + arguments validated against the real HA
  schema).
- A **realistic HA tool result** (success, partial, or failure as appropriate).
- A brief **confirmation or failure** assistant turn (TTS-style).

Use the **full tool catalog in only ~15–20%** of examples (uncertain-routing /
fallback scenarios). Do **not** attach every tool to every example; the model
must learn to route from the menu it is given.

## 3. FunctionGemma native format

Store examples as **structured messages and tools**, then render through the
**exact FunctionGemma chat template** used by llama.cpp at inference:

- `<start_function_declaration>` … tool schemas …
- `<start_function_call>` … expected call …
- `<start_function_response>` … HA result …

OpenAI-compatible `/v1/chat/completions` JSON is **transport only** at runtime.
**Do not** train on literal chat-completions response JSON blobs.

The Axolotl conversion path that parses `function.arguments` strings into
argument **dictionaries** for the FunctionGemma Jinja template is the correct
direction. Each message carries `train_on_turn`: train only on assistant tool-call
and final spoken-response turns.

## 4. Label-first example generation

Generation is **label-first**: deterministic code owns the label. Do not ask a larger
model to decide the correct tool or arguments.

Generation order:

1. **Choose** a valid HA tool and arguments from the pinned contract (respecting
   enums, required fields, and ranges).
2. **Create** a synthetic household and context (entities, areas, state).
3. **Generate** several natural user utterances for that label.
4. **Add** ASR-corrupted variants (~20–25% of examples across categories).
5. **Validate** the expected call against the real HA schema before emitting the
   row.

A larger model may **paraphrase** utterances or confirmations only; it must not
change tool name, arguments, or whether a call is required.

## 5. Dataset mixture

Target distribution across the full dataset:

| Category | Share | Notes |
|----------|-------|-------|
| Single tool calls | 50% | Core on/off, brightness, fan speed, etc. |
| Parameterized controls and confusing alternatives | 15% | Same intent, different entity/area/parameter |
| Independent parallel calls | 10% | Two+ tools with no ordering dependency |
| State and context queries | 10% | `GetLiveContext`, status questions |
| Ambiguity, unsupported, no-call | 10% | Refusals, already-in-state, out-of-scope |
| Tool failure and malformed-result recovery | 5% | Retry, clarify, or safe failure speech |

**ASR corruption:** apply to ~20–25% of examples, spread across categories (not
only single-call).

**Tool frequency:** ~70% drawn from realistic household frequency (lights,
switches, common rooms dominate); ~30% balanced sampling so rare timer, vacuum,
media, climate, list, and query tools still appear often enough to learn.

**Do not** fine-tune on Home-LLM ChatML `<tool_call>{"name","arguments"}</tool_call>`
labels or on eval case IDs from `evals/cases/`.

## 6. Curriculum and tool scope

Train in stages; reach convergence on each stage before expanding:

1. **Core tools** — turn on/off, lights, fans, locks, covers basics, live context.
2. **Media controls** — play, pause, volume, source selection.
3. **Full timer lifecycle** — start, pause, cancel, query.
4. **Lists and queries** — shopping lists, todo, calendar-style queries exposed
   by Assist.
5. **Vacuums and covers** — start, dock, open/close, position.
6. **Broadcast and date/time** — announce, reminders, time queries.
7. **Dynamic scripts and contributed tools** — only schemas HA Assist actually
   exposes for the pinned API.

**Do not** add Home-LLM-only tools (e.g. humidifier-specific helpers) unless the
targeted HA Assist API snapshot includes those schemas. If HA does not expose a
tool, SaySo does not train it.

## 7. Model and hyperparameters

**Base model:** `google/functiongemma-270m` (FunctionGemma 270M).

**Method:** full-parameter SFT. LoRA is optional only when experiment turnaround
matters more than final accuracy.

| Setting | Value | Rationale |
|---------|-------|-----------|
| Learning rate | ~**5e-5** | Google-documented starting point; 2e-4 is too aggressive for full FT |
| Epochs | 2–3 | Avoid overfitting small menus |
| Checkpoint selection | Held-out **tool accuracy** | Not training loss |
| Precision | FP16 | GTX 1070 / Pascal |
| BF16 | Off | Unsupported on Pascal |
| Flash Attention | Off | Unsupported on GTX 1070 |

Export to GGUF and verify tool calling with llama.cpp `--jinja` before
promoting a checkpoint.

## 8. `ALLOWED_HASS_TOOLS` and the pinned contract

`ALLOWED_HASS_TOOLS` in `training/adapters/schema.py` validates examples against
the **pinned training contract** during dataset build and CI. It is a **gate**,
not the permanent definition of what SaySo supports at runtime.

At inference, the model learns from **whatever schemas Home Assistant provides**
for that request. Runtime support follows HA exposure and Assist API tools, not
the training allowlist alone. When the HA contract changes, update the pin and
regenerate data; do not treat the allowlist as product capability documentation.

## 9. Evaluation and quality gates

- Hold out adversarial and validation sets from training and hyperparameter
  selection (`training/evals/adversarial.jsonl`, split val).
- Measure protocol-valid tool calls, schema-valid arguments, correct tool name
  on single-action turns, no-tool decisions, and parallel-call exact match.
- Basic behavioral eval remains under `evals/`; do not train on those case IDs.

Initial targets (adjust as the pipeline matures):

| Metric | Target |
|--------|--------|
| Protocol-valid tool calls | ≥99% |
| Single-action tool name | ≥95% |
| Schema-valid arguments | ≥98% |
| No-tool decision | ≥95% |
| Multi-action exact match | ≥90% |
| Unsupported-tool hallucination | <1% |

## 10. Explicit non-goals

- Memorizing a flat HA tool list without per-request schemas.
- Training on Home-LLM tool-call label format or vendored pile labels as ground
  truth.
- Hand-maintaining tool definitions parallel to Home Assistant.
- Complex multi-step chains beyond what FunctionGemma handles reliably.
- Replacing runtime Voluptuous validation or SaySo fail-closed barriers with
  model trust.

## 11. Implementation map (reference)

| Concern | Location |
|---------|----------|
| Dataset generation | `training/scripts/generate_dataset.py`, `training/generators/` |
| Schema validation | `training/adapters/schema.py` |
| Axolotl / FunctionGemma views | `training/adapters/`, `training/scripts/adapt_dataset.py` |
| Training configs | `training/configs/functiongemma-270m*.yml` |
| Pinned contract artifact | `schemas/sayso-tool-schema-v1.json` (runtime lock; training pin) |
| Runtime schema compiler | `custom_components/sayso/` (see SCHEMA_LOCK_PLAN.md) |

When adding or changing training code, update this document only if the **design**
changes; operational tweaks belong in `training/README.md` command tables.
