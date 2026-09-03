# SaySo LFM training plan

**Status:** minimal MVP plan for supervised fine-tuning.

SaySo trains `LiquidAI/LFM2.5-230M` to select and emit Home Assistant tools in
the same OpenAI-compatible shape used by the integration. Home Assistant still
owns the entities, schemas, permissions, execution, and final voice pipeline.
llama.cpp only hosts inference.

Start with a prompt-only/base-model baseline. Train only when it improves the
held-out SaySo evaluation; a baseline that already passes the gate needs no
fine-tune.

## 1. Training contract

Use `schemas/sayso-tool-schema-v1.json` as the pinned training contract. It is a
snapshot of the Home Assistant Assist LLM API and defines the tool names,
descriptions, parameters, required fields, and constraints used to validate
examples.

Every expected tool call must pass the pinned schema before it enters a
dataset. When the Home Assistant contract changes, update the pinned artifact
and regenerate data. Do not hand-maintain a second tool catalog.

`ALLOWED_HASS_TOOLS` is a dataset-build gate for the pinned contract. It is not
the runtime capability boundary: runtime support follows the tools Home
Assistant supplies for that request.

## 2. Training examples

Store each example as structured `messages` plus OpenAI-style `tools`:

- tool entries use `{"type":"function","function":...}`;
- assistant tool calls contain the validated name and arguments;
- tool results represent the Home Assistant response before a final spoken
  answer when a follow-up is needed;
- `train_on_turn` is true only for assistant tool-call and final-response
  messages.

Labels are deterministic and schema-validated. The dataset may vary household
names, phrasing, direct actions, queries, ambiguity, no-call requests,
multi-call requests, and tool failures. Keep the expected call authoritative;
paraphrasing must never change the tool, arguments, or call/no-call decision.

Use the current locked v1 contract first. Add tools or broader data only when a
held-out evaluation identifies a concrete gap.

Do not train on:

- Home-LLM ChatML `<tool_call>` labels;
- eval case IDs or examples from `evals/cases/`;
- unsupported tools or arguments;
- model-generated labels that have not passed SaySo schema validation.

Home-LLM fixture data may provide utterance diversity, but SaySo-generated
tool-call labels and schema validation remain authoritative.

## 3. LFM format and training configuration

LFM uses its tokenizer chat template. Keep the runtime-compatible JSON-string
form for `function.arguments`; do not convert arguments to native dictionaries
or add a custom Jinja function-call template.

Use the checked-in Axolotl configurations:

| Run | Config | Purpose |
|---|---|---|
| Smoke | `training/configs/lfm25-230m-smoke.yml` | Fixture-only pipeline check |
| Production | `training/configs/lfm25-230m-prod.yml` | Train on generated data |

The production configuration currently uses:

- base model `LiquidAI/LFM2.5-230M`;
- full-parameter SFT;
- three epochs and learning rate `2e-4`;
- FP16 enabled, BF16 and Flash Attention disabled;
- assistant-only loss masking through `train_on_turn`.

The smoke configuration is a one-epoch, smaller-sequence test. Treat the
checked-in configs as the executable source for values; tune only from held-out
results, one change at a time.

## 4. Splits and evaluation

Generate deterministic train, validation, and test splits. Keep related
template, phrasing, and seed families in one split so paraphrase variants do
not leak across evaluation boundaries.

Use validation for checkpoint selection and tuning. Keep the test set and
`training/evals/adversarial.jsonl` untouched until the final comparison.

Evaluate the base model and candidate checkpoints on the same cases. Record:

- protocol-valid tool calls;
- correct tool names and arguments;
- schema-valid arguments;
- correct no-call and unsupported-request behavior;
- multi-call behavior;
- final spoken response presence;
- latency and inference failures where available.

Promote a checkpoint only when it improves the target SaySo behavior without
regressing safety, schema validity, or the basic offline eval. Then export GGUF
and verify structured tool calls with llama.cpp before using it in Home
Assistant.

## 5. Runtime safety boundary

Training does not replace runtime controls. The SaySo integration must continue
to:

1. compile the tools supplied by Home Assistant;
2. treat model output as untrusted;
3. validate every call and every argument before execution;
4. validate every call in a batch before executing any call;
5. fail closed on ambiguity, unsupported tools, malformed output, and schema
   mismatch;
6. allow only the existing bounded correction path before execution;
7. execute through Home Assistant and use its result for the spoken response;
8. never retry an already executed action because of a later invalid response.

## 6. Non-goals for the MVP

- model bake-offs;
- a multi-stage curriculum;
- fixed dataset-percentage or ASR-corruption quotas;
- fine-tuning on Home-LLM tool-call labels;
- long autonomous chains;
- replacing Home Assistant validation with model trust;
- a SaySo server, broker, custom action protocol, or direct satellite-to-model
  connection.

## Implementation map

| Concern | Location |
|---|---|
| Dataset generation | `training/scripts/generate_dataset.py`, `training/generators/` |
| LFM adapter | `training/adapters/lfm.py` |
| Schema validation | `training/adapters/schema.py` |
| LFM configs | `training/configs/lfm25-230m*.yml` |
| Evaluation | `evals/`, `training/evals/` |
| Pinned contract | `schemas/sayso-tool-schema-v1.json` |

Operational commands belong in `training/README.md`; update this document only
when the training design or its safety boundary changes.
