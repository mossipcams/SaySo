# SaySo training plan

**Status:** current contract for `LFM2.5-230M-Base` supervised fine-tuning.

SaySo trains `LiquidAI/LFM2.5-230M-Base` to select and emit Home Assistant tools in
the same OpenAI-compatible shape used by the integration. Home Assistant still
owns the entities, schemas, permissions, execution, and final voice pipeline.
llama.cpp only hosts inference.

Do not revive Instruct-checkpoint, FunctionGemma, or Axolotl full-SFT
plans. Checked-in `training/configs/lfm25-230m*.yml` and `functiongemma-270m*.yml`
Axolotl files are leftovers, not the training path.

## 1. Training contract

Use `schemas/sayso-tool-schema-v1.json` as the pinned training contract. It is a
snapshot of the Home Assistant Assist LLM API and defines the tool names,
descriptions, parameters, required fields, and constraints used to validate
examples. `schemas/sayso-tool-schema-v2.json` mirrors the same tools grouped by
device-type tier (`query`, `generic`, `light`, `fan`, `climate`, `media_player`,
`vacuum`, `timer`) for catalog validation; the flat `tools` list remains the
training source of truth.

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

Labels are deterministic and schema-validated. Keep the expected call
authoritative; paraphrasing must never change the tool, arguments, or
call/no-call decision.

Do not train on:

- ChatML `<tool_call>` labels;
- eval case IDs or examples from `evals/cases/`;
- variants of the 38 recipe-lock golden utterances;
- unsupported tools or arguments;
- model-generated labels that have not passed SaySo schema validation.

Synthetic generation in `training/scripts/build_synthetic_dataset.py` owns
utterance diversity; schema validation remains authoritative for every label.

## 3. Format and trainer

Keep two representations separate:

- The canonical SaySo dataset keeps OpenAI-compatible `function.arguments` as
  validated JSON strings.
- The TRL view may parse those strings into native objects only while applying
  the LFM2.5-Base `chat_template.jinja`. Do not put ChatML in the dataset or in
  the Home Assistant integration.

```text
canonical JSONL (OpenAI messages, string arguments)
        ↓ tokenizer.apply_chat_template + tools= (dict arguments)
ChatML + <|tool_call_start|>[HassTurnOn(name='…')]
        ↓ rsLoRA SFT on Base (TRL)
merged FP16 adapter
        ↓ llama.cpp convert + Q8_0
GGUF served with --jinja
        ↓ SaySo HTTP still sends OpenAI messages, not ChatML
```

Train from Base with TRL rsLoRA, not by continuing a previous merged checkpoint:

- rank 32, alpha 32, `use_rslora: true`, `all-linear`, FP16, no BF16, no Flash Attention
- microbatch 1, gradient accumulation 16
- learning rate `2e-4`, cosine, assistant-only loss
- `save_strategy: epoch`, keep every epoch checkpoint needed for comparison

Smoke one update step on the target host before a full run. Pascal GTX 1070
(8 GiB) may log an allocator warning and still complete; treat a finished step
plus a reloadable adapter as the gate.

## 4. Data and eval

Locked gold eval is the 38 recipe-lock cases in `training/evals/recipe_lock.py`
(recipes 1–8, thermostat omitted). Generic nouns resolve in the SaySo entity
area. Do not train on those utterances.

The current mix is:

- 10k deterministic train from `training/scripts/build_synthetic_dataset.py`
- plus the existing 10k-plus supplement
- plus a small corrective set (500–800 rows) from
  `training/scripts/generate_training_supplement.py`

Corrective rows must use fresh homes, entities, and wording. Weight the four
remaining failure classes (light brightness vs fan speed, lock vs unlock,
apostrophe names, multi-action retention). Do not generate another 10k set and
do not start a third epoch on a corrective retrain.

Shadow eval is 100–150 cases covering the same concepts as the 38 gold rows,
with different entities and phrasing (`sayso_shadow_eval.jsonl`). Promote only
when golden and shadow both move the right way. If only golden improves, the
run is overfitting the benchmark.

Score generations with the apostrophe-safe parser in
`training/evals/lfm_python_parse.py` (raw `/completion` text). llama.cpp
structured `tool_calls` still truncates names such as `O'Malley's` and `Kids'`;
that is a serving bug, not a training label. Do not retrain to paper over it.

`training/scripts/generate_balanced_test_data.py` builds the 2,500-example
held-out set. Do not train on those prompts. Use it when asked; the 38 gold
cases plus shadow are the working quality gate.

Promote a checkpoint only when it improves target behavior without regressing
STT, status, no-call, multi-action, light/fan, or lock polarity. Then export
GGUF and verify with llama.cpp `--jinja`. Freeze a promoted champion and do
not overwrite its GGUF, merged weights, or epoch checkpoint.

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

## 6. Non-goals

- model bake-offs (Instruct LFM, FunctionGemma, Alexa+)
- Axolotl full-parameter SFT
- another 10k generation or Epoch 3 on the corrective mix
- fine-tuning on ChatML `<tool_call>` labels
- long autonomous chains
- replacing Home Assistant validation with model trust
- a SaySo server, broker, custom action protocol, or direct satellite-to-model
  connection

## Implementation map

| Concern | Location |
|---|---|
| Dataset generation | `training/scripts/build_synthetic_dataset.py`, `training/scripts/generate_training_supplement.py`, `training/scripts/generate_balanced_test_data.py` |
| Recipe-lock gold | `training/evals/recipe_lock.py`, `training/scripts/generate_recipe_lock_eval.py` |
| Raw tool-call parse | `training/evals/lfm_python_parse.py` |
| LFM adapter | `training/adapters/lfm.py` |
| Schema validation | `training/adapters/schema.py` |
| TRL recipe (checked-in) | `training/configs/lfm25-230m-synthetic-v2-trl.yml` |
| Evaluation | `evals/`, `training/evals/` |
| Pinned contract | `schemas/sayso-tool-schema-v1.json` |

Operational commands belong in `training/README.md`. Update this document only
when the training design or its safety boundary changes.
