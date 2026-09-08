# SaySo training plan

**Scope:** the stable design and constraints for `LFM2.5-230M-Base` supervised
fine-tuning. Rules that outlive any one run belong here. What ran, what it
scored, and which checkpoint is promoted belong in
[training/TRAINING_LOG.md](../training/TRAINING_LOG.md); commands and active
config belong in [training/README.md](../training/README.md).

SaySo trains `LiquidAI/LFM2.5-230M-Base` to select and emit Home Assistant tools in
the same OpenAI-compatible shape used by the integration. Home Assistant still
owns the entities, schemas, permissions, execution, and final voice pipeline.
llama.cpp only hosts inference.

Do not revive Instruct-checkpoint, FunctionGemma, or Axolotl full-SFT
plans. Checked-in `training/configs/lfm25-230m*.yml` and `functiongemma-270m*.yml`
Axolotl files are leftovers, not the training path.

## 1. Training contract

Use `schemas/sayso-tool-schema-v2.json` as the pinned training contract. It is a
snapshot of the Home Assistant Assist LLM API and defines the tool names,
descriptions, parameters, required fields, and constraints used to validate
examples. The flat `tools` list is the training source of truth; the same
artifact also groups tools by device-type tier (`query`, `generic`, `light`,
`fan`, `climate`, `media_player`, `vacuum`, `timer`) for catalog validation.
`schemas/sayso-tool-schema-v1.json` remains as a flat-only historical artifact
with an identical tool list and fingerprint.

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

The expanded v3 quality gate adds locked gold plus shadow rows from
`training/evals/v3_quality.py` and
`training/scripts/generate_v3_quality_eval.py`. Gold covers climate setpoint,
media play/pause/volume/mute, timer start/pause/status/cancel, vacuum
start/return/clean area, scene activate, script run, plus ordinary on/off,
status, ambiguity, and unsupported/no-call. Labels validate against
`sayso-tool-schema-v2` only. Shadow uses different homes, entities, and
phrasing. Gold, shadow, and recipe-lock prompts come from
`excluded_train_prompts()`; the v3 generator rejects any train utterance in that
set (`quality_eval_overlap`) instead of filtering contaminated rows out
afterwards. Generation is reproducible — the same seed yields a byte-identical
dataset across processes — so never seed generator randomness with builtin
`hash()` on a string.

Train each run from Base, not by continuing a previously merged checkpoint.

A run trains on one deterministic corpus. Do not blend corpora to make a set
larger: read the gold and shadow results first, then refine the cases the run
actually gets wrong and regenerate. Adding data before that evidence exists
hides which cases are weak. Corrective rows, when a refinement pass calls for
them, must use fresh homes, entities, and wording.

Shadow eval is 100–150 cases covering the same concepts as its gold set, with
different entities and phrasing. Promote only when gold and shadow both move the
right way. If only gold improves, the run is overfitting the benchmark.

Score generations with the apostrophe-safe parser in
`training/evals/lfm_python_parse.py` (raw `/completion` text). llama.cpp
structured `tool_calls` truncates names such as `O'Malley's` and `Kids'`; that is
a serving bug, not a training label. Do not retrain to paper over it, and never
compare a score from one scorer against a score from another — record which
scorer produced each result.

`training/scripts/generate_balanced_test_data.py` builds the 2,500-example
held-out set. Do not train on those prompts.

Which checkpoint is currently promoted, what each run scored, and which corpus
it used belong in [training/TRAINING_LOG.md](../training/TRAINING_LOG.md), not
here.

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
- fine-tuning on ChatML `<tool_call>` labels
- long autonomous chains
- replacing Home Assistant validation with model trust
- a SaySo server, broker, custom action protocol, or direct satellite-to-model
  connection

## Implementation map

| Concern | Location |
|---|---|
| Dataset generation | `training/generators/`, `training/scripts/build_synthetic_dataset.py`, `training/scripts/generate_training_supplement.py`, `training/scripts/generate_balanced_test_data.py` |
| Recipe-lock gold | `training/evals/recipe_lock.py`, `training/scripts/generate_recipe_lock_eval.py` |
| V3 quality gold + shadow | `training/evals/v3_quality.py`, `training/scripts/generate_v3_quality_eval.py` |
| Raw tool-call parse | `training/evals/lfm_python_parse.py` |
| LFM adapter | `training/adapters/lfm.py` |
| Schema validation | `training/adapters/schema.py` |
| TRL recipe (checked-in) | `training/configs/lfm25-230m-synthetic-v3-40k-trl.yml` |
| Evaluation | `evals/`, `training/evals/` |
| Pinned contract | `schemas/sayso-tool-schema-v2.json` (§1; v1 is a historical artifact) |
| Run history and scores | `training/TRAINING_LOG.md` |

Operational commands belong in `training/README.md`. Update this document only
when the training design or its safety boundary changes.
