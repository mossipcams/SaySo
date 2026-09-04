# SaySo LFM training plan

**Status:** minimal MVP plan for supervised fine-tuning.

The first Base run is [FIRST_TRAINING.md](FIRST_TRAINING.md) (`LiquidAI/LFM2.5-230M-Base`, rsLoRA). This document remains the dataset and safety contract.

SaySo trains `LiquidAI/LFM2.5-230M-Base` to select and emit Home Assistant tools in
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

Keep two representations separate:

- The canonical SaySo dataset keeps OpenAI-compatible
  `function.arguments` as validated JSON strings. This is the runtime contract
  and the format checked by the LFM adapter.
- The tokenizer-rendering view may parse those strings into native objects only
  while applying the LFM2.5 chat template. This view is transient; do not
  replace the canonical dataset or call the converted objects training labels.

The current LFM2.5 checkpoint ships a `TokenizersBackend` tokenizer and a
separate `chat_template.jinja` containing `{% generation %}` markers. The
initial GTX 1070 setup showed that an older Axolotl stack could not consume
those files directly, and its Jinja parser could not process the markers. Use a
version-pinned Transformers/Axolotl combination that supports the exact
checkpoint, or a tested SaySo-owned rendering compatibility layer. Do not
silently patch downloaded model files or vendor packages.

Before production training, the smoke gate must prove all of the following on
the target host:

1. the exact LFM2.5 tokenizer and model load;
2. a string-argument SaySo example renders successfully;
3. one forward/backward/update step completes; and
4. the saved checkpoint reloads and produces output through the planned
   llama.cpp verification path.

Use the checked-in Axolotl configurations:

| Run | Config | Purpose |
|---|---|---|
| Smoke | `training/configs/lfm25-230m-smoke.yml` | Fixture-only pipeline check |
| Production | `training/configs/lfm25-230m-prod.yml` | Train on generated data |

The production configuration currently declares:

- base model `LiquidAI/LFM2.5-230M`;
- full-parameter SFT;
- three epochs and learning rate `2e-4`;
- BF16 and Flash Attention disabled for Pascal;
- assistant-only loss masking through `train_on_turn`.

The GTX 1070 has 8 GiB VRAM and supports neither BF16 nor Flash Attention. In
the tested Axolotl/Accelerate environment, FP16 backpropagation failed with
`Attempting to unscale FP16 gradients`, so the successful one-step smoke run
used FP32. Treat precision as a host compatibility result, not an assumption:
keep FP16 only after the exact production stack passes the smoke gate; otherwise
set the host config to FP32 before training.

The smoke configuration is a one-epoch, smaller-sequence test. The current
floating-point `training/requirements.txt` is not a reproducible GPU lock: an
unconstrained install resolved a newer Torch/CUDA stack than this Pascal host
can safely use. Pin the complete environment, including Axolotl, Transformers,
tokenizers, Accelerate, Torch, and CUDA-compatible packages, before a
production run. Treat the checked-in configs as the executable source for
training values; tune only from held-out results, one change at a time.

Budget storage before training. The smoke run produced about 0.9 GB for the
final model and about 1.7 GB for a resumable optimizer checkpoint. Keep an
intermediate checkpoint only when resumption is needed; do not fill a nearly
full host with duplicate model copies.

## 4. Splits and evaluation

Generate deterministic train, validation, and test splits. Keep related
template, phrasing, and seed families in one split so paraphrase variants do
not leak across evaluation boundaries.

Use validation for checkpoint selection and tuning. Keep the test set and
`training/evals/adversarial.jsonl` untouched until the final comparison.

`training/scripts/generate_balanced_test_data.py` builds the deterministic
2,500-example final test set. It excludes exact prompts found in the generated
train and validation files and never consumes IDs or labels from `evals/cases/`.

The first supervised train set is 10,000 curated examples from
`training/scripts/build_synthetic_dataset.py` (see [FIRST_TRAINING.md](FIRST_TRAINING.md)
for the locked category mix). Held-out eval stays separate; do not train on
`sayso_test_balanced.jsonl` prompts.

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
Assistant. A successful trainer smoke run alone is not a promotion signal.

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
- fixed training-corpus percentage or ASR-corruption quotas;
- fine-tuning on Home-LLM tool-call labels;
- long autonomous chains;
- replacing Home Assistant validation with model trust;
- a SaySo server, broker, custom action protocol, or direct satellite-to-model
  connection.

## Implementation map

| Concern | Location |
|---|---|
| Dataset generation | `training/scripts/generate_dataset.py`, `training/scripts/build_synthetic_dataset.py`, `training/generators/` |
| LFM adapter | `training/adapters/lfm.py` |
| Schema validation | `training/adapters/schema.py` |
| LFM configs | `training/configs/lfm25-230m*.yml` |
| Evaluation | `evals/`, `training/evals/` |
| Pinned contract | `schemas/sayso-tool-schema-v1.json` |

Operational commands belong in `training/README.md`; update this document only
when the training design or its safety boundary changes.
