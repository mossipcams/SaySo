# SaySo model training design

This document retains the model, data-contract, and runtime-safety design
constraints plus the historical LLaMA-Factory proposal. It is not the operator
guide for current training. Use the [local training lifecycle](SAYSO_TRAINING_LIFECYCLE.md)
for commands and gates, [training/README.md](../training/README.md) for host and
generator notes, and [training/TRAINING_LOG.md](../training/TRAINING_LOG.md)
for run history and scores.

The active trainer is full-parameter Unsloth training of
`LiquidAI/LFM2.5-230M-Base` on the `llm` VM. The existing SaySo contract remains
binding: Home Assistant owns entities, schemas, permissions, execution, and the
voice pipeline; llama.cpp hosts inference only. The current checkpoint
promotion sequence is separate from artifact publishing or deployment.

**Historical stack proposal:** LLaMA-Factory was evaluated on the 24 GB Radeon
RX 7900 XTX, then replaced by Unsloth. LLaMA-Factory recipes and settings below
are design history, not active configuration. The active trainer settings live
in `training/scripts/train_unsloth_full.py`; the local promotion flow is
specified in the lifecycle guide. Wake-word training remains a separate
[LiveKit pipeline](../training/wake/README.md).

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

Synthetic generation in `training/generators/pipeline.py` owns
utterance diversity; schema validation remains authoritative for every label.

Quotas are accounted on the supervision a row carries, not on its metadata.
`training/generators/coverage.py` classifies each accepted row by outcome
(action, state query, clarification, absence, unsupported), tool, domain and
targeting mode. A row satisfies an operation's positive quota only when it calls
that operation's tool on that capability's domain against a real target;
refusals fill a separate negative allowance, and offering a tool in the schema is
never coverage. Every retained supported tool and valid domain/action combination
must have positive rows, or generation fails. Tools deliberately outside the
declared coverage (`TRAINING_COVERAGE_EXCLUDED`) are also withheld from distractor
sampling, which is a dataset-scope decision and says nothing about which tools
Home Assistant supplies at runtime.

Generate actions only where the contract and the entity permit them. Operations
declare the entity features they need and Home Assistant's `supported_features`
supplies them, so a media player without power control never receives a
`HassTurnOn` label.

`training/generators/grounding.py` holds paired scenarios in which the request is
fixed and the entity graph moves — renamed, relocated, aliased, re-domained, or
differently capable targets, with individual and area targeting, genuine
ambiguity and presence/absence pairs. Expected behavior is derived from the
supplied graph and the current tool contract before any OHF wording or
paraphrasing is applied.

Both synthetic rendering paths share `training/generators/utterances.py`.
English phrasing uses a pinned OHF-Voice/intents subset through Hassil, with
literal home names and aliases bound to grammar slots. The adapter must preserve
all supplied settings, scopes and calls; unmatched combinations use an explicit
semantic fallback. Generation filters incomplete recognition fragments before
adding conversational framing. Numeric validation accepts digits and the number
words emitted by the existing STT transform, and rejects missing settings.
See `training/generators/OHF_SOURCE.md` for provenance and reproduction.

Corpus review must report actual upstream grammar sources, fallback usage,
multi-call/exclusion/alias coverage and label consistency, in addition to unique
request counts. Never regenerate a running job's pinned input in place.

## 3. Format and trainer

The render/loss invariants below apply to the current Unsloth trainer. Tooling-specific LLaMA-Factory details are historical.

### Dataset view and serving-format parity

Keep two representations separate:

- The canonical SaySo dataset keeps OpenAI-compatible `function.arguments` as
  validated JSON strings.
- The derived LLaMA-Factory view renders the pinned Base `chat_template.jinja`,
  temporarily parsing argument strings into objects for that rendering only.
  Canonical records remain structured; generic ChatML `<tool_call>` labels are
  still forbidden. The native LFM delimiters belong only to the rendered view.

```text
canonical JSONL (OpenAI messages, string arguments)
        ↓ pinned native LFM template + one view per supervised assistant turn
rendered prompt/completion pairs + source/turn lineage
        ↓ LLaMA-Factory full SFT + ROCm PyTorch
complete HF weights + tokenizer/template (no adapter merge)
        ↓ llama.cpp convert + Q8_0
GGUF used by embedded llama.cpp or external llama-server --jinja
        ↓ SaySo still supplies structured messages and current HA tools
```

Upstream LLaMA-Factory v0.9.5 includes `lfm2`, but its tool formatting differs
from the pinned SaySo Base template: tool-list/response wrappers, function
argument quoting, and the prefix are not identical. Its OpenAI converter also
combines adjacent tool results and changes system text. Do not select a template
by name and assume parity. These differences were found by source inspection,
not by a training run. See the upstream
[template](https://github.com/hiyouga/LlamaFactory/blob/v0.9.5/src/llamafactory/data/template.py),
[tool formatter](https://github.com/hiyouga/LlamaFactory/blob/v0.9.5/src/llamafactory/data/tool_utils.py),
and [converter](https://github.com/hiyouga/LlamaFactory/blob/v0.9.5/src/llamafactory/data/converter.py).

Use a small deterministic export in the existing training adapter:

1. For each assistant message with `train_on_turn: true`, render the complete
   preceding history and that target with the pinned native template. Export a
   prompt ending at the assistant generation boundary and its target completion.
   Keep the full tool catalog, original tool-result order, and exact prompt text.
2. Register the derived files through LLaMA-Factory's `dataset_info.json` using
   its ordinary prompt/response (`alpaca`) columns, mapped to `instruction` and
   `output`. This is a column format, not an Alpaca chat template. Do not populate
   separate system/tools columns: they are already in the rendered prompt.
3. Register a minimal `sayso_lfm2_rendered` pass-through template: no added
   prompt text, BOS, EOS, tool wrappers, or default system message. The rendered
   completion already contains the native end-of-message token. Keep the native
   tokenizer chat template for export/serving. This registration is planned work,
   not a built-in template or a second training loop.
4. Use `train_on_prompt: false` and `mask_history: false`. Each derived row has
   one target; all earlier turns, including invalid calls awaiting correction,
   are inside its masked prompt. Export both call and speech targets when both
   need supervision. Applying `mask_history: true` to an unexpanded conversation
   ending in speech would omit the earlier tool-call target.
5. Assert that the prepared token IDs equal the native full render and decoded
   non-ignored labels contain exactly the selected target, including its end
   marker. Fail on token-boundary drift, duplicated special tokens, dropped
   examples, or loss on user/system/tool/history tokens.

The 40,000-row budget below counts canonical scenarios. Derived training views
may exceed 40,000: record canonical count, view count, supervised-turn coverage,
and tokens separately, without counting an expanded scenario twice toward its
primary quota. Derive optimizer steps from view count and effective batch size.
This uses LLaMA-Factory's native prompt masking rather than relying on TRL's
`assistant_only_loss` or treating `train_on_turn` as an automatically supported
dataset field. Its [SFT processor](https://github.com/hiyouga/LlamaFactory/blob/v0.9.5/src/llamafactory/data/processor/supervised.py)
also truncates to `cutoff_len`; preflight must prevent that from losing context
or supervision.

### Full-parameter recipe

Train all Base parameters, including embeddings, convolution/attention blocks,
normalization, and the tied output head. Disable PEFT/LoRA and quantized training;
assert that the trainable parameter count equals the total unique parameter
count. Resume only an interrupted instance of the same full-training run with
its optimizer/scheduler state, never a historical merged model.

Proposed LLaMA-Factory settings, subject to a GPU smoke on the actual 24 GB card:

| Setting | Initial value |
|---|---|
| Method | `stage: sft`, `do_train: true`, `finetuning_type: full`; no adapter or quantization configuration |
| Template | Planned `template: sayso_lfm2_rendered`, as defined above |
| Weight/optimizer storage | FP32 parameters, gradients, and AdamW states; `pure_bf16: false` |
| Compute | `bf16: true`, `fp16: false` only after ROCm BF16 forward/backward passes; otherwise separately validate FP16 with loss scaling |
| Attention | `flash_attn: sdpa`; `disable_gradient_checkpointing: false`; verify training disables the KV cache |
| Sequence limit | `cutoff_len: 8192`, including prompt, tools, history, results, and labels |
| Batch | `per_device_train_batch_size: 1`, `gradient_accumulation_steps: 32`; try 2 × 16 only after memory measurement |
| Optimizer | `optim: adamw_torch`, `learning_rate: 2e-5`, `lr_scheduler_type: cosine`, `warmup_ratio: 0.03`, `weight_decay: 0.01`, `max_grad_norm: 1.0` |
| Loss | `train_on_prompt: false`, `mask_history: false` on the single-target view |
| Duration | `num_train_epochs: 2`, `save_strategy: steps`, `save_steps: 250`; development evaluation at the same interval; retain epoch checkpoints separately |
| Packing | `packing: false` for the first run |

These are initial experiment choices, not measured optimal hyperparameters.
Do not carry the old adapter learning rate of `2e-4` into full fine-tuning.
For 230M parameters, FP32 weights + gradients + two Adam moments total roughly
3.43 GiB (`230M × 16 bytes`), before activations, logits, temporary buffers,
and allocator overhead. The 65,536-token vocabulary makes long-row logits
material; 24 GB does not establish a safe batch size by itself.

Run 20 optimizer steps spanning the longest rows and each supervision shape.
Require finite losses/gradients, actual updates in representative embedding,
convolution, attention, and output weights, successful save/reload, and at least
10% free device memory at peak. Record tokens/second, assistant tokens/second,
peak allocated/reserved VRAM, and wall time. Estimate duration from that run;
do not extrapolate the GTX 1070's timing. The real run starts from fresh Base.

### ROCm environment and compatibility gate

Use a fresh ROCm environment, not the inspected NVIDIA host's virtualenv. Plan
for native Linux; if the chosen host uses WSL or Windows, validate that exact
platform's training support instead. Record GPU model and `gfx` target, dedicated
VRAM, OS/kernel, driver, ROCm/HIP, Python, PyTorch wheel or container digest,
Transformers, LLaMA-Factory release/commit, and resolved dependencies. Select a
supported combination from [AMD's compatibility matrix](https://rocm-handbook.amd.com/projects/amd-rocm-programming-guide/en/latest/compatibility/compatibility-matrix.html).
An MI300X tutorial or a nominal 24 GB capacity does not qualify a different GPU.

Use LLaMA-Factory v0.9.5 as the inspected compatibility reference, not a claim
that it has passed on this card. Its
[dependency constraints](https://github.com/hiyouga/LlamaFactory/blob/v0.9.5/pyproject.toml)
do not match the old host's newer Transformers/TRL stack. Pin one supported
release/commit and its compatible dependencies, including any internal TRL
dependency, without bypassing version checks. Install the ROCm PyTorch build
first and ensure subsequent dependency resolution preserves it. Record the Base
revision and native tokenizer/template hashes; the previously inspected Base
revision is `9d2be5519834990d30996f878b6771cccbd24f2c`.

Preflight must confirm nonempty `torch.version.hip`,
`torch.cuda.is_available()`, the expected device/VRAM, and a real on-device
forward/backward/update with LFM's convolution and attention blocks. ROCm uses
PyTorch's `torch.cuda` API names; they do not mean that a CUDA wheel is required.
Use those memory/timing APIs plus `rocminfo` or AMD SMI for the inventory.
[PyTorch HIP semantics](https://docs.pytorch.org/docs/main/notes/hip.html)
documents this distinction.

Start with native attention and standard AdamW. Leave FlashAttention-2,
xFormers, Liger, Unsloth, bitsandbytes, fused optimizers, and the old Pascal
patches out of the first recipe. If SDPA falls back to a memory-heavy kernel,
measure the actual 8k backward pass; first reduce microbatch while retaining
accumulation and checkpointing. Add a compatible optimization only when that
measurement requires it. Do not silently shrink the context distribution,
offload the run to CPU, or substitute LoRA. ROCm full training remains a gate to
prove, and no wall-time or batch-size estimate is an observed result yet.

## 4. Data and eval

Locked gold, shadow, grounding, and the original 120 realistic cases live in
`evals/cases/`. Smoke (~24 promotion cases) is checkpoint selection;
promotion is the locked 120. Do not train on those utterances.
`evals.cases.excluded_train_utterances()` is the holdout set; the v3 generator
rejects any train utterance in that set (`quality_eval_overlap`), including
normalized prompts, instead of filtering contaminated rows out afterwards.
Generation is reproducible — the same seed yields a byte-identical dataset
across processes — so never seed generator randomness with builtin `hash()` on
a string.

Train each run from Base, not by continuing a previously merged checkpoint.
Evaluate the first 250-step checkpoint on smoke and development data using the
same tokenizer/serving configuration as Base. Keep both epoch checkpoints;
select using development evidence and smoke before the locked promotion run.
An early result does not trigger automatic deployment. Stop on numerical
failure or broken data/masks; use the predeclared two-epoch cap for an otherwise
healthy run.

A run trains on one deterministic corpus. Do not blend corpora to make a set
larger: read the gold and shadow results first, then refine the cases the run
actually gets wrong and regenerate. Adding data before that evidence exists
hides which cases are weak. Corrective rows, when a refinement pass calls for
them, must use fresh homes, entities, and wording.

Shadow eval is 100–150 cases covering the same concepts as its gold set, with
different entities and phrasing. Promote only when gold and shadow both move the
right way. If only gold improves, the run is overfitting the benchmark.

Score generations with the apostrophe-safe production parser in
`custom_components/sayso/lfm_parse.py` (raw `/completion` text). llama.cpp
structured `tool_calls` truncates names such as `O'Malley's` and `Kids'`; that is
a serving bug, not a training label. Do not retrain to paper over it, and never
compare a score from one scorer against a score from another — record which
scorer produced each result.

`training/scripts/generate_balanced_test_data.py` builds the 2,500-example
held-out set. Do not train on those prompts.

`evals/cases/regressions.jsonl` (tag `grounding`) holds the entity-grounding
regressions, including Living Room + `media_player.living_room_tv` named "TV" →
`intent__HassTurnOn(name="TV", domain=["media_player"])` in the production prompt,
context and tool-schema format, plus held-out variations with different names,
ids, areas, distractors and presence/absence conditions. Check targets and
arguments, not tool selection alone. Its prompts belong to
`excluded_train_utterances()`; neither they nor near-duplicate scenario variants may
be trained on.

Home Assistant is authoritative for which entities exist, which are exposed to
Assist, and what each can do. `fetch_ha_home.py` records `exposure_source` on
every snapshot, and only `assist_exposure` is a valid input for the home-specific
recipe. That recipe enables real-home mixing at an explicit nonzero rate;
`--synthetic-only` is the deliberate override. Entity-frequency caps and the
per-capability holdout still apply, and the manifest records requested versus
achieved mixing, real-home rows (rows, never targets), per-entity target counts,
positive coverage and the negative distribution.

Which checkpoint is currently promoted, what each run scored, and which corpus
it used belong in [training/TRAINING_LOG.md](../training/TRAINING_LOG.md), not
here.

Promote a checkpoint only when it improves target behavior without regressing
STT, status, no-call, multi-action, light/fan, or lock polarity. Then export
GGUF and verify with llama.cpp `--jinja`. Freeze a promoted champion and do
not overwrite its GGUF, merged weights, or epoch checkpoint.

Promotion, deployment, and shipping are three separate decisions. A model may be
deployed without being promoted, and shipped without either — Run 013 step-2500
is all three states at once. Record which one applies; never infer promotion
from the fact that a model is serving or published. Ship with
`scripts/publish_model.sh`, which pins the artifact by sha256.

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
- Axolotl, distributed training, and a larger model for this run
- fine-tuning on ChatML `<tool_call>` labels
- long autonomous chains
- replacing Home Assistant validation with model trust
- a SaySo server, broker, custom action protocol, or direct satellite-to-model
  connection

## Generator corrections vs locked eval categories

The canonical generator (`training/generators/`, recipe YAML under
`training/configs/generation/`) targets these promotion-suite failure classes.
This is a label/coverage intent map only — not a model accuracy claim.

| Generator correction | Intended eval categories |
|---|---|
| Status rows label `GetLiveContext`, not state-changing intents | `status` |
| Genuine ambiguity → clarify; context-resolved names → action | `ambiguity`, `aliases` |
| Exclusion rows call all non-forbidden targets | `exclusion` |
| Withheld-capability rows omit the tool from the offered catalog | `unavailable` |
| Distinct absence / unsupported / clarify families | `unavailable`, `ordinary` (refusal control) |
| `production_catalog(home)` for every row (no answer-first subsets) | `ordinary`, `climate`, `routines_vacuum`, `light_fan_settings`, `multi_action` |
| Area scenarios as a sibling family (subtracted from count, tagged `family="area"`) | area grounding in `ordinary`, `multi_action`, `exclusion` |
| Honest `GetDateTime` in catalog when exposed | general time queries (no dedicated promotion tag) |
| Contrast groups preserved in planning/split metadata | `status`, `ambiguity`, `unavailable`, `exclusion`, `multi_action` |

Allocation shares for accepted rows live in
`training/configs/generation/production.yaml`.
Those are the current v3 shares. The next-run budget below replaces them only
when implemented and audited; it is not available by running that YAML today.

### Family allocations and area rows

`GeneratorConfig.allocations` drives `FamilyTracker` in `planning.py`. Recipe-backed
runs (`recipe_path` set from `from_yaml`) always use this planner and never enter
`QuotaTracker`. Programmatic family runs pass `allocations=default_allocations()`
or explicit shares. Bare `GeneratorConfig()` with empty allocations still uses the
legacy tier/capability/operation `QuotaTracker` in `sampling.py` for colocated unit
tests only — not a production path.

Area scenarios are a **sibling family**, not a cross-cut inside primary families.
When `cross_cutting.area` is enabled, area minimums are subtracted from
`count`, the remaining slots are drawn from primary-family allocations, and each
accepted area row is tagged `family="area"`. An area row is not also counted as
`ordinary`, `status`, or `exclusion`. In the historical v3 recipe, “ordinary
40%” was 40% of `(count − area_total)`, not 40% of the full corpus.

### Cross-cutting rates (grounding and discrimination)

`grounding.rate` and `discrimination.rate` in the recipe are shares of accepted
rows. `enforce_rate_gate` in `rates.py` compares achieved share against the
reachable ceiling (`grounding_available_share` /
`discrimination_available_share`), not the raw recipe value when capacity is lower.

The historical v3 recipe set `grounding.rate: 0.0`; active v5 sets it to `0.08`.
Promotion failures had been clarify/status/exclusion/unavailable, so v3 did not
spend rows on grounding until the catalogue and gate could support it.

`discrimination.rate` is capped at the measured delivery ceiling
(`DISCRIMINATION_DELIVERY_CEILING = 0.02` in `scenarios/discrimination.py`).
The historical v3 recipe requested `0.02`; active v5 requests `0.015`.

## Implementation map

| Concern | Location |
|---|---|
| Local candidate/promotion workflow | `./sayso`; [lifecycle guide](SAYSO_TRAINING_LIFECYCLE.md) |
| Static dataset preflight | `scripts/preflight.py`, `training/configs/preflight.yaml`, `training/configs/training_baseline.json` |
| Unsloth full trainer and checkpoint scoring | `training/scripts/train_unsloth_full.py`, `training/scripts/eval_checkpoint_cpu.py`, shared `evals/` runner/scorer |
| Canonical generator and recipes | `training/generators/{cli,config,pipeline,planning,rendering,validation,manifest}.py`, `training/configs/generation/` |
| Dataset build entry points | `training/generators/cli.py`, `training/scripts/generate_balanced_test_data.py` |
| Scenario facts and area families | `training/generators/scenarios/`, `training/generators/grounding.py` |
| Coverage accounting and audit | `training/generators/coverage.py`, `training/generators/planning.py`, `training/generators/audit.py` |
| Entity-grounding families | `training/generators/grounding.py`, `evals/cases/regressions.jsonl` |
| Real-home export and mixing | `training/scripts/fetch_ha_home.py`, `training/scripts/ha_websocket.py`, `training/generators/real_home.py` |
| Recipe-lock / quality / grounding eval | `evals/cases/regressions.jsonl` |
| Promotion and smoke | `evals/cases/realistic_v3.jsonl`, `evals/suites/` |
| LFM Python parse | `custom_components/sayso/lfm_parse.py` |
| LFM adapter | `training/adapters/lfm.py` |
| Schema validation | `training/adapters/schema.py` |
| Historical TRL adapter recipe | `training/configs/lfm25-230m-synthetic-v3-40k-trl.yml` |
| Historical LLaMA-Factory recipes | `training/configs/lfm25-230m-full-24gb-rocm-llamafactory*.yml` (not active) |
| Rendered training view | `training/adapters/lfm.py`, `training/scripts/export_llamafactory_view.py`; consumed by Unsloth |
| Evaluation | `evals/` |
| Pinned contract | `schemas/sayso-tool-schema-v2.json` (§1; v1 is a historical artifact) |
| Run history and scores | `training/TRAINING_LOG.md` |

Operational commands belong in [the training lifecycle guide](SAYSO_TRAINING_LIFECYCLE.md).
Update this document only when the training design or its safety boundary changes.

## Historical corpus and full fine-tuning proposal (2026-09-22)

This proposal records the defect analysis that informed later generator work. Its
LLaMA-Factory commands, settings, quotas, and implementation sequence are not the
active workflow; use the lifecycle guide and current training recipe for runs.
Keep the model at 230M parameters; use available VRAM for full optimization and
realistic contexts, not automatic corpus or model growth.

### Defects that determine the data

| Evidence | Training response | Separate requirement |
|---|---|---|
| [#39: valid tools refused or misrouted](https://github.com/mossipcams/SaySo/issues/39); its full-catalog probe passed 11/27 routes | Positive supervision for every standard tool, confusing sibling tools, and exact target arguments with the full catalog present | Rebaseline through the current scorer; old 293-case totals are not scores on today's suites |
| [#94: TV target mismatch](https://github.com/mossipcams/SaySo/issues/94#issuecomment-5769959131) | Separate canonical names from areas; never invent an area-plus-name string. Include duplicated names across areas and genuinely unresolved noisy targets | Confirm HA's accepted arguments and exposure; training cannot repair HA matching or replace pre-execution checks |
| [#102: letter-spaced STT TV (`T V`)](https://github.com/mossipcams/SaySo/issues/102) | Keep gold `name="TV"`; add meaning-preserving utterance variants (`T V`, `T.V.`, `teevee`) like existing `out let` STT noise — not area-prefixed names (#94) | Optional HA alias `T V` is a runtime bandage; confirm failure mode from trace (`T V` vs `Living room TV`) before closing |
| [#97: noisy STT emits unavailable tools/malformed calls](https://github.com/mossipcams/SaySo/issues/97) | Pair recoverable transcripts with correct actions; missing intent/target information requires clarification or safe no-action speech | Preserve schema validation, bounded correction, and fail-closed behavior |
| [#96: junk no-tool turns counted as success](https://github.com/mossipcams/SaySo/issues/96) | Distinguish action, query, clarification, conversational reply, and junk outcomes in corpus/eval reporting | Trace classification is a runtime observability fix; a model cannot change the trace success field |
| [#95: CPU-local STT/model latency](https://github.com/mossipcams/SaySo/issues/95) | Brief outputs; production-length schemas and context; report generated tokens and follow-up count | Keep CPU-local inference as the deployment target. Training hardware does not fix Whisper contention or prompt prefill |

The existing handoff records 96.3% verbatim target naming in Run 013 and 91.4%
in the subsequent gauntlet corpus. Those are historical measurements, not an
audit of a new corpus. When this proposal was written, v3 disabled grounding
and requested 2% discrimination. The later v5 recipe enables grounding at 8%
and requests 1.5% discrimination. Merely increasing a rate cannot create valid
sibling entities or new contrast scenarios; scenario diversity must be built.

The September 17 generator handoff is also historical: the current code already
checks family outcomes and includes positive GetDateTime generation. Reuse those
fixes rather than treating every old finding as open. GitHub #49 and #52 were
closed at inspection; their regression coverage remains useful. The only open
non-defect issue was the dependency dashboard (#63).

### One reviewed 40,000-row training corpus

Build one fresh corpus through `generators.cli`; do not concatenate old gauntlet,
HA-contract, Haven, and live transcripts. Host presence does not establish label
quality or permission to reuse a row. Use existing reviewed scenario/gold logic
and pinned OHF wording, adding only the behavior needed below. Labels come from
facts and schemas, never from the old model's answer.

Proposed accepted training-row budget (exactly 40,000; one primary bucket per
row). These are semantic audit buckets, not YAML keys that already exist:

| Primary behavior | Rows | Share |
|---|---:|---:|
| Ordinary single actions across all supported tools/domains | 12,000 | 30% |
| Entity resolution, aliases, explicit/implicit area and floor targeting | 8,000 | 20% |
| Settings: light, fan, climate, volume, numeric/unit boundaries | 4,000 | 10% |
| State, timer-status, and date/time queries with grounded answers | 4,000 | 10% |
| Multi-action and exclusion requests | 3,200 | 8% |
| Short follow-ups and user corrections with sufficient history | 2,400 | 6% |
| Genuine ambiguity and incomplete command clarification | 2,400 | 6% |
| Absent entities, missing tools, and observable unsupported capability | 1,600 | 4% |
| Junk/non-command transcripts and brief conversational replies | 1,600 | 4% |
| Eligible pre-execution correction and tool-result follow-up behavior | 800 | 2% |
| **Total** | **40,000** | **100%** |

Within the last bucket, allocate 400 correction rows and 400 result-follow-up
rows; within multi-action/exclusion, reserve at least 1,200 exclusion rows. Keep
all existing safety, lock polarity, timer, query, and follow-up concepts. Reserve
at least 200 positive rows per standard tool and 40 per valid domain/operation
combination, counted from actual calls. These floors overlap the primary budget;
they are not additional rows. Maintain a separate positive floor for per-home
script tools. Capacity checks must show all floors fit before generation.

Additional distributions overlap the primary buckets:

- **Grounding:** no more than 60% verbatim target naming among target-bearing
  rows, using `audit_discrimination.py`. At least 20% of those rows must require
  resolving a target among same-domain competitors using visible context;
  aliases alone do not satisfy this second floor. Vary names, areas, distractors,
  exposed tools, and target presence in paired scenarios. Preserve canonical
  output names and add area only when needed and accepted by the active schema.
- **STT:** target 20% meaning-preserving noise across actionable/query rows and
  their clean counterparts; count accepted transformed rows. Keep action words,
  negation, quantities, exclusions, and intent intact. Meaning-destroying noise
  belongs in clarification/junk buckets with newly derived labels. A lost
  “off” must never retain its original action label by assumption. Review noisy
  examples using the transcript and context the model actually receives.
- **Context:** target 60% of complete rendered rows at ≤4,096 tokens, 30% at
  4,097–6,144, and 10% at 6,145–8,192. Include the observed 30–32-tool prompt
  shape around 6k tokens. Grow actual home/context diversity, not filler text;
  retain short realistic homes. No answer-conditioned catalog filtering.
- **Real home:** target 10% only with a fresh `assist_exposure` snapshot and
  entity-frequency caps. Keep the existing per-capability entity holdout.
  Otherwise explicitly version the recipe as synthetic-only; never disguise
  the checked-in snapshot as a current export. Trace-derived rows must be
  reviewed, stripped of identifying data, and grouped by source session.

The current planner subtracts area and datetime reservations before distributing
primary shares. Translate the above budget into that accounting explicitly,
then audit the final total. No row may fill two primary slots. Context, noise,
real-home, and grounding counters report both numerator and denominator.

### Label and format requirements

1. Render the current production namespaced contract with the existing compiler:
   e.g. `intent__HassTurnOn`, `homeassistant__GetLiveContext`, and
   `media_player__HassMediaPause`. Preserve dynamic script tools. Validate each
   call against that row's offered schema, not just the bare-name catalog.
2. All decisions must be supported by model-visible information. The static
   overview contains names/domain/area, not arbitrary physical descriptions,
   live state, or every `supported_features` bit. Do not create contradictory
   labels from invisible features. Use an exposed schema or actual tool result
   where it supplies the necessary evidence; otherwise omit that negative.
3. Query results must carry the actual answer: the current renderer substitutes
   generic success for most tools. Add realistic date/time, timer, and state
   result shapes from the HA contract. Never teach a guessed value or “Done.”
   as a state answer. Keep final speech short and supported by the result.
4. Match runtime control flow. Single successful action turns currently return
   deterministic “Done.” in the integration, so their essential training target
   is the call. Queries and multi-call turns need result-to-speech supervision.
   Execution failure currently returns a controlled runtime error; do not invent
   a model recovery loop or train automatic retries of executed actions.
5. Correction rows use the existing correction prompt and only eligible failures
   before execution. Invalid historical calls are input context, never loss
   targets. Export only selected supervised turns as targets, as defined in §3;
   ordinary multi-turn assistant masking would also train invalid history.
6. Follow-up labels depend on the complete supplied history. Keep conversations
   together, including clarification then resolution and changes of target or
   setting. No pronoun resolution without a unique contextual referent.

### Splits, audit, and release gates

Create a separate 2,000-row development set from disjoint source groups; retain
the existing balanced 2,500-row holdout and the locked evaluation corpora.
Assign source groups before noise/paraphrasing/rendering. A home/scenario contrast
group, its aliases, conversation, and all transcript variants stay in one split.
The current splitter hashes template/phrasing/seed metadata; it must explicitly
retain this lineage instead of assuming different seeds establish independence.
Report achieved split sizes and tool coverage after grouping.

Before GPU work, require:

- Exact row counts and quotas; no mislabeled primary families or missing tool
  positives; no label contradictions for the same model-visible input.
- Zero overlap with `excluded_train_utterances()`, all held-out prompts, and
  source groups. Check normalized text and scenario lineage; exclude near-copy
  variants of locked cases, not just byte-identical utterances.
- Production parser/schema round trips for every supervised call; zero unknown
  tools, invalid arguments, invalid batches, or excluded targets in labels.
- Exact native-template and LLaMA-Factory-prepared token equality on every view,
  zero truncation/dropped rows, and correct nonempty target masks. Reconcile
  canonical and derived row counts by source/turn ID. The generator's
  character-count fallback is insufficient for this gate: tokenizer failure
  must stop a training build.
- Deterministic repeated pilot builds; inspect at least 20 rows per primary
  bucket plus all pilot corrections and noisy no-action cases. Resolve any
  contradictions before the full build. Keep a hash-pinned immutable manifest.

Freeze baselines for Base, the shipped Gauntlet artifact, and any later candidate
whose provenance is verified, all through the current `evals` runner and the
same production parser/contract. The host's Newhaven and HA-contract checkpoints
are candidates to inspect, not established champions. Never compare their old
report totals directly with the current locked 120.

Use the existing 24-case smoke plus development loss/behavior for checkpoint
selection. Evaluate selected and epoch checkpoints on gold/shadow regressions;
run the locked 120-case promotion gate after selection. Preserve its denominator
and `evals/config/gates.yaml` thresholds: ≥84/120 overall, existing category
floors, error-rate limits, and all zero-tolerance flags. Also require no regression
in the existing safety slices, and improvement on both gold/shadow evidence as
described in §4. Do not tune data or thresholds to a final promotion result.

Add only focused supplemental regressions needed by the defects, through the
same runner. An inventory of the current 329 canonical cases found positive
calls for 18 of the 27 standard tools. The nine missing are GetDateTime,
HassCancelTimer, HassIncreaseTimer, HassDecreaseTimer, HassUnpauseTimer,
HassMediaNext, HassMediaPrevious, HassMediaPlayerUnmute, and
HassMediaSearchAndPlay. Reuse existing cases and add the missing positives to
form a 27-tool matrix. Require 27/27 with the full catalog as well as the
isolated-tool diagnostic. Freeze a small noisy-STT regression set covering
recoverable, ambiguous, and junk inputs; require zero actions on ambiguous/junk
cases and report useful-command success separately from safe abstention.

Export complete HF weights directly to F16/Q8_0 GGUF. Verify tokenizer/template
and BOS/EOS parity, quoted names, batch parsing, and scores after quantization.
Recheck both embedded and external parsing paths where supported. Model
promotion requires these model gates; device-demo readiness additionally needs
live HA execution/state verification, no duplicate actions, truthful TTS, and
measured CPU-local latency. Track #94–#96 separately until their runtime evidence
passes. No training result alone closes an STT, matcher, or tracing defect.

### Implementation sequence and files

| Phase | Files to touch | Verification before proceeding |
|---|---|---|
| 1. Freeze contract and baselines | Existing schema fixtures only if live comparison proves drift; `evals/cases/regressions.jsonl` and `evals/homes/` for the focused missing coverage | Current parser/schema checks; 24/120 suite membership unchanged; baseline reports record contract/artifact hashes |
| 2. Make the corpus truthful | `training/generators/{planning,row_generation,rendering,validation,coverage,audit,manifest,config}.py`; existing `scenarios/`, `grounding.py`, and `stt_noise.py` as needed; `training/scripts/split_dataset.py`; generation recipes | Small deterministic pilot, quota/lineage/observable-label checks, existing generator suite, discrimination audit |
| 3. Make full SFT reproducible | `training/adapters/lfm.py`, `training/scripts/train_lfm.py`, `training/requirements.txt`, new `training/configs/lfm25-230m-full-24gb-rocm-llamafactory.yml` | Build the derived view and pass-through registration; use a thin in-process entry point that registers the template then invokes LLaMA-Factory's native SFT runner; verify ROCm dependencies, full trainable count, token/mask parity, and save/reload |
| 4. Build and train | Immutable canonical/view JSONL, `dataset_info.json`, manifests, and run bundle outside Git; `training/README.md` for implemented commands | All corpus/view audits; exact AMD GPU/ROCm preflight; 20-step optimizer/memory smoke; fresh-Base run |
| 5. Evaluate and qualify | `training/scripts/export_gguf.py` if replacing its current command-printing stub; existing `evals` runner; `training/TRAINING_LOG.md` | Checkpoint selection, frozen gates, executable export, Q8/serving parity, separate live-device checks |

Run the smallest existing relevant checks after each phase, including
`python -m pytest training/tests training/scripts training/adapters -q` for
generator/trainer changes and the existing eval/integration checks when their
contracts are affected. Add a small colocated runnable check for new logic;
do not modify files in any `tests/` directory or weaken existing assertions.
Do not regenerate a running job's input or overwrite a checkpoint. This plan
authorizes documentation only; implementation and execution are subsequent work.
