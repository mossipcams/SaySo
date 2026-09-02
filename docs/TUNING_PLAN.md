# SaySo model tuning strategy

Status: approved data-then-tune order  
Companion: [Architecture](ARCHITECTURE.md), [Evaluation](EVALUATION_PLAN.md)

Do not SFT LFM2.5-230M until this pipeline has produced a SaySo-format train
mix and a frozen held-out eval set. Home-LLM's ChatML / `<tool_call>` labels
are not the training target. The tuned model must emit one `ControlPlan`
JSON object from `build_lfm_prompt`.

## Order

```text
home-LLM synthetic generator
        ↓
SaySo-modified generator
        ↓
SaySo voice dataset
        ↓
adversarial / eval dataset
        ↓
tune
```

Each stage feeds the next. Tune is last. Do not run a "quick" Home-LLM-format
LoRA on the 1070 as a stand-in.

## 1. Home-LLM synthetic generator

Use `acon96/home-llm` `data/generate_data.py` as the utterance and household
graph factory (rooms, devices, states, intents, follow-ups).

Keep:

- device/area randomization
- English command coverage
- `--sample` / `--train` volume knobs

Discard as labels:

- OpenAI-style tools
- Home Assistant service-call / `<tool_call>` assistant turns
- ChatML that is not a SaySo prompt

This stage is raw material, not a dataset the trainer may load.

## 2. SaySo-modified generator

Fork the generator so every kept row is a SaySo training example:

- user: `build_lfm_prompt(...)` (instruction + origin, conversation_state,
  candidate_entities, user_text)
- assistant: one `ControlPlan` JSON object (`action`, `query`,
  `clarification`, `unsupported`, or `no-action`)
- targets are semantic names / aliases, never entity IDs
- candidates and origin match the prompt the runtime will actually send

Map only what ControlPlan can express. Drop rows that require arbitrary HA
services, invented tools, or entity-ID targeting. Invalid ControlPlan JSON
fails generation, not training.

CLI:

```bash
uv run python -m train <home-llm.jsonl> <sayso-train.jsonl>
```

Output is train-only JSONL (`prompt` + `completion`) under a train-only path,
never `evals/datasets/`.

Current mapper keeps single-tool action rows. Status/refusal/query Home-LLM
rows without tool calls are dropped until the mapper learns those outcomes.

## 3. SaySo voice dataset

Add the transcripts Assist STT will actually produce:

- casual household phrasing
- fillers, self-corrections, room-relative "in here"
- deterministic ASR substitutions and dropped words
- later: recorded clips through the real Assist STT, stored as text plus
  audio handle (audio is not the SFT target)

Same prompt/ControlPlan contract as stage 2. Volume can be smaller than the
synthetic set; it must be present so the tune is not clean-text-only.

## 4. Adversarial / eval dataset

Freeze a reviewed held-out set before any SFT:

- existing authored corpora: `evals/datasets/{core,safety,language_noise,followup}.jsonl`
- additional adversarial cases in the same `EvalCase` schema: lookalike
  names, exclusions, negation, expired follow-ups, mixed current states,
  ASR-corrupted action cases

Rules:

- `case_id` values in eval never appear in train
- eval remains authored and reviewed; generator output is not counted as
  eval coverage
- a disjoint adversarial *train* slice may be produced by stage 2, but it
  must not clone eval IDs, wording, or expected plans

This set is the Gate B/D measuring stick, not extra SFT fuel.

## 5. Tune

One LoRA SFT of `LFM2.5-230M` on the concatenated train mix from stages 2–3
(plus any disjoint adversarial train slice):

- prompt and completion match production (`GENERATION_INSTRUCTION` +
  ControlPlan JSON)
- fp16, SDPA, no flash-attn, no `torch.compile` on the GTX 1070 (sm_61)
- keep llama-server stopped while the GPU trains
- merge adapter, export GGUF/MLX, then score the frozen eval set

Do not resume or merge a checkpoint trained on Home-LLM tool-call format.

## Non-goals

- Teaching the model to emit Home Assistant service calls
- Training on eval JSONL
- Axolotl CUDA 12.8 images (dropped Pascal)
- Model bake-offs as a substitute for this mix

## Current stop point

The SaySo-modified generator exists under `train/`. Next is the voice dataset,
then freeze eval, then tune. Do not launch SFT.

```bash
uv run pytest train/test_generator.py
uv run python -m train train/fixtures/sample.jsonl /tmp/sayso_train.jsonl
```
