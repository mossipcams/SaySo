# SaySo model training design

Use the [local training lifecycle](SAYSO_TRAINING_LIFECYCLE.md) for commands,
validation gates, dataset and model promotion, and run metadata. The active
trainer settings live in `training/scripts/train_unsloth_full.py`; host details
and generator notes live in [training/README.md](../training/README.md), with
run history in [training/TRAINING_LOG.md](../training/TRAINING_LOG.md).

SaySo trains `LiquidAI/LFM2.5-230M-Base` with full-parameter Unsloth training.
The wake-word model has a separate [LiveKit pipeline](../training/wake/README.md).

## Training contract

`schemas/sayso-tool-schema-v2.json` is the pinned training contract for tool
names, descriptions, arguments, and constraints. Every expected tool call must
pass that schema before it enters a dataset. Update the pinned artifact and
regenerate data when the Home Assistant contract changes; do not maintain a
second tool catalog.

`ALLOWED_HASS_TOOLS` gates dataset construction against the pinned contract. It
does not define runtime capability: Home Assistant supplies the tools available
for each request and owns entities, permissions, schemas, execution, and the
voice pipeline. llama.cpp hosts inference only.

## Training examples

Canonical examples use structured messages and OpenAI-compatible tools, with
`function.arguments` stored as validated JSON strings. Tool results represent
Home Assistant responses. `train_on_turn` marks supervised assistant calls and
final responses. Labels come from deterministic scenario facts and schema
validation, not from model-generated answers.

Coverage counts actual supervised behavior: a tool is positive only when a row
calls it for a real supported target. Refusals and unavailable targets have
separate coverage. Offering a tool in the schema does not count as positive
coverage. Generation fails when a retained supported tool or valid
domain/action combination lacks positive examples.

Do not train on ChatML `<tool_call>` labels, eval cases in `evals/cases/`,
recipe-lock utterances, unsupported calls, or unreviewed model-generated labels.
The eval suites remain held out for scoring and promotion.

## Formatting and loss

The canonical SaySo dataset remains structured. `training/adapters/lfm.py`
produces the prompt/completion view used by the existing trainer, rendering with
the pinned Base tokenizer and chat template. Each supervised assistant turn gets
its own view; prompt and history tokens are masked from loss, while the selected
target and its end marker are supervised. Preflight checks the context limit and
rejects rows that lose context or supervision.

The canary runs through the same trainer, tokenizer, formatting, masking, model,
and optimizer settings as full training. Only its step limit and output
location differ. Keep the Base revision and relevant tokenizer/template
identity with run metadata for reproducibility.

## Runtime safety boundary

Training data does not replace runtime validation. Home Assistant remains
authoritative for current entities and exposed tools. Before execution, SaySo
validates model-generated tool calls against the current request schema and
retains the existing fail-closed safety, ambiguity, and capability barriers.
Where implemented, SaySo verifies outcomes after actions. The integration is
the only bridge between model output and Home Assistant execution.

For the runnable offline evaluation path, use the existing suites and shared
runner/scorer under `evals/`. The local lifecycle runs the locked gold suite,
smoke/gauntlet checks, and final evaluation before promotion.
