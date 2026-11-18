# SaySo language model training

Start with the [local training lifecycle](../docs/SAYSO_TRAINING_LIFECYCLE.md) for
the supported candidate-to-model-promotion workflow, which runs from
ajax-optimization `sayso-training/`. This page records VM setup and training
assets that remain in SaySo.

- Stable model design and constraints: [docs/SAYSO_LFM_TRAINING_PLAN.md](../docs/SAYSO_LFM_TRAINING_PLAN.md)
- Run history and scores: [TRAINING_LOG.md](TRAINING_LOG.md)

This directory holds dataset preparation and historical training assets for
`LFM2.5-230M-Base`. The active v5 path is a **full-parameter** fine-tune on the 24 GB AMD
host (`ssh llm`, see [Training host](#training-host-observed-2026-09-23)).
The active trainer is [Unsloth](scripts/train_unsloth_full.py), invoked by the
local promotion command. Canonical JSONL stays OpenAI-shaped on disk; training
consumes a derived prompt/completion view rendered with the Base chat template.
Older trainer recipes and smoke results are historical, not the v5 path.

SaySo has two models with separate pipelines: this one (LFM, Unsloth) and the
wake word (LiveKit), which has its own guide:
[training/wake/README.md](wake/README.md).

## Training host (observed 2026-09-23)

One VM does generation, training, and LLM serving. SSH alias `llm`
(`LLM@192.168.1.76`, key `~/.ssh/id_ed25519_sayso`). No passwordless sudo;
docker group yes. The VM has no GitHub credentials.

| Item | Observed |
|---|---|
| GPU | Radeon RX 7900 XTX, gfx1100, 24 GiB; host ROCm 7.2.4 |
| CPU / RAM | 8 vCPU; 19 GiB RAM + 12 GiB swapfile |
| Disks | SSD `/` 300 G (Docker images, models, runs; ~60 G free). HDD `/srv/llm/data` 400 G (datasets and generation code; ~361 G free). Layout plan: [docs/PLAN_VM_LAYOUT.md](../docs/PLAN_VM_LAYOUT.md) |
| Repo checkout | `/srv/llm/sayso` → `/srv/llm/data/sayso` (HDD), git tracking `origin/main`, read-only deploy key so the VM can `git pull`. Generation venv: `training/.venv`. |
| Base weights | `/srv/llm/lfm/models/LFM2.5-230M-Base` (SSD; `tokenizer_class` patched to `PreTrainedTokenizerFast`, original kept as `tokenizer_config.json.tf5`) |
| v5 datasets / runs | `/srv/llm/data/lfm/datasets/` (HDD), `/srv/llm/lfm/runs/` (SSD) |
| Training UI | Unsloth Studio container `unsloth` (`unsloth/unsloth-rocm:studio`), UI on host `:8002` (login required). Compose `/srv/llm/services/unsloth/`. Mounts only `/srv/llm/lfm` (rw), `data/lfm/datasets` (ro), `data/sayso` (ro) under `/workspace/host/`; shared SSD `hf-cache/` and SSD `services/unsloth/studio/`; `restart: unless-stopped` |
| LLM serving | vLLM container `vllm`, Qwen3.8-27B-INT4 on `:8000` at 0.98 GPU memory (vLLM v0.30.0, 64k context). Compose `/srv/llm/serve/vllm/docker-compose.yml`, never autostarts. `/srv/llm/bin/gpu serve` / `stop` |
| GPU sharing | vLLM, Unsloth, and LiveKit share one XTX. Every training run starts through `/srv/llm/bin/gpu train lfm|wake` (stops vLLM, holds the GPU, releases on exit); `gpu serve` / `gpu stop` / `gpu status` for vLLM. Source: `scripts/llm-host/gpu`. Runs started by hand in the Studio UI bypass the lock, so only explore there while `gpu status` shows `idle`. |
| Retired 2026-09-23 | Earlier trainer containers and launchers were removed; setup helpers are archived under `data/archive/task3-retired/`. |
| Migrated history | The retired NVIDIA host's `/srv/models`, `/srv/datasets`, `/srv/training-runs` are now `/srv/llm/data/lfm/legacy-140/{models,datasets,training-runs}` (HDD), cleaned 2026-09-23 (see `TRAINING_LOG.md` "Host cleanup"). Champion markers: `legacy-140/CHAMPION.md`. Paths in `TRAINING_LOG.md` and handoffs use the old prefix. |

### Retired host (`192.168.1.140`)

`ssh sayso-train` (`ubuntu@192.168.1.140`, hostname `ai-inference`, GTX 1070
8 GiB) ran every run up to Run 013. It was unreachable on 2026-09-23. It used
`/srv/training-runs/.venv` (Torch `2.13.0+cu126`, Transformers `5.16.1`, TRL
`1.12.0`, PEFT `0.20.0`, xFormers `0.0.35`, Liger Kernel `0.8.2`) with the TRL
launcher `/srv/training-runs/train_lfm2_liger.py`. That environment is not a
dependency lock for v5.

The later host recipes are still rsLoRA rank/alpha 32, batch 1/accumulation 16,
learning rate `2e-4`, and 8,192-token context. They use **FP32, xFormers, and a
custom Liger loss patch**, with checkpoints every 50 steps. Their headers record
Pascal memory/speed limitations behind those choices. The launcher applies both
patches in-process before TRL; a plain CLI invocation is not equivalent.

An earlier ROCm proposal called for full-parameter training, native SDPA,
standard AdamW, and BF16 autocast after validation on the chosen AMD card. The
old Pascal patches are historical. Current formatting and loss constraints are
summarized in the model training design.

Axolotl is not used. `training/scripts/train.py`, `train_lfm.py`, and
`configs/detect_gpu.py` are dead Axolotl launchers. `training/requirements.txt`
is the generator/view/test environment only (CPU, no torch, no trainer). The
GGUF export script prints instructions rather than executing a complete export.

## Quick start

Generator, data-view, and test environment (CPU). On the VM it lives at
`/srv/llm/sayso/training/.venv`:

```bash
cd training
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
HIP_VISIBLE_DEVICES= HF_HOME=/srv/llm/hf-cache .venv/bin/python -m pytest tests -q
```

## Generation and training workflow

The training-data generator, preflight, and the gated `./sayso` promotion and
training flow live in ajax-optimization `sayso-training/`, along with the
generator design notes (entity naming, real-home mixing, coverage gates,
grounding families). Follow the
[training lifecycle guide](../docs/SAYSO_TRAINING_LIFECYCLE.md) from that tree
on the `llm` VM.

## Eval sets and what each is for

Canonical cases live in `evals/cases/`. Suites are ID lists in `evals/suites/`.
See `evals/README.md`.

| Set | Rows | Role |
|---|---|---|
| `evals/suites/smoke.yaml` | 24 | Checkpoint selection. Subset of the locked 120. Never includes held-out grounding/gold/shadow/recipe-lock rows. |
| `evals/suites/promotion.yaml` | 120 | Locked realistic promotion gate, 10 per category. Supplemental regressions do not change this denominator. |
| `evals/cases/regressions.jsonl` | unique recipe-lock, quality gold/shadow, and grounding cases | Diagnostic regressions via `--tag`. Held out of training and of smoke. |
| `sayso_test_balanced.jsonl` | 2,500 | Broad held-out regression sample for dataset generation. Not the promotion gate. |

Promotion needs gold and shadow to move the right way together. Gold alone
improving means the benchmark is being overfit.

## Scoring

All eval results use the shared `evals.runner` and `evals.scorer`. Use the
endpoint CLI below for a running server; the checkpoint evaluator in the
[training lifecycle](../docs/SAYSO_TRAINING_LIFECYCLE.md) calls the same runner
and scorer directly.

```bash
python -m evals.cli run --suite smoke --adapter endpoint --server http://127.0.0.1:8080
python -m evals.cli run --suite promotion --adapter endpoint --server http://127.0.0.1:8080
python -m evals.cli run --tag grounding --adapter endpoint --server http://127.0.0.1:8080
```

Checkpoint evaluation uses `InMemoryAdapter` through the same runner and
scorer; its smoke and promotion gates are part of the lifecycle guide.

## Dataset views

- **canonical**: OpenAI-compatible envelope with JSON-string `function.arguments`
- **historical TRL render**: dict `function.arguments` for `apply_chat_template` only
- **Rendered training view**: prompt/completion pairs (`instruction` /
  `output`), one per `train_on_turn: true` assistant message; built by
  `adapters/lfm.py` and consumed by
  the Unsloth trainer

## Layout

```
training/
  adapters/          Schema validation and LFM helpers
  artifacts/         Checkpoints, eval outputs (gitignored)
  configs/           historical training recipes
  datasets/          Generated JSONL (gitignored)
  fixtures/          Test fixtures
  scripts/           Training, export, exposure, split, and verification operations
  tests/             Unit tests (no model downloads)
```

## Active trainer safeguards

The current Unsloth entry point starts from Base with full-parameter training,
an 8,192-token limit, BF16, batch size 1, gradient accumulation 32, and
completion-only labels. It checks ROCm availability, rejects rows beyond the
context limit or with no supervised completion, and requires promotion metadata
plus a matching rendered-data hash for full training. Static dataset checks and
the guarded promotion sequence are documented in the
[training lifecycle guide](../docs/SAYSO_TRAINING_LIFECYCLE.md).

### GPU reservation and cancellation

Use `/srv/llm/bin/gpu train lfm|wake [--serve-after] COMMAND ...` for every
training run. Commands must stay in the foreground. `gpu status` reports the
reservation, owner/worker liveness, and VRAM use. A dead wrapper or worker leader does not free
a reservation while non-zombie members of its process group survive.

SIGTERM, SIGINT, and SSH hangup terminate the worker group before releasing the
GPU. A failed or interrupted LFM command restarts Unsloth to stop container-side
workers; Studio briefly disconnects. If that cleanup fails, serving remains
blocked pending inspection of `/srv/llm/run/gpu.state` and the container.
`--serve-after` restarts vLLM after successful cleanup, including command failure;
without it, the GPU remains idle. Studio is for exploration while idle; launching
training through its UI bypasses the reservation and is unsupported.

After an LFM wrapper and Docker client both crash, `gpu status` reports
`orphan:lfm` without changing services. The next `serve`, `stop`, or `train`
command restarts Unsloth under the lock before reusing the GPU; restart failure
leaves it blocked. An interrupted launch with no recorded worker PID stays
blocked for process inspection instead of assuming the GPU is idle.
