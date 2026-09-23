# v5 full SFT on 192.168.1.76 (LlamaFactory + 7900 XTX)

> **Status 2026-09-23 (evening): LlamaFactory removed from the host.** The
> `llm-rocm` container and image are gone, replaced by Unsloth Studio
> (`unsloth` container, UI `:8002`). `/srv/llm/scripts/llm train rocm` no longer
> works. The data contract below still holds; the LlamaFactory yml, launch
> commands, and HIP smoke are a record of what ran, not a runnable path. Current
> host layout: `training/README.md` § Training host. Paths below are as they were
> on 2026-09-23 before the reorg ([PLAN_VM_LAYOUT.md](PLAN_VM_LAYOUT.md)): Base is now
> `/srv/llm/lfm/models/`, datasets `/srv/llm/data/lfm/datasets/`, runs `/srv/llm/lfm/runs/`.

Get this round of training onto the 24 GB box. Do not use the host’s existing
LoRA / ShareGPT / `<|tool_call_start|>` smoke path. That contradicts
`docs/SAYSO_LFM_TRAINING_PLAN.md`. No eval/runtime edits. No commit unless asked.

## Host facts (2026-09-23)

| Item | Value |
|---|---|
| SSH | `LLM@192.168.1.76` (`Host llm`) |
| GPU | Radeon RX 7900 XTX, gfx1100, 24.0 GiB |
| Host ROCm | 7.2.4 |
| Train container | ~~`llm-rocm:torch2.7.1-rocm6.3-lf0.9.3`~~ removed 2026-09-23; now `unsloth` (`unsloth/unsloth-rocm:studio`, UI :8002) |
| Base weights | `/srv/llm/models/LFM2.5-230M-Base` (already downloaded) |
| Launch | ~~`/srv/llm/scripts/llm train rocm <yml>`~~ dead with `llm-rocm`; Unsloth recipe not written |
| GPU sharing | `vllm` (Qwen3.8-27B-INT4, `:8000`) holds ~97% VRAM when up. Training goes through `/srv/llm/bin/gpu train lfm` (see `training/README.md`). |
| Disk | SSD `/` ~56 GiB free; HDD `/srv/llm/data` ~232 GiB free (2026-09-23 evening) |
| RAM | 19 GiB + 12 GiB swapfile; 8 vCPU |
| Passwordless sudo | no; docker group yes |
| Train venv pin | transformers **4.55.4** + `DISABLE_VERSION_CHECK=1`. LF 0.9.3 pins `<=4.52.4` (no `lfm2`). 4.56.2 loads `lfm2` but breaks LF 0.9.3 `HfArgumentParser` (`ParallelismConfig`). |
| Tokenizer patch | `/srv/llm/models/LFM2.5-230M-Base/tokenizer_config.json` `tokenizer_class` is `PreTrainedTokenizerFast` (backup `tokenizer_config.json.tf5`; original was `TokenizersBackend`). |

## Contract for this round

- Model: `LFM2.5-230M-Base` (path above), **full** SFT, not LoRA, not Instruct, not continue-train.
- Data: v5 feasible mix (`training/configs/generation/full_sft_v5.yaml`).
- Labels: canonical OpenAI JSONL stays on disk. LlamaFactory trains a **derived**
  prompt/completion view: one row per `train_on_turn: true` assistant turn,
  rendered with the Base `chat_template.jinja`. Parse `function.arguments` to
  objects **only** for that render. No ChatML `<tool_call>`. No ShareGPT
  `tool_call_start` rewrite.
- `dataset_info.json` alpaca columns `instruction`/`output`. Template is a
  pass-through (`empty` if 0.9.3 has it, else register `sayso_lfm2_rendered`).
- `train_on_prompt: false`, `mask_history: false`, `finetuning_type: full`,
  `cutoff_len: 8192`, `learning_rate: 2.0e-5`, batch 1 / accum 32, `bf16: true`
  (HIP bf16 smoke passed 2026-09-23; fp16 had inf `grad_norm`).
- Invalid correction-history calls stay in the prompt, never in `output`.

## Files (repo)

- `training/adapters/lfm.py` — expand supervised turns into prompt/completion
- `training/scripts/export_llamafactory_view.py` — CLI over a canonical JSONL
- `training/configs/lfm25-230m-full-24gb-rocm-llamafactory.yml` — host-path recipe
- `training/tests/test_llamafactory_view.py`
- `training/README.md` — implemented host commands only

## Historical host layout (before Task 3)

```
/srv/llm/datasets/sayso_full_sft_v5.jsonl              # canonical, immutable
/srv/llm/datasets/sayso_full_sft_v5_rendered.jsonl     # derived view
/srv/llm/datasets/dataset_info.json                    # add sayso_v5_rendered
/srv/llm/config/sayso-lfm-v5-full.yml
/srv/llm/runs/sayso-lfm-v5-full/
```

## Sequence

1. Exporter + yaml + tests (smoke fixture is enough to prove the view).
2. Generate v5 40k **on the LLM box** (Mac generate was still empty-log / ~26–476 MiB after 20 min).
   Tree at `/srv/llm/sayso` (git checkout of `origin/main` since 2026-09-23; sync from the Mac, the VM has no GitHub access). Tokenizer path `/srv/llm/models/LFM2.5-230M-Base`.
   Do not grab the XTX (`HIP_VISIBLE_DEVICES=`). Slim deps in the train venv:
   `hassil==3.12.1` + `jsonschema` (no Axolotl). Smoke `configs/generation/smoke.yaml`
   first, then `full_sft_v5.yaml`. Output `/srv/llm/datasets/sayso_full_sft_v5.jsonl`.
3. Export the derived view on the box; merge `dataset_info` `sayso_v5_rendered`.
4. Free the XTX (today: `/srv/llm/bin/gpu train lfm ...` does it).
5. HIP smoke: 2-step full SFT on the rendered **smoke** view first, then 20
   optimizer steps on the longest v5 rows if VRAM allows. Require finite loss,
   `torch.version.hip` set, device = 7900 XTX.
6. Full 2-epoch train is **ready** after that smoke. Do not start the 2-epoch
   run in this setup pass unless the smoke is green and the 40k view is on disk.

## HIP 2-step result (2026-09-23)

`/srv/llm/scripts/llm train rocm /srv/llm/config/sayso-lfm-v5-smoke.yml` exited 0.

| Check | Result |
|---|---|
| HIP | `torch.version.hip` 6.3.42131, device `cuda:0` = AMD Radeon Graphics (7900 XTX) |
| Method | Full, trainable 229,693,184 (100%) |
| Compute | bf16 after a follow-up 2-step (`grad_norm` 66.1 finite). fp16 also trained but had inf `grad_norm` on some steps. |
| Labels | native `<\|tool_call_start\|>` from Base jinja; prompt tokens masked `-100`; no ChatML `<tool_call>` |
| Loss | 1.3565 finite; `train_runtime` 10.61s |
| Output | `/srv/llm/runs/sayso-lfm-v5-smoke/` |

20-step fp16 smoke (`sayso-lfm-v5-smoke-20.yml`) reached `learning_rate` 2e-5 by step 3; loss 1.3565 → 0.0001 on the 5-row set. Longest-row 8192 VRAM gate still waits on the 40k rendered view. Smoke `model.safetensors` were deleted to free disk (~24 GiB free); trainer logs remain under `/srv/llm/runs/`.

Host persistence (not git): `DISABLE_VERSION_CHECK=1` in `/srv/llm/config/runtime.env`; train wrapper re-pips `transformers==4.55.4` if missing; ROCm Dockerfile pins the same (rebuild not done this pass). Recreating `llm-rocm` without that pin loses `lfm2` load until the wrapper reinstalls it.

## Verification

```bash
cd training && .venv/bin/python -m pytest tests/test_llamafactory_view.py -q
ssh 192.168.1.76 'docker exec llm-rocm /opt/venv-train/bin/python -c "import torch; print(torch.version.hip, torch.cuda.is_available())"'
ssh 192.168.1.76 '/srv/llm/scripts/llm train rocm /srv/llm/config/sayso-lfm-v5-smoke.yml'
```

Smoke yaml is the full recipe with `max_steps: 2` and the rendered smoke file.
Fail closed on ChatML markers, truncated `cutoff_len`, or LoRA in the v5 yaml.

Task 3 service paths: vLLM compose, secrets, weights, and kernel caches are
under `/srv/llm/serve/vllm/`; Unsloth compose and Studio state are under
`/srv/llm/services/unsloth/`. Both use `/srv/llm/hf-cache/` on the SSD.
