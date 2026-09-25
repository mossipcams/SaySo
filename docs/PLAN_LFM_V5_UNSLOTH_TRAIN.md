# v5 full SFT with Unsloth on the llm VM

Date: 2026-09-24. Scope: one launcher script plus the host runs. No generator,
eval, or runtime changes.

## Contract (unchanged from PLAN_LFM_HOST_V5_SETUP.md)

Full SFT from `LFM2.5-230M-Base`, trained on the rendered view
(`instruction`/`output`; the instruction already carries `<|startoftext|>` and
the Base chat template). Loss on `output` tokens only. 8192-token cutoff with
**zero truncation**: a longer row fails preflight, it is not clipped. lr `2e-5`
cosine, batch 1 / accum 32, bf16, 2 epochs.

## Files

- `training/scripts/train_unsloth_full.py`: loads Base through
  `unsloth.FastModel(full_finetuning=True)`, tokenizes prompt and completion
  separately (prompt labels `-100`), checks the length gate, and trains with the
  HF `Trainer`. `--max-steps` for the smoke.

## Sequence

1. v5 generation + export finish on CPU (`/srv/llm/lfm/runs/v5-gen-20260924/gen.log`),
   artifacts copied to `/srv/llm/data/lfm/datasets/`.
2. Smoke: `gpu train lfm --serve-after python .../train_unsloth_full.py --max-steps 20`
   on the v5 rendered view. It needs finite loss, a HIP device, and the longest row
   under the VRAM limit (the script sorts the longest rows first under `--max-steps`).
3. Full run: the same command without `--max-steps`, through `gpu train lfm --serve-after`,
   detached. Output `/srv/llm/lfm/runs/sayso-lfm-v5-full/`.

## Verification

- Preflight prints row count, max tokens, supervised token count; exits non-zero on
  any row > 8192 tokens or with an empty completion.
- Smoke loss finite and falling; `gpu status` shows `train:lfm` during the run.
