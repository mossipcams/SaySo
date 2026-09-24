#!/usr/bin/env python3
"""Full SFT of LFM2.5-230M-Base on the rendered v5 view, inside the unsloth container.

Plan: docs/PLAN_LFM_V5_UNSLOTH_TRAIN.md. Launch through the GPU lock:
  /srv/llm/bin/gpu train lfm --serve-after python /workspace/host/sayso/training/scripts/train_unsloth_full.py ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from unsloth import FastModel  # must import before transformers

import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

HOST = "/workspace/host"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=f"{HOST}/lfm/models/LFM2.5-230M-Base")
    p.add_argument("--data", default=f"{HOST}/datasets/sayso_full_sft_v5_20260922_rendered.jsonl")
    p.add_argument("--out", default=f"{HOST}/lfm/runs/sayso-lfm-v5-full")
    p.add_argument("--cutoff", type=int, default=8192)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--accum", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=-1, help="smoke: train on the longest rows only")
    args = p.parse_args()

    print(f"torch {torch.__version__} hip={torch.version.hip} device={torch.cuda.get_device_name(0)}")
    if not torch.version.hip or not torch.cuda.is_available():
        sys.exit("no HIP device")

    model, tokenizer = FastModel.from_pretrained(
        args.model,
        max_seq_length=args.cutoff,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        full_finetuning=True,
    )

    def encode(row):
        # The instruction is the full Base-template render, <|startoftext|> included.
        prompt = tokenizer(row["instruction"], add_special_tokens=False)["input_ids"]
        completion = tokenizer(row["output"], add_special_tokens=False)["input_ids"]
        return {
            "input_ids": prompt + completion,
            "attention_mask": [1] * (len(prompt) + len(completion)),
            "labels": [-100] * len(prompt) + completion,
            "length": len(prompt) + len(completion),
            "supervised": len(completion),
        }

    # Only the two training columns: per-row metadata structs vary, and arrow
    # schema inference over them fails.
    def rows(path, stamp):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                yield {"instruction": row["instruction"], "output": row["output"]}

    # The HF cache keys on the generator and its kwargs, not the file: without the
    # size/mtime stamp a regenerated corpus at the same path reuses the old rows.
    stat = os.stat(args.data)
    ds = Dataset.from_generator(rows, gen_kwargs={"path": args.data, "stamp": f"{stat.st_size}:{stat.st_mtime_ns}"})
    ds = ds.map(encode, remove_columns=ds.column_names, num_proc=6)

    # Zero truncation: a row over the cutoff fails the run, it is never clipped.
    lengths = ds["length"]
    too_long = sum(n > args.cutoff for n in lengths)
    empty = sum(n == 0 for n in ds["supervised"])
    print(f"rows={len(ds)} max_tokens={max(lengths)} supervised_tokens={sum(ds['supervised'])} "
          f"over_cutoff={too_long} empty_completion={empty}")
    if too_long or empty:
        sys.exit("preflight failed")

    if args.max_steps > 0:
        ds = ds.sort("length", reverse=True).select(range(min(len(ds), args.max_steps * args.accum)))
    ds = ds.remove_columns(["length", "supervised"])

    trainer = Trainer(
        model=model,
        train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
        args=TrainingArguments(
            output_dir=args.out,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.accum,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            weight_decay=0.0,
            bf16=True,
            optim="adamw_torch",
            logging_steps=5,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=3,
            report_to="none",
            seed=20260922,
        ),
    )
    trainer.train()
    trainer.save_model(f"{args.out}/final")
    tokenizer.save_pretrained(f"{args.out}/final")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
