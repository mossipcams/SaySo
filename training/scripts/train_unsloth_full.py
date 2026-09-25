#!/usr/bin/env python3
"""Full SFT of LFM2.5-230M-Base on SaySo's rendered v5 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

from unsloth import FastModel  # import before transformers
import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

HOST = "/workspace/host"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=f"{HOST}/lfm/models/LFM2.5-230M-Base")
    parser.add_argument("--data", default=f"{HOST}/datasets/sayso_full_sft_v5_20260922_rendered.jsonl")
    parser.add_argument("--out")
    parser.add_argument("--cutoff", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--accum", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--canary", action="store_true", help="Run 250 production-config optimizer steps")
    parser.add_argument("--promotion-record", help="Required promotion metadata for full training")
    args = parser.parse_args()
    if args.canary:
        args.max_steps = 250
    elif not args.promotion_record:
        parser.error("full training requires --promotion-record from ./sayso promote-dataset")
    else:
        with open(args.promotion_record, encoding="utf-8") as handle:
            promotion = json.load(handle)
        digest = hashlib.sha256()
        with open(args.data, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if (not promotion.get("promoted") or not promotion.get("validation_passed")
                or not promotion.get("canary_passed")
                or digest.hexdigest() != promotion.get("rendered_sha256")):
            parser.error("full training blocked: data does not match the explicitly promoted dataset")
    args.out = args.out or (
        f"{HOST}/lfm/runs/sayso-lfm-v5-canary" if args.canary else f"{HOST}/lfm/runs/sayso-lfm-v5-full"
    )

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
        prompt = tokenizer(row["instruction"], add_special_tokens=False)["input_ids"]
        completion = tokenizer(row["output"], add_special_tokens=False)["input_ids"]
        return {
            "input_ids": prompt + completion,
            "attention_mask": [1] * (len(prompt) + len(completion)),
            "labels": [-100] * len(prompt) + completion,
            "length": len(prompt) + len(completion),
            "supervised": len(completion),
        }

    def rows(path, stamp):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                yield {"instruction": row["instruction"], "output": row["output"]}

    stat = os.stat(args.data)
    dataset = Dataset.from_generator(
        rows, gen_kwargs={"path": args.data, "stamp": f"{stat.st_size}:{stat.st_mtime_ns}"}
    )
    dataset = dataset.map(encode, remove_columns=dataset.column_names, num_proc=6)
    lengths = dataset["length"]
    too_long = sum(length > args.cutoff for length in lengths)
    empty = sum(count == 0 for count in dataset["supervised"])
    print(
        f"rows={len(dataset)} max_tokens={max(lengths)} "
        f"supervised_tokens={sum(dataset['supervised'])} over_cutoff={too_long} empty_completion={empty}"
    )
    if too_long or empty:
        sys.exit("preflight failed")
    dataset = dataset.remove_columns(["length", "supervised"])
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
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
