#!/usr/bin/env python3
"""Score a training checkpoint on CPU through evals.runner, without touching the GPU.

Runs inside the unsloth container while a GPU training run holds the lock:
  docker exec -e HIP_VISIBLE_DEVICES= -e CUDA_VISIBLE_DEVICES= -w /workspace/host/sayso unsloth \
    python training/scripts/eval_checkpoint_cpu.py CHECKPOINT --suite smoke --out DIR

The repo mount is read-only, so results go to --out instead of evals/results/.
Generation is greedy and capped like production (temperature 0, 160 tokens); the
raw text (native <|tool_call_start|> markers) goes through the production parser.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import evals.runner as runner
from evals.adapters import InMemoryAdapter
from evals.cases import select_cases
from evals.outcomes import PRODUCTION_MAX_OUTPUT_TOKENS
from evals.scorer import summarize


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--suite", choices=("smoke", "promotion"), default="smoke")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    torch.set_num_threads(args.threads)  # leave cores for the training run's host side
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, dtype=torch.float32).eval()
    end = "<|im_end|>"

    def predict(messages, tools):
        ids = tok.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        with torch.inference_mode():
            out = model.generate(**ids, max_new_tokens=PRODUCTION_MAX_OUTPUT_TOKENS, do_sample=False)
        text = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=False)
        text = text.split(end)[0]
        return {"choices": [{"message": {"role": "assistant", "content": text}}]}

    runner.RESULTS_DIR = args.out
    cases = select_cases(suite=args.suite)
    adapter = InMemoryAdapter(predict)
    results = runner.evaluate(cases, adapter)
    dest = runner.write_run(
        cases=cases, results=results, adapter=adapter, suite=args.suite, extra_metadata={"checkpoint": args.checkpoint}
    )
    summary = summarize(results)
    print(f"{summary['passed']}/{summary['total']} passed -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
