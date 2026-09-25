#!/usr/bin/env python3
"""Evaluate a checkpoint on CPU through SaySo's production eval runner/scorer.

Runs inside the unsloth container, without touching the GPU, even while a
training run holds the lock:
  docker exec -e HIP_VISIBLE_DEVICES= -e CUDA_VISIBLE_DEVICES= -w /workspace/host/sayso unsloth \
    python training/scripts/eval_checkpoint_cpu.py CHECKPOINT --suite smoke --out DIR

The repo mount is read-only, so results go to --out instead of evals/results/.
Generation is greedy and capped like production (temperature 0, 160 tokens); the
raw text (native <|tool_call_start|> markers) goes through the production parser.
"""

from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--suite", choices=("smoke", "promotion"), default="smoke")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT / "training/configs/training_baseline.json")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, dtype=torch.float32).eval()
    end = "<|im_end|>"

    def predict(messages, tools):
        encoded = tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=PRODUCTION_MAX_OUTPUT_TOKENS, do_sample=False)
        text = tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=False).split(end)[0]
        return {"choices": [{"message": {"role": "assistant", "content": text}}]}

    runner.RESULTS_DIR = args.out
    adapter = InMemoryAdapter(predict)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))["eval"]
    failures: list[str] = []
    for label, cases in [(args.suite, select_cases(suite=args.suite)), *[(tag, select_cases(tag=tag)) for tag in args.tag]]:
        results = runner.evaluate(cases, adapter)
        summary = summarize(results)
        dest = runner.write_run(
            cases=cases,
            results=results,
            adapter=adapter,
            suite=label if label in {"smoke", "promotion"} else None,
            extra_metadata={"checkpoint": args.checkpoint, "gate_label": label},
        )
        print(f"{label}: {summary['passed']}/{summary['total']} passed -> {dest}")
        from evals.cases import load_gates
        from evals.scorer import check_gates

        gate_suite = args.suite if label == args.suite and args.suite in {"smoke", "promotion"} else None
        if gate_suite:
            failures.extend(
                f"{label} {failure}"
                for failure in check_gates(summary, load_gates(gate_suite), expected_count=len(cases))
            )
        if label != "smoke":
            continue
        accepted = baseline["smoke"]
        tolerance = baseline["max_drop"]
        rates = {"overall_pass_rate": summary["overall_pass_rate"], **summary["category_pass_rate"]}
        for metric, old_rate in {"overall_pass_rate": accepted["overall_pass_rate"], **accepted["category_pass_rate"]}.items():
            actual = rates.get(metric)
            if actual is None or actual < old_rate - tolerance:
                failures.append(f"smoke {metric}: {actual} < baseline {old_rate:.4f} - {tolerance:.2f}")
        tool_validity = 1 - summary["malformed_output_rate"] - summary["unknown_tool_rate"] - summary["invalid_argument_rate"]
        if tool_validity < accepted["tool_validity"] - tolerance:
            failures.append(f"smoke tool_validity: {tool_validity:.4f} < baseline {accepted['tool_validity']:.4f} - {tolerance:.2f}")
        for flag, old_count in accepted.get("flags", {}).items():
            count = summary["flags"].get(flag, 0)
            if count > old_count:
                failures.append(f"smoke {flag}: {count} > baseline {old_count}")

    for failure in failures:
        print(f"EVAL FAIL: {failure}", file=sys.stderr)
    print(f"EVAL: {'FAIL' if failures else 'PASS'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
