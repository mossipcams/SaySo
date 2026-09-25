#!/usr/bin/env python3
"""Score a checkpoint on held-out rendered rows (same distribution as training), on CPU.

Plan: docs/PLAN_V6_DATASET_ARCHITECTURE.md, Step 1. Each row's first supervised
turn is replayed exactly as training rendered it. The output is compared three
ways: exact text, decision class (call / ask / refuse / unsupported / absent /
ignore / other), and parsed tool calls (production parser, order-insensitive).
High here but low on the eval means a distribution gap; low here means the
model is not learning the task.

  docker exec -e HIP_VISIBLE_DEVICES= -e CUDA_VISIBLE_DEVICES= -w /workspace/host/sayso unsloth \\
    python training/scripts/eval_rendered_cpu.py CHECKPOINT VIEWS.jsonl --out RESULT.json --limit 300
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sayso_contract import completion

END = "<|im_end|>"


def decision(text: str) -> str:
    if "<|tool_call_start|>" in text or text.lstrip().startswith("["):
        return "call"
    t = text.lower()
    for label, marker in (("ignore", "didn't catch"), ("refuse", "can't do that"), ("refuse", "can't help"),
                          ("unsupported", "does not support"), ("absent", "has no"), ("absent", "there is no")):
        if marker in t:
            return label
    return "ask" if "?" in t else "other"


def calls(text: str) -> list[str] | None:
    try:
        _, parsed = completion.extract_tool_calls(text)
    except Exception:
        return None
    return sorted(json.dumps([c.name, c.arguments], sort_keys=True) for c in parsed)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("views", type=Path, help="rendered view JSONL (export_llamafactory_view.py output)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    # First supervised turn per source row, then a seeded sample.
    first: dict[int, dict] = {}
    for line in args.views.open(encoding="utf-8"):
        view = json.loads(line)
        first.setdefault(view["metadata"]["source_row_index"], view)
    views = list(first.values())
    random.Random(args.seed).shuffle(views)
    views = views[: args.limit]

    torch.set_num_threads(args.threads)
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, dtype=torch.float32).eval()

    by_family: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    rows = []
    for i, view in enumerate(views):
        ids = tok(view["instruction"], add_special_tokens=False, return_tensors="pt")
        with torch.inference_mode():
            out = model.generate(**ids, max_new_tokens=160, do_sample=False)
        got = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=False).split(END)[0]
        want = view["output"].split(END)[0]
        score = {
            "exact": got.strip() == want.strip(),
            "decision": decision(got) == decision(want),
            "calls": decision(want) != "call" or calls(got) == calls(want),
        }
        family = view["metadata"].get("family") or "?"
        for key, ok in score.items():
            by_family[family][key] += ok
            by_family["ALL"][key] += ok
        by_family[family]["n"] += 1
        by_family["ALL"]["n"] += 1
        rows.append({"family": family, "want": want, "got": got, **score})
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(views)} exact={by_family['ALL']['exact']}", flush=True)

    summary = {
        family: {k: round(c[k] / c["n"], 3) for k in ("exact", "decision", "calls")} | {"n": c["n"]}
        for family, c in sorted(by_family.items())
    }
    args.out.write_text(json.dumps({"checkpoint": args.checkpoint, "summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary["ALL"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
