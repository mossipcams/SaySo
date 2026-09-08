#!/usr/bin/env python3
"""Generate v3 quality eval JSONL (gold + shadow) for the 40k synthetic contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_synthetic_dataset import write_jsonl  # noqa: E402
from evals.v3_quality import (  # noqa: E402
    DEFAULT_SHADOW_COUNT,
    build_gold_examples,
    build_shadow_examples,
    excluded_train_prompts,
    gold_user_prompts,
    shadow_user_prompts,
    v3_quality_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold-out",
        type=Path,
        default=ROOT / "datasets" / "sayso_quality_eval_v3_gold.jsonl",
    )
    parser.add_argument(
        "--shadow-out",
        type=Path,
        default=ROOT / "datasets" / "sayso_quality_eval_v3_shadow.jsonl",
    )
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--shadow-count", type=int, default=DEFAULT_SHADOW_COUNT)
    args = parser.parse_args()

    gold_rows = build_gold_examples()
    shadow_rows = build_shadow_examples(seed=args.seed, count=args.shadow_count)

    args.gold_out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.gold_out, gold_rows)
    write_jsonl(args.shadow_out, shadow_rows)

    gold_prompts = {_normalized_prompt(row) for row in gold_rows}
    shadow_prompts = {_normalized_prompt(row) for row in shadow_rows}
    overlap = gold_prompts.intersection(shadow_prompts)

    report = {
        "v3_quality": {
            **v3_quality_summary(),
            "gold_path": str(args.gold_out),
            "shadow_path": str(args.shadow_out),
            "gold_rows": len(gold_rows),
            "shadow_rows": len(shadow_rows),
            "gold_shadow_prompt_overlap": len(overlap),
            "excluded_train_prompt_count": len(excluded_train_prompts()),
        }
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _normalized_prompt(row: dict) -> str:
    from evals.v3_quality import _normalized

    user = next(message for message in row["messages"] if message.get("role") == "user")
    return _normalized(str(user.get("content", "")))


if __name__ == "__main__":
    raise SystemExit(main())
