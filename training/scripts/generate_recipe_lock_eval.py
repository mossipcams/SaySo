#!/usr/bin/env python3
"""Generate recipe-lock quality eval JSONL and deterministic 10k train JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_synthetic_dataset import (  # noqa: E402
    DEFAULT_TRAIN_COUNT,
    build_deterministic_train_examples,
    render_for_trl,
    write_jsonl,
)
from evals.recipe_lock import (  # noqa: E402
    build_quality_eval_examples,
    quality_eval_user_prompts,
    recipe_lock_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-out",
        type=Path,
        default=ROOT / "datasets" / "sayso_quality_eval_recipe_lock.jsonl",
    )
    parser.add_argument(
        "--train-out",
        type=Path,
        default=ROOT / "datasets" / "sayso_train_first_10000.jsonl",
    )
    parser.add_argument(
        "--train-render-out",
        type=Path,
        default=ROOT / "datasets" / "sayso_train_first_10000_render.jsonl",
    )
    parser.add_argument("--count", type=int, default=DEFAULT_TRAIN_COUNT)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    eval_rows = build_quality_eval_examples()
    args.eval_out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.eval_out, eval_rows)

    report: dict[str, object] = {
        "quality_eval": {
            **recipe_lock_summary(),
            "path": str(args.eval_out),
            "rows": len(eval_rows),
        }
    }

    if not args.skip_train:
        excluded = quality_eval_user_prompts()
        train_rows = build_deterministic_train_examples(
            args.count,
            seed=args.seed,
            excluded_utterances=excluded,
        )
        rendered_rows = [render_for_trl(row) for row in train_rows]
        write_jsonl(args.train_out, train_rows)
        write_jsonl(args.train_render_out, rendered_rows)
        train_prompts = {
            next(message["content"] for message in row["messages"] if message.get("role") == "user")
            for row in train_rows
        }
        overlap = {prompt for prompt in train_prompts if prompt in excluded}
        report["train"] = {
            "path": str(args.train_out),
            "render_path": str(args.train_render_out),
            "rows": len(train_rows),
            "seed": args.seed,
            "excluded_prompt_count": len(excluded),
            "overlap_count": len(overlap),
        }

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
