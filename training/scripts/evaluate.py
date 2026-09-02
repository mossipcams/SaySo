#!/usr/bin/env python3
"""Evaluate a checkpoint against held-out and adversarial sets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.harness import evaluate_checkpoint, load_eval_jsonl, write_results  # noqa: E402


def _stub_infer(example: dict) -> dict:
    """Placeholder infer_fn for offline harness wiring tests."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Stub response.",
                }
            }
        ]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_set", type=Path)
    parser.add_argument("--checkpoint-id", default="base")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "eval_results.json")
    args = parser.parse_args()

    examples = load_eval_jsonl(args.eval_set)
    summary = evaluate_checkpoint(examples, _stub_infer, checkpoint_id=args.checkpoint_id)
    write_results(args.out, summary)
    print(json.dumps(summary.rates(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
