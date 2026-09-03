#!/usr/bin/env python3
"""Adapt Home-LLM V2 JSONL to SaySo training JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters.home_llm_v2 import convert_jsonl_stream  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Home-LLM V2 JSONL input")
    parser.add_argument("output", type=Path, help="SaySo training JSONL output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--view",
        choices=("axolotl", "sayso"),
        default="axolotl",
        help="axolotl: dict arguments for FunctionGemma template; sayso: JSON strings",
    )
    parser.add_argument("--stats", type=Path, help="Write rejection stats JSON")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as handle, args.output.open(
        "w", encoding="utf-8"
    ) as out:
        stats, written = convert_jsonl_stream(
            handle, seed=args.seed, output=out, view=args.view
        )

    print(f"Wrote {written} examples to {args.output} (view={args.view})")
    print(f"Rejections: {json.dumps(stats.counts, sort_keys=True)}")

    if args.stats:
        args.stats.write_text(json.dumps(stats.counts, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
