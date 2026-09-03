#!/usr/bin/env python3
"""CLI wrapper around the Home-LLM pile SaySo generator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generators.home_llm_piles import SAMPLE_FACTORS, SMALL_FACTORS, generate_pile_examples  # noqa: E402
from generators.piles import GenerationStats  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=ROOT / "datasets" / "sayso_piles.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--count", type=int, default=None)
    args = parser.parse_args()

    factors = SAMPLE_FACTORS if args.sample else SMALL_FACTORS
    stats = GenerationStats()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for example in generate_pile_examples(seed=args.seed, factors=factors, stats=stats):
            if args.count is not None and written >= args.count:
                break
            handle.write(json.dumps(example, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1

    print(f"Wrote {written} examples to {args.output}")
    if stats.dropped:
        print("Dropped:", stats.dropped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
