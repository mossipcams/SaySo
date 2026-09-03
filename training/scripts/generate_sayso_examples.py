#!/usr/bin/env python3
"""Generate SaySo-specific English training examples (Home-LLM V2 JSONL shape)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generators.home_llm_piles import (  # noqa: E402
    SAMPLE_FACTORS,
    SMALL_FACTORS,
    generate_pile_examples,
)
from generators.piles import GenerationStats  # noqa: E402


def generate_examples(
    count: int | None = None,
    *,
    seed: int = 42,
    small: bool = False,
    sample: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield Home-LLM V2 shaped examples from English piles (v1 tools only)."""
    factors = SAMPLE_FACTORS if sample else SMALL_FACTORS
    stats = GenerationStats()
    produced = 0
    for example in generate_pile_examples(seed=seed, factors=factors, stats=stats):
        yield example
        produced += 1
        if count is not None and produced >= count:
            break


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=ROOT / "datasets" / "sayso_generated.jsonl",
    )
    parser.add_argument("--count", type=int, default=None, help="Optional cap (default: full pile run)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--small",
        action="store_true",
        help="Home-LLM-small factors (static=1, template=10, status=8, refusal=3, failure=1)",
    )
    parser.add_argument("--sample", action="store_true", help="Minimal factors (all 1) for smoke tests")
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
