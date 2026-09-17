#!/usr/bin/env python3
"""Generate a held-out dataset with the canonical deterministic generator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generators.config import GeneratorConfig  # noqa: E402
from generators.pipeline import run_generation, write_jsonl  # noqa: E402

DEFAULT_COUNT = 2_500


def build_balanced_test_set(
    *,
    count: int = DEFAULT_COUNT,
    seed: int = 1042,
) -> list[dict]:
    """Build deterministic held-out rows from the production generator."""
    if count <= 0:
        raise ValueError("count must be positive")
    result = run_generation(
        GeneratorConfig(count=count, seed=seed, split="test", synthetic_only=True)
    )
    rows = result["rows"]
    for row in rows:
        metadata = row.setdefault("metadata", {})
        metadata.update(
            evaluation_category=metadata.get("category", "ordinary"),
            evaluation_subcategory=metadata.get("subcategory", ""),
            evaluation_seed=seed,
            held_out=True,
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=1042)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "datasets" / "sayso_test_balanced.jsonl",
    )
    args = parser.parse_args()
    examples = build_balanced_test_set(count=args.count, seed=args.seed)
    write_jsonl(args.out, examples)
    print(f"Wrote {len(examples)} examples to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
