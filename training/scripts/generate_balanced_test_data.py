#!/usr/bin/env python3
"""Generate a held-out synthetic test set via the canonical generator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from generators.config import GeneratorConfig
from generators.pipeline import run_build

DEFAULT_COUNT = 2_500


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate held-out balanced test data")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "datasets" / "sayso_test_balanced.jsonl",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "generation" / "production.yaml",
    )
    args = parser.parse_args()

    config = GeneratorConfig.from_yaml(args.config, repo_root=REPO)
    config.count = args.count
    config.seed = args.seed
    config.split = "test"
    config.output_path = args.output
    config.manifest_path = args.output.with_suffix(".manifest.json")

    run_build(config)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
