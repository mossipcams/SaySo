#!/usr/bin/env python3
"""Generate SaySo LFM training dataset from the pile-based SaySo example generator."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default="english", help="Language tag for output filenames")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use Home-LLM-small pile multipliers (thousands of v1 examples)",
    )
    parser.add_argument("--sample", action="store_true", help="Minimal pile multipliers for smoke tests")
    parser.add_argument(
        "--sayso-count",
        type=int,
        default=None,
        help="Optional cap on generated examples (default: full pile run)",
    )
    parser.add_argument(
        "--view",
        choices=("lfm", "sayso", "axolotl"),
        default="lfm",
        help="Adapter output view (default: lfm for LFM / llama.cpp training)",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "datasets")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = args.out_dir / f"sayso_raw_{args.language}.jsonl"
    adapted = args.out_dir / f"sayso_adapted_{args.language}.jsonl"

    gen_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "generate_sayso_examples.py"),
        str(raw),
        "--seed",
        str(args.seed),
    ]
    if args.small:
        gen_cmd.append("--small")
    if args.sample:
        gen_cmd.append("--sample")
    if args.sayso_count is not None:
        gen_cmd.extend(["--count", str(args.sayso_count)])

    _run(gen_cmd)

    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "adapt_dataset.py"),
            str(raw),
            str(adapted),
            "--seed",
            str(args.seed),
            "--view",
            args.view,
        ]
    )

    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "split_dataset.py"),
            str(adapted),
            "--out-dir",
            str(args.out_dir),
            "--seed",
            str(args.seed),
        ]
    )

    print(f"Pipeline complete: {adapted} -> splits in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
