"""CLI for converting Home-LLM synthetic JSONL to SaySo SFT JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from train.generator import convert_home_llm_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert Home-LLM synthetic JSONL into SaySo train JSONL.",
    )
    parser.add_argument("input", type=Path, help="Home-LLM JSONL input path")
    parser.add_argument("output", type=Path, help="SaySo SFT JSONL output path")
    args = parser.parse_args(argv)

    stats = convert_home_llm_jsonl(args.input, args.output)
    print(
        f"converted {stats.kept_rows}/{stats.input_rows} rows "
        f"(dropped {stats.dropped_rows}) -> {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
