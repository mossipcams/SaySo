"""CLI entry for synthetic dataset generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from generators.config import DEFAULT_TRAIN_COUNT, GeneratorConfig
from generators.pipeline import run_generation, write_jsonl, write_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate SaySo synthetic v3 training dataset")
    parser.add_argument("--count", type=int, default=DEFAULT_TRAIN_COUNT)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=TRAINING_ROOT / "datasets" / "synthetic_v3_train.jsonl")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--stt-rate", type=float, default=0.15)
    parser.add_argument("--paraphrase", action="store_true", default=False)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--exclude-prompts", type=Path, default=None)
    args = parser.parse_args(argv)

    config = GeneratorConfig(
        count=args.count,
        seed=args.seed,
        split=args.split,
        output_path=args.output,
        manifest_path=args.manifest or args.output.with_suffix(".manifest.json"),
        stt_noise_rate=args.stt_rate,
        paraphrase_enabled=args.paraphrase,
        token_budget=args.token_budget,
        exclude_prompts_path=args.exclude_prompts,
    )
    result = run_generation(config)
    write_jsonl(config.output_path, result["rows"])
    write_manifest(config.manifest_path, result["stats"])
    print(json.dumps(result["stats"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
