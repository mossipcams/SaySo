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
    parser.add_argument(
        "--real-home",
        type=Path,
        default=None,
        help="Home JSON from scripts/fetch_ha_home.py to mix into generation",
    )
    parser.add_argument(
        "--real-home-rate",
        type=float,
        default=0.0,
        help="Fraction of rows generated over the real home (needs --real-home)",
    )
    parser.add_argument(
        "--real-home-entity-cap",
        type=int,
        default=0,
        help="Max rows one real entity may be the target of (0 derives it)",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Explicitly disable real-home mixing, overriding --real-home",
    )
    parser.add_argument(
        "--allow-stale-home",
        action="store_true",
        help="Mix a home snapshot that was not filtered by Home Assistant's Assist "
             "exposure list (see scripts/fetch_ha_home.py --allow-unexposed)",
    )
    parser.add_argument("--negative-rate", type=float, default=None)
    parser.add_argument("--grounding-rate", type=float, default=None)
    parser.add_argument("--max-absence-rate", type=float, default=None)
    args = parser.parse_args(argv)

    if args.real_home and not args.synthetic_only:
        from generators.real_home import TESTABLE_EXPOSURE_SOURCES, require_exposure_source

        if not args.allow_stale_home:
            require_exposure_source(args.real_home, allowed=TESTABLE_EXPOSURE_SOURCES)

    overrides = {
        key: value
        for key, value in (
            ("negative_rate", args.negative_rate),
            ("grounding_rate", args.grounding_rate),
            ("max_absence_rate", args.max_absence_rate),
        )
        if value is not None
    }
    if args.real_home_rate and not args.real_home and not args.synthetic_only:
        parser.error("--real-home-rate needs --real-home")
    config = GeneratorConfig(
        synthetic_only=args.synthetic_only,
        **overrides,
        count=args.count,
        seed=args.seed,
        split=args.split,
        output_path=args.output,
        manifest_path=args.manifest or args.output.with_suffix(".manifest.json"),
        stt_noise_rate=args.stt_rate,
        paraphrase_enabled=args.paraphrase,
        token_budget=args.token_budget,
        exclude_prompts_path=args.exclude_prompts,
        real_home_path=args.real_home,
        real_home_rate=args.real_home_rate,
        real_home_entity_cap=args.real_home_entity_cap,
    )
    result = run_generation(config)
    write_jsonl(config.output_path, result["rows"])
    write_manifest(config.manifest_path, result["stats"])
    print(json.dumps(result["stats"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
