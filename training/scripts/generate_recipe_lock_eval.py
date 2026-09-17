#!/usr/bin/env python3
"""Generate the locked recipe quality eval JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from generators.pipeline import write_jsonl  # noqa: E402
from evals.recipe_lock import (  # noqa: E402
    build_quality_eval_examples,
    recipe_lock_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-out",
        type=Path,
        default=ROOT / "datasets" / "sayso_quality_eval_recipe_lock.jsonl",
    )
    args = parser.parse_args()

    eval_rows = build_quality_eval_examples()
    args.eval_out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.eval_out, eval_rows)

    report: dict[str, object] = {
        "quality_eval": {
            **recipe_lock_summary(),
            "path": str(args.eval_out),
            "rows": len(eval_rows),
        }
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
