#!/usr/bin/env python3
"""Run Axolotl training with GPU-aware config selection."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "configs"))

from detect_gpu import detect_gpu  # noqa: E402

BACKENDS = {
    "lfm": "lfm25-230m",
    "functiongemma": "functiongemma-270m",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=sorted(BACKENDS),
        default="lfm",
        help="lfm=LFM2.5-230M SaySo envelope (default), functiongemma=FunctionGemma path",
    )
    parser.add_argument(
        "--config",
        choices=["smoke", "prod"],
        default="smoke",
        help="smoke=small fixture dataset, prod=full dataset",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    profile = detect_gpu()
    prefix = BACKENDS[args.backend]
    config_name = f"{prefix}-{args.config}.yml"
    config_path = ROOT / "configs" / config_name

    if not config_path.exists():
        print(f"Missing config: {config_path}", file=sys.stderr)
        return 1

    print(f"Backend: {args.backend}")
    print(f"GPU profile: {profile.name} ({profile.notes})")
    print(f"Config: {config_path}")

    if profile.name == "none":
        print("SKIP: No GPU detected. Config and scripts are valid; run when GPU available.")
        return 0

    cmd = ["axolotl", "train", str(config_path)]
    if args.dry_run:
        print("Dry run:", " ".join(cmd))
        return 0

    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
