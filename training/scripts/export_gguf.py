#!/usr/bin/env python3
"""Export trained checkpoint to GGUF via llama.cpp convert scripts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--quantize", nargs="*", default=["Q8_0", "Q4_K_M"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    f16_out = args.out_dir / "model-f16.gguf"

    convert_cmd = [
        "python",
        "convert_hf_to_gguf.py",
        str(args.checkpoint),
        "--outfile",
        str(f16_out),
        "--outtype",
        "f16",
    ]
    print("Step 1:", " ".join(convert_cmd))
    if not args.dry_run:
        print("NOTE: Requires llama.cpp convert_hf_to_gguf.py on PATH", file=sys.stderr)
        return 0

    for quant in args.quantize:
        q_out = args.out_dir / f"model-{quant.lower()}.gguf"
        quant_cmd = ["llama-quantize", str(f16_out), str(q_out), quant]
        print("Step 2:", " ".join(quant_cmd))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
