#!/usr/bin/env python3
"""Run LFM2.5-230M Axolotl training (SaySo OpenAI tool envelope)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    argv = [sys.executable, str(ROOT / "scripts" / "train.py"), "--backend", "lfm", *sys.argv[1:]]
    return subprocess.call(argv)


if __name__ == "__main__":
    raise SystemExit(main())
