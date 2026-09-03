#!/usr/bin/env python3
"""Detect GPU capabilities for SaySo training configs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "configs"))

from detect_gpu import detect_gpu  # noqa: E402


def main() -> int:
    profile = detect_gpu()
    print(f"gpu_name={profile.name}")
    print(f"fp16={profile.fp16}")
    print(f"bf16={profile.bf16}")
    print(f"flash_attention={profile.flash_attention}")
    print(f"notes={profile.notes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
