#!/usr/bin/env python3
"""Pin Home-LLM upstream commit recorded in upstream.lock.json."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "upstream.lock.json"


def main() -> int:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    commit = lock["commit"]
    repo = lock["repository"]
    url = f"{repo}.git"
    dest = ROOT / ".upstream" / "home-llm"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if (dest / ".git").exists():
        subprocess.run(["git", "fetch", "origin"], cwd=dest, check=True)
        subprocess.run(["git", "checkout", commit], cwd=dest, check=True)
    else:
        subprocess.run(["git", "clone", url, str(dest)], check=True)
        subprocess.run(["git", "checkout", commit], cwd=dest, check=True)

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dest, text=True).strip()
    if head != commit:
        print(f"ERROR: checked out {head}, expected {commit}", file=sys.stderr)
        return 1
    print(f"Pinned {repo} at {commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
