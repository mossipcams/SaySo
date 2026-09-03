#!/usr/bin/env python3
"""Verify GGUF model returns structured tool_calls via llama-server."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.llamacpp import parse_chat_completion  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, default=ROOT / "fixtures" / "sayso_payload.json")
    parser.add_argument("--server-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    print(f"Payload messages: {len(payload.get('messages', []))}")
    print(f"Payload tools: {len(payload.get('tools', []))}")

    if args.dry_run:
        print("Dry run: would POST to", args.server_url + "/chat/completions")
        return 0

    try:
        import requests
    except ImportError:
        print("Install requests to run live verification", file=sys.stderr)
        return 1

    response = requests.post(
        f"{args.server_url}/chat/completions",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    parsed = parse_chat_completion(response.json())
    print(json.dumps(parsed, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
