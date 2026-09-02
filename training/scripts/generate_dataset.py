#!/usr/bin/env python3
"""Generate SaySo training dataset from pinned Home-LLM upstream and local generator."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default="english")
    parser.add_argument("--size", default="small", choices=["small", "medium", "large", "xl"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sayso-count", type=int, default=200, help="Local SaySo examples")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "datasets")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mixed_raw = args.out_dir / f"sayso_mixed_raw_{args.language}.jsonl"
    adapted = args.out_dir / f"sayso_adapted_{args.language}.jsonl"

    with tempfile.TemporaryDirectory(prefix="sayso-gen-") as tmp:
        tmp_dir = Path(tmp)
        parts: list[Path] = []

        sayso_gen = tmp_dir / "sayso_generated.jsonl"
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "generate_sayso_examples.py"),
                str(sayso_gen),
                "--count",
                str(args.sayso_count),
                "--seed",
                str(args.seed),
            ]
        )
        parts.append(sayso_gen)

        upstream = ROOT / ".upstream" / "home-llm" / "data"
        if upstream.exists():
            cmd = [
                sys.executable,
                str(upstream / "generate_data.py"),
                "--train",
                f"--{args.size}",
                "--language",
                args.language,
                "--seed",
                str(args.seed),
            ]
            try:
                _run(cmd, cwd=upstream)
                raw_output = upstream / "output" / f"home_assistant_train_{args.language}.jsonl"
                if not raw_output.exists():
                    candidates = list((upstream / "output").glob("*.jsonl"))
                    raw_output = candidates[0] if candidates else None
                if raw_output and raw_output.exists():
                    parts.append(raw_output)
            except subprocess.CalledProcessError:
                print("Home-LLM generate failed; continuing with SaySo-only mix", file=sys.stderr)
        else:
            print("Upstream not pinned; using SaySo generator only", file=sys.stderr)

        with mixed_raw.open("w", encoding="utf-8") as out:
            for part in parts:
                out.write(part.read_text(encoding="utf-8"))

    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "adapt_dataset.py"),
            str(mixed_raw),
            str(adapted),
            "--seed",
            str(args.seed),
            "--view",
            "axolotl",
        ]
    )

    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "split_dataset.py"),
            str(adapted),
            "--out-dir",
            str(args.out_dir),
            "--seed",
            str(args.seed),
        ]
    )

    print(f"Pipeline complete: {adapted} -> splits in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
