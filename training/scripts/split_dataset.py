#!/usr/bin/env python3
"""Leakage-resistant 80/10/10 split by template/phrasing/seed families."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _family_key(record: dict, seed: int) -> str:
    meta = record.get("metadata") or {}
    template = meta.get("template_family") or meta.get("template") or ""
    phrasing = meta.get("phrasing_family") or meta.get("phrasing") or ""
    gen_seed = str(meta.get("seed", seed))
    if not template and not phrasing:
        messages = record.get("messages") or []
        user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
        template = hashlib.sha256(str(user).encode()).hexdigest()[:8]
    return f"{template}|{phrasing}|{gen_seed}"


def assign_split(family: str, *, train_ratio: float = 0.8, val_ratio: float = 0.1) -> str:
    """Deterministically assign family to train/val/test."""
    digest = int(hashlib.sha256(family.encode()).hexdigest(), 16) % 1000
    train_cut = int(train_ratio * 1000)
    val_cut = train_cut + int(val_ratio * 1000)
    if digest < train_cut:
        return "train"
    if digest < val_cut:
        return "val"
    return "test"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "datasets")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    buckets: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    families_seen: dict[str, str] = {}

    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            family = _family_key(record, args.seed)
            if family not in families_seen:
                families_seen[family] = assign_split(family)
            buckets[families_seen[family]].append(line)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, lines in buckets.items():
        out = args.out_dir / f"sayso_{split}.jsonl"
        out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"{split}: {len(lines)} examples -> {out}")

    print(f"Families: {len(families_seen)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
