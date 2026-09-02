"""Tests for leakage-resistant dataset splitting."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_split_assigns_whole_families(tmp_path: Path) -> None:
    records = []
    for template in ("turn_on", "turn_off"):
        for seed in (1, 2):
            records.append(
                {
                    "messages": [{"role": "user", "content": f"{template}-{seed}"}],
                    "metadata": {
                        "template_family": template,
                        "phrasing_family": "direct",
                        "seed": seed,
                    },
                }
            )
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "split"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "split_dataset.py"),
            str(input_path),
            "--out-dir",
            str(out_dir),
            "--seed",
            "42",
        ],
        check=True,
    )
    train = (out_dir / "sayso_train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    val = (out_dir / "sayso_val.jsonl").read_text(encoding="utf-8").strip().splitlines()
    test = (out_dir / "sayso_test.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(train) + len(val) + len(test) == len(records)
    # Same template+phrasing+seed must not appear in multiple splits
    def families(lines: list[str]) -> set[str]:
        result = set()
        for line in lines:
            meta = json.loads(line).get("metadata", {})
            result.add(
                f"{meta.get('template_family')}|{meta.get('phrasing_family')}|{meta.get('seed')}"
            )
        return result

    assert not (families(train) & families(val))
    assert not (families(train) & families(test))
    assert not (families(val) & families(test))
