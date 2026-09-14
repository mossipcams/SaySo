#!/usr/bin/env python3
"""Pre-flight probe: does this corpus teach entity resolution, or name echoing?

Run 013 trained fine and taught nothing about grounding. The reason was visible
in the corpus without a GPU: 96.3% of rows with a target named that target
verbatim in the user utterance, so the model could answer by echoing a name it
had just been handed. This probe is the cheap gate that should run before any
training run.

    python training/scripts/audit_discrimination.py training/datasets/<corpus>.jsonl

Exit status is 1 when the verbatim share exceeds ``--max-verbatim``, so it can
gate a build script.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_WORDS = re.compile(r"[a-z0-9]+")


def _normalize(text: str) -> str:
    return " ".join(_WORDS.findall(text.casefold()))


def _user_utterance(row: dict[str, Any]) -> str:
    return next(
        (m.get("content") or "" for m in row.get("messages", []) if m.get("role") == "user"),
        "",
    )


def names_target_verbatim(row: dict[str, Any]) -> bool | None:
    """True/False for a row with a target, None for a row that has none.

    Substring on normalized words, the same shape the corpus audit used: a row
    "names" its target when the target's words appear in the utterance in order.
    A row that refers to the target by alias or by description does not.
    """
    targets = row.get("metadata", {}).get("expected_target_names") or []
    if not targets:
        return None
    utterance = f" {_normalize(_user_utterance(row))} "
    return any(f" {_normalize(name)} " in utterance for name in targets if name)


def audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    verbatim = described = 0
    utterances: Counter[str] = Counter()
    flags: Counter[str] = Counter()
    for row in rows:
        utterances[_normalize(_user_utterance(row))] += 1
        named = names_target_verbatim(row)
        if named is None:
            continue
        verbatim += named
        described += not named
        meta = row.get("metadata", {})
        if not named:
            if meta.get("grounding_family"):
                flags["described_via_grounding"] += 1
            elif meta.get("discrimination"):
                flags["described_via_discrimination"] += 1
            else:
                flags["described_via_other"] += 1
    with_target = verbatim + described
    return {
        "rows": len(rows),
        "rows_with_target": with_target,
        "names_target_verbatim": verbatim,
        "describes_target": described,
        "verbatim_share": round(verbatim / max(with_target, 1), 4),
        "described_share": round(described / max(with_target, 1), 4),
        "described_by_source": dict(sorted(flags.items())),
        "grounding_rows": sum(
            1 for row in rows if row.get("metadata", {}).get("grounding_family")
        ),
        "discrimination_rows": sum(
            1 for row in rows if row.get("metadata", {}).get("discrimination")
        ),
        "distinct_utterances": len(utterances),
        "duplicate_utterances": len(rows) - len(utterances),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="JSONL corpus to probe")
    parser.add_argument(
        "--max-verbatim",
        type=float,
        default=0.60,
        help="Fail when this share of target rows name the target verbatim "
             "(v1 measured 0.963; the v2 contract asks for < 0.60)",
    )
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in args.corpus.read_text(encoding="utf-8").splitlines() if line]
    report = audit(rows)
    print(json.dumps(report, indent=2))
    if report["verbatim_share"] > args.max_verbatim:
        print(
            f"FAIL: {report['verbatim_share']:.1%} of target rows name the target "
            f"verbatim (limit {args.max_verbatim:.0%}). This corpus teaches name "
            "echoing, which is the Run 013 failure.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
