"""Audit canonical or TRL JSONL before training: python -m generators.audit FILE."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from adapters.schema import tool_schema_map, validate_tool_arguments
from generators.pipeline import _check_quality_eval_overlap


def audit_rows(rows, *, expected_count=None):
    counts = Counter()
    casing = {"call": Counter(), "no_call": Counter()}
    seen = set()
    for row in rows:
        meta = row.get("metadata", {})
        row_id = meta.get("candidate_id")
        if not row_id or row_id in seen:
            raise ValueError(f"missing or duplicate candidate ID: {row_id}")
        seen.add(row_id)
        messages = row["messages"]
        user = next(m["content"] for m in messages if m["role"] == "user")
        if not user.strip() or _check_quality_eval_overlap(user):
            raise ValueError(f"empty request or eval overlap: {row_id}")
        calls = [c["function"] for m in messages for c in m.get("tool_calls", [])]
        schemas = tool_schema_map(row["tools"])
        for call in calls:
            arguments = call["arguments"]
            arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
            reason = validate_tool_arguments(call["name"], arguments, schemas)
            if reason:
                raise ValueError(f"{row_id}: {reason}")
            if arguments.get("name") in meta.get("excluded_names", []):
                raise ValueError(f"excluded target called: {row_id}")
        if set(meta.get("unavailable_tools", [])) & schemas.keys():
            raise ValueError(f"blocked tool is offered: {row_id}")
        excluded = meta.get("excluded_names", [])
        if excluded and ("leave" not in user.lower() or any(name.lower() not in user.lower() for name in excluded)):
            raise ValueError(f"missing exclusion wording: {row_id}")
        if meta.get("category") in {"multi_action", "exclusion"} and len(calls) < 2:
            raise ValueError(f"false multiple-action tag: {row_id}")
        if meta.get("category") == "exclusion" and not excluded:
            raise ValueError(f"false exclusion tag: {row_id}")
        kind = "call" if calls else "no_call"
        casing[kind]["upper" if user[0].isupper() else "lower"] += 1
        counts["rows"] += 1
        counts[kind] += 1
        counts["tool_calls"] += len(calls)
        counts["multi_call_rows"] += len(calls) > 1
        counts["exclusion_rows"] += bool(excluded)
        counts["alias_rows"] += any(alias.lower() in user.lower() and name.lower() not in user.lower()
                                    for name, alias in meta.get("spoken_targets", {}).items())
        counts["conversational_rows"] += bool(re.search(r"\b(?:please|could you|can you)\b", user, re.I))
    if expected_count is not None and counts["rows"] != expected_count:
        raise ValueError(f"expected {expected_count} rows, got {counts['rows']}")
    for kind, cases in casing.items():
        total = sum(cases.values())
        if total >= 10 and not 0.25 <= cases["upper"] / total <= 0.75:
            raise ValueError(f"label-dependent casing for {kind}: {dict(cases)}")
    if counts["rows"] >= 1000:
        for behavior in ("multi_call_rows", "exclusion_rows", "alias_rows"):
            if counts[behavior] < counts["rows"] * 0.005:
                raise ValueError(f"insufficient actual {behavior}: {counts[behavior]}")
    return {**counts, "casing": {kind: dict(cases) for kind, cases in casing.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--count", type=int)
    args = parser.parse_args()
    with args.dataset.open() as handle:
        print(json.dumps(audit_rows((json.loads(line) for line in handle), expected_count=args.count), indent=2))
