"""Audit canonical or TRL JSONL before training: python -m generators.audit FILE."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from adapters.schema import tool_schema_map, validate_tool_arguments
from generators.capability_registry import SCRIPT_ACTION_TOOL
from generators.coverage import ABSENCE, classify_row, expected_tool
from generators.pipeline import _check_quality_eval_overlap


def audit_rows(
    rows,
    *,
    expected_count=None,
    required_operations=None,
    min_positive_per_operation=0,
    min_positive_per_tool=0,
    max_absence_rate=None,
):
    """Audit accepted rows on what they actually label, not on their metadata.

    ``required_operations`` is the set of ``(tier, capability, operation)`` buckets
    the run planned positive supervision for; each must reach
    ``min_positive_per_operation`` rows that really call its tool, and every tool
    those buckets map to must reach ``min_positive_per_tool``. ``max_absence_rate``
    caps how much of the corpus may be an "there is no such device" answer.
    """
    counts = Counter()
    casing = {"call": Counter(), "no_call": Counter()}
    seen = set()
    grammar_sources = Counter()
    grammar_templates = set()
    requests = set()
    positive_by_operation = Counter()
    positive_by_tool = Counter()
    by_outcome = Counter()
    by_targeting = Counter()
    by_tool = Counter()
    by_domain = Counter()
    negatives_by_reason = Counter()
    real_home_rows = 0
    for row in rows:
        facets = classify_row(row)
        by_outcome[facets["outcome"]] += 1
        by_targeting[facets["targeting"]] += 1
        real_home_rows += bool(row.get("metadata", {}).get("real_home") or facets["real_home"])
        for tool in facets["tools"]:
            by_tool[tool] += 1
        if facets["positive"]:
            key = (facets["tier"], facets["capability"], facets["operation"])
            positive_by_operation[key] += 1
            by_domain[(facets["capability"], facets["operation"], facets["targeting"])] += 1
            tool = expected_tool(facets["capability"], facets["operation"])
            if tool:
                positive_by_tool[tool] += 1
        else:
            negatives_by_reason[facets["no_action_reason"] or facets["outcome"]] += 1
        meta = row.get("metadata", {})
        row_id = meta.get("candidate_id")
        if not row_id or row_id in seen:
            raise ValueError(f"missing or duplicate candidate ID: {row_id}")
        seen.add(row_id)
        messages = row["messages"]
        user = next(m["content"] for m in messages if m["role"] == "user")
        requests.add(user.casefold())
        grammar = [entry for entry in meta.get("linguistics", [])
                   if entry.get("source", "").startswith("sentences/en/")]
        counts["ohf_rows"] += bool(grammar)
        counts["fallback_rows"] += any(entry.get("source") == "sayso_fallback"
                                       for entry in meta.get("linguistics", []))
        for entry in grammar:
            grammar_sources[entry["source"]] += 1
            grammar_templates.add((entry["source"], entry["block"], entry["template"]))
        if not user.strip() or _check_quality_eval_overlap(user):
            raise ValueError(f"empty request or eval overlap: {row_id}")
        calls = [c["function"] for m in messages for c in m.get("tool_calls", [])]
        schemas = tool_schema_map(row["tools"])
        final = messages[-1].get("content", "")
        if not calls and meta.get("capability") == "timers":
            required = expected_tool("timers", meta.get("operation") or "")
            if required in schemas or re.search(r"has no .* available", final, re.I):
                raise ValueError(f"contradictory timer refusal: {row_id}")
        if any(call["name"] == "GetLiveContext" for call in calls) and final.strip().lower() == "done.":
            raise ValueError(f"state query labeled as action: {row_id}")
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
        counts["alias_rows"] += meta.get("category") != "ambiguity" and any(alias.lower() in user.lower() and name.lower() not in user.lower()
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

    missing_operations = {}
    missing_tools = {}
    if required_operations:
        for key in sorted(required_operations):
            got = positive_by_operation.get(key, 0)
            if got < min_positive_per_operation:
                missing_operations[str(key)] = got
        required_tools = {
            tool
            for (_tier, capability, operation) in required_operations
            if (tool := expected_tool(capability, operation)) and tool != SCRIPT_ACTION_TOOL
        }
        # Script tools are named per home; their positive rows are counted under
        # the placeholder, so check that separately.
        if any(cap == "scripts" and op == "run" for _t, cap, op in required_operations):
            required_tools.add(SCRIPT_ACTION_TOOL)
        for tool in sorted(required_tools):
            got = positive_by_tool.get(tool, 0)
            if got < min_positive_per_tool:
                missing_tools[tool] = got
    if missing_operations:
        raise ValueError(f"missing positive supervision for operations: {missing_operations}")
    if missing_tools:
        raise ValueError(f"missing positive supervision for tools: {missing_tools}")
    absence_rate = by_outcome.get(ABSENCE, 0) / max(counts["rows"], 1)
    if max_absence_rate is not None and absence_rate > max_absence_rate:
        raise ValueError(
            f"absence answers are {absence_rate:.1%} of the corpus, above the "
            f"{max_absence_rate:.1%} cap ({by_outcome.get(ABSENCE, 0)}/{counts['rows']})"
        )

    return {**counts, "unique_requests": len(requests),
            "ohf_source_count": len(grammar_sources), "ohf_template_count": len(grammar_templates),
            "ohf_sources": dict(sorted(grammar_sources.items())),
            "casing": {kind: dict(cases) for kind, cases in casing.items()},
            "real_home_rows": real_home_rows,
            "by_outcome": dict(sorted(by_outcome.items())),
            "by_targeting": dict(sorted(by_targeting.items())),
            "by_tool": dict(sorted(by_tool.items())),
            "negatives_by_reason": dict(sorted(negatives_by_reason.items())),
            "absence_rate": round(absence_rate, 4),
            "positive_by_tool": dict(sorted(positive_by_tool.items())),
            "positive_by_operation": {str(k): v for k, v in sorted(positive_by_operation.items())},
            "positive_by_domain_targeting": {
                f"{cap}/{op}/{mode}": n for (cap, op, mode), n in sorted(by_domain.items())
            }}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--count", type=int)
    parser.add_argument("--max-absence-rate", type=float, default=None)
    args = parser.parse_args()
    with args.dataset.open() as handle:
        report = audit_rows(
            (json.loads(line) for line in handle),
            expected_count=args.count,
            max_absence_rate=args.max_absence_rate,
        )
    print(json.dumps(report, indent=2))
