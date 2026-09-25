#!/usr/bin/env python3
"""Cheap static gate for a generated SaySo training JSONL corpus."""

from __future__ import annotations

import argparse
import functools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from adapters.schema import v2_openai_tools  # noqa: E402
from generators.validation import check_quality_eval_overlap  # noqa: E402
from generators.tools import namespaced_tool_name  # noqa: E402


def normalized(text: str) -> str:
    return " ".join("".join(c.lower() if c.isalnum() else " " for c in text).split())


@functools.lru_cache(maxsize=128)
def _validator(schema_json: str) -> Draft202012Validator:
    return Draft202012Validator(json.loads(schema_json))


def inspect_dataset(
    path: Path, config: dict[str, Any], baseline: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    limits = config["limits"]
    requirements = config["requirements"]
    allowed_schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in v2_openai_tools()}
    known_tool_names = {
        wire_name: name
        for name in allowed_schemas
        for wire_name in (name, namespaced_tool_name(name))
    }
    errors: list[str] = []
    row_count = valid_rows = 0
    invalid_rows = invalid_calls = 0
    positive_tools: Counter[str] = Counter()
    negative_tools: Counter[str] = Counter()
    positives: Counter[str] = Counter()
    negatives: Counter[str] = Counter()
    families: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    prompt_labels: dict[tuple[str, str], set[str]] = defaultdict(set)
    prompt_counts: Counter[tuple[str, str]] = Counter()
    contaminated = 0
    core_tools = {"llm__GetDateTime", "homeassistant__GetLiveContext", "intent__HassTurnOn", "intent__HassTurnOff"}

    if not path.is_file():
        return {
            "rows": 0, "invalid_rows": 0, "invalid_calls": 0,
            "positive_tools": Counter(), "positives": Counter(), "negatives": Counter(),
            "families": Counter(), "categories": Counter(), "duplicate_rate": 0,
            "conflict_rate": 0, "contaminated": 0,
        }, [f"missing dataset: {path}"]

    for fixture in config.get("critical_fixtures", []):
        if not (ROOT / fixture).is_file():
            errors.append(f"missing critical SaySo fixture: {fixture}")

    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row_count += 1
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("row must be an object")
                messages, tools, meta = row.get("messages"), row.get("tools"), row.get("metadata", {})
                if not isinstance(messages, list) or not messages or not isinstance(tools, list) or not isinstance(meta, dict):
                    raise ValueError("messages/tools/metadata have invalid types")
                targets = meta.get("expected_target_names", [])
                unavailable = meta.get("unavailable_tools", [])
                if not isinstance(targets, list) or not all(isinstance(target, str) for target in targets):
                    raise ValueError("metadata.expected_target_names must be a list of strings")
                if not isinstance(unavailable, list) or not all(isinstance(tool, str) for tool in unavailable):
                    raise ValueError("metadata.unavailable_tools must be a list of strings")
                if any(
                    not isinstance(message, dict)
                    or message.get("role") not in {"system", "user", "assistant", "tool"}
                    or not isinstance(message.get("content", ""), str)
                    or ("train_on_turn" in message and not isinstance(message["train_on_turn"], bool))
                    for message in messages
                ):
                    raise ValueError("messages must have valid roles, string content, and boolean train_on_turn")
                utterance = next((m.get("content") for m in messages if isinstance(m, dict) and m.get("role") == "user"), None)
                supervised = [m for m in messages if isinstance(m, dict) and m.get("role") == "assistant" and m.get("train_on_turn")]
                if not isinstance(utterance, str) or not utterance.strip() or not supervised:
                    raise ValueError("missing user utterance or supervised assistant turn")
                tool_map: dict[str, dict[str, Any]] = {}
                for tool in tools:
                    fn = tool.get("function") if isinstance(tool, dict) else None
                    if tool.get("type") != "function" or not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not isinstance(fn.get("parameters"), dict):
                        raise ValueError("malformed offered tool schema")
                    base = known_tool_names.get(fn["name"])
                    dynamic_script = base is None and fn["parameters"].get("properties") == {}
                    if base not in allowed_schemas and not dynamic_script:
                        raise ValueError(f"unknown offered tool {fn['name']!r}")
                    if base and fn["parameters"] != allowed_schemas[base]:
                        raise ValueError(f"offered schema differs from pinned SaySo tool {base!r}")
                    if fn["name"] in tool_map:
                        raise ValueError(f"duplicate offered tool {fn['name']!r}")
                    tool_map[fn["name"]] = fn
                offered = set(tool_map)
                if not core_tools <= offered:
                    raise ValueError(f"missing core tools {sorted(core_tools - offered)}")
            except (json.JSONDecodeError, ValueError, AttributeError) as exc:
                invalid_rows += 1
                if len(errors) < 50:
                    errors.append(f"{path.name}:{line_no}: {exc}")
                continue

            valid_rows += 1
            if check_quality_eval_overlap(utterance):
                contaminated += 1
            family = str(meta.get("family") or "unknown")
            category = str(meta.get("category") or "unknown")
            families[family] += 1
            categories[category] += 1
            context = json.dumps([meta.get("home_id"), meta.get("area_context")], sort_keys=True)
            key = (normalized(utterance), context)
            prompt_counts[key] += 1

            row_positive_tools: set[str] = set()
            row_calls: list[tuple[str, dict[str, Any]]] = []
            for message in supervised:
                calls = message.get("tool_calls", [])
                if not isinstance(calls, list):
                    invalid_calls += 1
                    if len(errors) < 50:
                        errors.append(f"{path.name}:{line_no}: tool_calls must be a list")
                    continue
                for call in calls:
                    try:
                        fn = call["function"]
                        name = fn["name"]
                        args = fn["arguments"]
                        if (
                            call.get("type") != "function"
                            or not isinstance(call.get("id"), str)
                            or not call["id"]
                            or not isinstance(name, str)
                            or name not in tool_map
                        ):
                            raise ValueError(f"unknown or malformed tool {name!r}")
                        if not isinstance(args, str):
                            raise ValueError("function.arguments must be a JSON string")
                        arguments = json.loads(args)
                        if not isinstance(arguments, dict):
                            raise ValueError("function.arguments must decode to an object")
                        base = known_tool_names.get(name, name)
                        schema = allowed_schemas.get(base, tool_map[name]["parameters"])
                        _validator(json.dumps(schema, sort_keys=True)).validate(arguments)
                        target = arguments.get("name")
                        expected_targets = meta.get("expected_target_names") or []
                        if target and (not isinstance(expected_targets, list) or target not in expected_targets):
                            raise ValueError(f"unknown target {target!r} (not in metadata.expected_target_names)")
                        row_positive_tools.add(name)
                        row_calls.append((name, arguments))
                    except Exception as exc:
                        invalid_calls += 1
                        if len(errors) < 50:
                            errors.append(f"{path.name}:{line_no}: invalid tool call: {exc}")
            positive_tools.update(row_positive_tools)
            capability = str(meta.get("capability") or "unknown")
            if row_positive_tools:
                positives[capability] += 1
            elif meta.get("no_action_reason"):
                negatives[capability] += 1
            for tool in meta.get("unavailable_tools") or []:
                negative_tools[namespaced_tool_name(tool)] += 1
            signature = json.dumps([(name, args) for name, args in row_calls], sort_keys=True)
            prompt_labels[key].add(signature)

    duplicates = sum(count - 1 for count in prompt_counts.values())
    conflicts = sum(len(labels) > 1 for labels in prompt_labels.values())
    report: dict[str, Any] = {
        "rows": row_count,
        "invalid_rows": invalid_rows,
        "invalid_calls": invalid_calls,
        "positive_tools": positive_tools,
        "negative_tools": negative_tools,
        "positives": positives,
        "negatives": negatives,
        "families": families,
        "categories": categories,
        "duplicate_rate": duplicates / row_count if row_count else 0,
        "conflict_rate": conflicts / row_count if row_count else 0,
        "contaminated": contaminated,
    }

    if row_count < requirements["minimum_rows"]:
        errors.append(f"rows {row_count} < minimum {requirements['minimum_rows']}")
    for capability, minimum in requirements.get("minimum_positive_by_capability", {}).items():
        actual = positives[capability]
        if actual < minimum:
            errors.append(f"positive examples for {capability}: {actual} < {minimum}")
    for tool, minimum in requirements.get("minimum_positive_by_tool", {}).items():
        actual = positive_tools[tool]
        if actual < minimum:
            errors.append(f"positive examples for {tool}: {actual} < {minimum}")
    for family, minimum in requirements.get("minimum_family_examples", {}).items():
        actual = families[family]
        if actual < minimum:
            errors.append(f"{family} examples: {actual} < {minimum}")
    for category, minimum in requirements.get("minimum_category_examples", {}).items():
        actual = categories[category]
        if actual < minimum:
            errors.append(f"{category} category examples: {actual} < {minimum}")
    for tool in requirements.get("required_positive_tools", []):
        if not positive_tools[tool]:
            errors.append(f"required tool has zero positive examples: {tool}")
    for capability, maximum in limits.get("max_negative_ratio_by_capability", {}).items():
        ratio = negatives[capability] / max(1, positives[capability])
        if ratio > maximum:
            errors.append(f"negative/positive ratio for {capability}: {ratio:.3f} > {maximum:.3f}")
    for tool, maximum in limits.get("max_negative_ratio_by_tool", {}).items():
        ratio = negative_tools[tool] / max(1, positive_tools[tool])
        if ratio > maximum:
            errors.append(f"negative/positive ratio for {tool}: {ratio:.3f} > {maximum:.3f}")
    if report["duplicate_rate"] > limits["max_duplicate_rate"]:
        errors.append(f"duplicate rate {report['duplicate_rate']:.3%} > {limits['max_duplicate_rate']:.3%}")
    if report["conflict_rate"] > limits["max_conflict_rate"]:
        errors.append(f"conflicting example rate {report['conflict_rate']:.3%} > {limits['max_conflict_rate']:.3%}")
    if contaminated:
        errors.append(f"train/eval contamination: {contaminated} utterance(s) match active eval cases")
    base_shares = baseline.get("dataset", {}).get("family_shares", {})
    for family, base_share in base_shares.items():
        share = families[family] / max(1, valid_rows)
        if abs(share - base_share) > limits["max_distribution_shift"]:
            errors.append(f"distribution shift for {family}: {share:.1%} vs baseline {base_share:.1%} (max ±{limits['max_distribution_shift']:.1%})")
    return report, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "training/configs/preflight.yaml")
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset = args.dataset or ROOT / config["dataset"]
    baseline_path = args.baseline or ROOT / config["baseline"]
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    report, errors = inspect_dataset(dataset, config, baseline)
    total = report["rows"]
    print("SaySo Training Preflight\n------------------------")
    print(f"Rows:                 {total:,}")
    print(f"Invalid rows:         {report['invalid_rows']:,}")
    print(f"Invalid tool calls:   {report['invalid_calls']:,}")
    media_pos = report["positives"].get("media_players", 0)
    media_neg = report["negatives"].get("media_players", 0)
    print(f"Media positives:      {media_pos:,}")
    print(f"Media negatives:      {media_neg:,}")
    print(f"Ambiguity examples:   {report['categories'].get('ambiguity', 0):,}")
    print(f"Unavailable examples: {report['families'].get('unavailable', 0):,}")
    print(f"Multi-action rows:    {report['families'].get('multi_action', 0):,}")
    print(f"Status-query rows:    {report['families'].get('status', 0):,}")
    print(f"Duplicate rate:      {report['duplicate_rate']:.2%}")
    coverage_terms = ("tool", "target", "media_players", "capability", "positive examples", "negative/positive ratio")
    coverage_failed = any(any(term in error for term in coverage_terms) for error in errors)
    print(f"Tool coverage:        {'FAIL' if coverage_failed else 'PASS'}")
    print(f"Distribution drift:   {'FAIL' if any('distribution shift' in e for e in errors) else 'PASS'}")
    for error in errors:
        print(f"FAIL: {error}", file=sys.stderr)
    print(f"STATIC PREFLIGHT: {'FAIL' if errors else 'PASS'}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
