"""Evaluator loading, suites, and migration coverage."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from evals.cases import (
    ARCHIVE_DIR,
    CaseError,
    cases_with_tag,
    load_all_cases,
    load_gates,
    load_home,
    load_suite,
    select_cases,
)
from evals.runner import render_case
from sayso_contract import tool_contract

MAP = json.loads((Path(__file__).resolve().parent / "migration_map.json").read_text(encoding="utf-8"))
V1_SOURCE_IDS = [
    "sayso-eval-v1-core-turn-on-light",
    "sayso-eval-v1-failure-wrong-tool",
    "sayso-eval-v1-failure-invalid-args",
    "sayso-eval-v1-safety-ambiguous-light",
    "sayso-eval-v1-query-live-context",
    "sayso-eval-v1-multi-scene-sequence",
    "sayso-eval-v1-failure-partial-execution",
    "sayso-eval-v1-followup-spoken-confirmation",
]


def test_archived_fixture_contents_are_unchanged() -> None:
    archived = ARCHIVE_DIR / "realistic_eval_20260908_v2.json"
    assert hashlib.sha256(archived.read_bytes()).hexdigest() == (
        "60ebb318d76b87d45cc41cc03cf143c1a24f8c9b0f9b8f7ce0a0a18df7ee8bea"
    )
    md = ARCHIVE_DIR / "realistic_eval_20260908_v2.md"
    assert md.is_file() and md.stat().st_size > 0


def test_archive_is_not_discovered_as_cases() -> None:
    assert not any(path.suffix == ".jsonl" for path in ARCHIVE_DIR.glob("*"))
    for case in load_all_cases():
        assert "archive" not in case.source


def test_every_suite_id_resolves_uniquely() -> None:
    all_ids = [case.id for case in load_all_cases()]
    assert len(all_ids) == len(set(all_ids))
    smoke = load_suite("smoke")
    promotion = load_suite("promotion")
    assert len(smoke) == len(set(smoke)) == 24
    assert len(promotion) == len(set(promotion)) == 120
    assert set(smoke) <= set(promotion)


def test_smoke_covers_required_behaviors() -> None:
    cases = select_cases(suite="smoke")
    categories = {case.category for case in cases}
    assert {
        "ordinary",
        "aliases",
        "multi_action",
        "status",
        "ambiguity",
        "unavailable",
        "exclusion",
    } <= categories
    sources = {render_case(case)["area_context"]["target_area_source"] for case in cases}
    assert {"explicit", "satellite"} <= sources
    held_out_smoke = [
        case.id
        for case in cases
        if {"grounding", "gold", "shadow", "recipe-lock"} & set(case.tags)
    ]
    assert held_out_smoke == []


def test_held_out_cases_are_not_in_smoke() -> None:
    smoke = set(load_suite("smoke"))
    for tag in ("grounding", "gold", "shadow", "recipe-lock"):
        assert not {case.id for case in cases_with_tag(tag)} & smoke


def test_promotion_denominator_is_only_the_original_120() -> None:
    promotion = select_cases(suite="promotion")
    assert len(promotion) == 120
    assert dict(Counter(case.category for case in promotion)) == {
        category: 10 for category in {case.category for case in promotion}
    }
    assert all("promotion" in case.tags for case in promotion)
    assert all("regression" not in case.tags for case in promotion)


def test_useful_migrated_cases_retain_expected_behavior() -> None:
    index = {case.id: case for case in load_all_cases()}
    retained = [row for row in MAP if row["status"] == "retained" and row["canonical_id"]]
    assert len(retained) >= 120 + 38
    for row in retained:
        case = index[row["canonical_id"]]
        assert case.utterance
        assert case.expected["response_type"] in {"action", "status", "clarification", "refusal"}
        if case.expected["response_type"] in {"action", "status"}:
            assert case.expected["calls"]


def test_v1_and_adversarial_coverage_is_recorded() -> None:
    by_source = {(row["source"], row["source_id"]): row for row in MAP}
    for source_id in V1_SOURCE_IDS:
        assert by_source[("evals/cases/v1.json", source_id)]["canonical_id"]
    assert by_source[("training/evals/adversarial.jsonl", "adversarial.jsonl")]["status"] == (
        "equivalent_toy_prompts"
    )


def test_case_labels_validate_against_production_schemas() -> None:
    for case in load_all_cases():
        load_home(case.household)
        rendered = render_case(case)
        assert not tool_contract.check_tool_calls(
            [(call["name"], call["arguments"]) for call in case.expected["calls"]],
            rendered["tools"],
            rendered["exposed_domains"],
        ), case.id


def test_select_unknown_case_fails() -> None:
    try:
        select_cases(case_id="not-a-real-case")
    except CaseError:
        return
    raise AssertionError("expected CaseError")


def test_gates_exist_for_both_suites() -> None:
    assert load_gates("smoke")["zero_tolerance_flags"]
    assert load_gates("promotion")["category_min_pass_rate"]
