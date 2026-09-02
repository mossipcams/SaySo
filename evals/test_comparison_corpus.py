"""Home-LLM comparison benchmark corpus validation tests."""

from __future__ import annotations

from collections import Counter

import pytest

from evals.corpus import (
    COMPARISON_CASE_COUNT,
    COMPARISON_DATASET_PATH,
    COMPARISON_SCENARIO_COUNTS,
    load_comparison_corpus,
    load_home_graph_entity_ids,
    load_home_graph_origin_areas,
    validate_comparison_corpus,
    verify_expected_resolutions,
)
from evals.schema import EvalCase, ExpectedOutcome


def test_comparison_dataset_file_exists() -> None:
    assert COMPARISON_DATASET_PATH.is_file()


def test_comparison_corpus_has_one_case_per_scenario() -> None:
    cases = load_comparison_corpus()
    assert len(cases) == COMPARISON_CASE_COUNT == 6
    counts = Counter(case.category for case in cases)
    assert dict(counts) == COMPARISON_SCENARIO_COUNTS


def test_comparison_corpus_category_counts_match() -> None:
    cases = load_comparison_corpus()
    validate_comparison_corpus(cases)


def test_comparison_corpus_case_ids_are_unique() -> None:
    cases = load_comparison_corpus()
    case_ids = [case.case_id for case in cases]
    assert len(case_ids) == len(set(case_ids))


def test_comparison_corpus_uses_eval_home_graph() -> None:
    cases = load_comparison_corpus()
    valid_origins = load_home_graph_origin_areas()
    valid_entities = load_home_graph_entity_ids()

    for case in cases:
        assert case.home == "eval-home"
        assert case.origin in valid_origins
        for entity_id in (
            *case.expected_candidate_entities,
            *case.expected_resolved_entities,
        ):
            assert entity_id in valid_entities


def test_comparison_corpus_expected_resolutions_match_resolver() -> None:
    cases = load_comparison_corpus()
    verify_expected_resolutions(cases)


def test_comparison_corpus_rows_are_eval_cases() -> None:
    cases = load_comparison_corpus()
    assert all(isinstance(case, EvalCase) for case in cases)


def test_comparison_corpus_defaults_to_dry_run_safety() -> None:
    cases = load_comparison_corpus()
    assert cases
    assert all(not case.execution_allowed for case in cases)


def test_comparison_corpus_includes_ambiguous_target_clarification() -> None:
    cases = load_comparison_corpus()
    ambiguous = [case for case in cases if case.category == "ambiguous_target"]
    assert len(ambiguous) == 1
    case = ambiguous[0]
    assert case.expected_outcome == ExpectedOutcome.CLARIFICATION
    assert case.expected_resolved_entities == []


def test_comparison_corpus_includes_multi_target_action() -> None:
    cases = load_comparison_corpus()
    multi = [case for case in cases if case.category == "multi_target"]
    assert len(multi) == 1
    case = multi[0]
    assert case.expected_outcome == ExpectedOutcome.VALID_ACTION
    assert len(case.expected_resolved_entities) >= 2


def test_validate_comparison_corpus_rejects_wrong_count() -> None:
    cases = load_comparison_corpus()[:1]
    with pytest.raises(ValueError, match="case count"):
        validate_comparison_corpus(cases)
