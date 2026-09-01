"""Core evaluation corpus validation tests."""

from __future__ import annotations

from collections import Counter

import pytest

from evals.corpus import (
    CORE_CASE_COUNT,
    CORE_CATEGORY_COUNTS,
    CORE_DATASET_PATH,
    load_core_corpus,
    load_home_graph_entity_ids,
    load_home_graph_origin_areas,
    validate_core_corpus,
    verify_expected_resolutions,
)
from evals.schema import EvalCase


def test_core_dataset_file_exists() -> None:
    assert CORE_DATASET_PATH.is_file()


def test_core_corpus_has_expected_case_count() -> None:
    cases = load_core_corpus()
    assert len(cases) == CORE_CASE_COUNT == 120


def test_core_corpus_category_counts_match() -> None:
    cases = load_core_corpus()
    validate_core_corpus(cases)


def test_core_corpus_categories_exact() -> None:
    cases = load_core_corpus()
    counts = Counter(case.category for case in cases)
    assert dict(counts) == CORE_CATEGORY_COUNTS


def test_core_corpus_case_ids_are_unique() -> None:
    cases = load_core_corpus()
    case_ids = [case.case_id for case in cases]
    assert len(case_ids) == len(set(case_ids))


def test_core_corpus_uses_eval_home_graph() -> None:
    cases = load_core_corpus()
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


def test_core_corpus_expected_resolutions_match_resolver() -> None:
    cases = load_core_corpus()
    verify_expected_resolutions(cases)


def test_core_corpus_rows_are_eval_cases() -> None:
    cases = load_core_corpus()
    assert all(isinstance(case, EvalCase) for case in cases)


def test_validate_core_corpus_rejects_wrong_count() -> None:
    cases = load_core_corpus()[:1]
    with pytest.raises(ValueError, match="case count"):
        validate_core_corpus(cases)
