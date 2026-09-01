"""Safety evaluation corpus validation tests."""

from __future__ import annotations

from collections import Counter

import pytest

from evals.corpus import (
    SAFETY_CASE_COUNT,
    SAFETY_CATEGORY_COUNTS,
    SAFETY_DATASET_PATH,
    load_safety_corpus,
    validate_safety_corpus,
    verify_expected_resolutions,
)
from evals.schema import EvalCase, ExpectedOutcome


def test_safety_dataset_file_exists() -> None:
    assert SAFETY_DATASET_PATH.is_file()


def test_safety_corpus_has_expected_case_count() -> None:
    cases = load_safety_corpus()
    assert len(cases) == SAFETY_CASE_COUNT == 100


def test_safety_corpus_category_counts_match() -> None:
    cases = load_safety_corpus()
    validate_safety_corpus(cases)


def test_safety_corpus_categories_exact() -> None:
    cases = load_safety_corpus()
    counts = Counter(case.category for case in cases)
    assert dict(counts) == SAFETY_CATEGORY_COUNTS


def test_safety_corpus_case_ids_are_unique() -> None:
    cases = load_safety_corpus()
    case_ids = [case.case_id for case in cases]
    assert len(case_ids) == len(set(case_ids))


def test_safety_corpus_every_category_has_positive_and_negative_cases() -> None:
    cases = load_safety_corpus()
    by_category: dict[str, list[EvalCase]] = {}
    for case in cases:
        by_category.setdefault(case.category, []).append(case)

    negative_outcomes = {
        ExpectedOutcome.CLARIFICATION,
        ExpectedOutcome.UNSUPPORTED,
        ExpectedOutcome.NO_ACTION,
    }
    for category in SAFETY_CATEGORY_COUNTS:
        category_cases = by_category[category]
        positives = [
            case
            for case in category_cases
            if case.expected_outcome == ExpectedOutcome.VALID_ACTION and case.execution_allowed
        ]
        negatives = [
            case
            for case in category_cases
            if case.expected_outcome in negative_outcomes and not case.execution_allowed
        ]
        assert positives, f"{category} missing positive cases"
        assert negatives, f"{category} missing negative cases"


def test_safety_corpus_expected_resolutions_match_resolver() -> None:
    cases = load_safety_corpus()
    verify_expected_resolutions(cases)


def test_safety_corpus_rows_are_eval_cases() -> None:
    cases = load_safety_corpus()
    assert all(isinstance(case, EvalCase) for case in cases)


def test_validate_safety_corpus_rejects_wrong_count() -> None:
    cases = load_safety_corpus()[:1]
    with pytest.raises(ValueError, match="case count"):
        validate_safety_corpus(cases)
