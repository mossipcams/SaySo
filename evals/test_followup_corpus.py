"""Follow-up evaluation corpus validation tests."""

from __future__ import annotations

from collections import Counter

import pytest

from evals.corpus import (
    FOLLOWUP_CASE_COUNT,
    FOLLOWUP_CATEGORY_COUNTS,
    FOLLOWUP_DATASET_PATH,
    load_followup_corpus,
    validate_followup_corpus,
    verify_follow_up_resolutions,
)
from evals.corpus import FollowUpEvalCase


def test_followup_dataset_file_exists() -> None:
    assert FOLLOWUP_DATASET_PATH.is_file()


def test_followup_corpus_has_expected_case_count() -> None:
    cases = load_followup_corpus()
    assert len(cases) == FOLLOWUP_CASE_COUNT
    assert 60 <= FOLLOWUP_CASE_COUNT <= 80


def test_followup_corpus_category_counts_match() -> None:
    cases = load_followup_corpus()
    validate_followup_corpus(cases)


def test_followup_corpus_categories_exact() -> None:
    cases = load_followup_corpus()
    counts = Counter(case.category for case in cases)
    assert dict(counts) == FOLLOWUP_CATEGORY_COUNTS


def test_followup_corpus_case_ids_are_unique() -> None:
    cases = load_followup_corpus()
    case_ids = [case.case_id for case in cases]
    assert len(case_ids) == len(set(case_ids))


def test_followup_corpus_cases_are_paired_turns() -> None:
    cases = load_followup_corpus()
    for case in cases:
        assert len(case.turns) == 2, f"{case.case_id} must have exactly 2 turns"


def test_followup_corpus_rows_are_follow_up_eval_cases() -> None:
    cases = load_followup_corpus()
    assert all(isinstance(case, FollowUpEvalCase) for case in cases)


def test_followup_corpus_follow_up_resolutions_match_resolver() -> None:
    cases = load_followup_corpus()
    verify_follow_up_resolutions(cases)


def test_validate_followup_corpus_rejects_wrong_count() -> None:
    cases = load_followup_corpus()[:1]
    with pytest.raises(ValueError, match="case count"):
        validate_followup_corpus(cases)
