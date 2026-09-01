"""Language-noise evaluation corpus validation tests."""

from __future__ import annotations

from collections import Counter

import pytest

from evals.corpus import (
    CORE_CASE_COUNT,
    LANGUAGE_NOISE_CASE_COUNT,
    LANGUAGE_NOISE_CATEGORY_COUNTS,
    LANGUAGE_NOISE_DATASET_PATH,
    REVIEWED_CORPUS_MAX,
    REVIEWED_CORPUS_MIN,
    SAFETY_CASE_COUNT,
    load_language_noise_corpus,
    reviewed_corpus_case_count,
    validate_language_noise_corpus,
    verify_expected_resolutions,
)
from evals.schema import EvalCase


def test_language_noise_dataset_file_exists() -> None:
    assert LANGUAGE_NOISE_DATASET_PATH.is_file()


def test_language_noise_corpus_has_expected_case_count() -> None:
    cases = load_language_noise_corpus()
    assert len(cases) == LANGUAGE_NOISE_CASE_COUNT
    assert 100 <= LANGUAGE_NOISE_CASE_COUNT <= 200


def test_language_noise_corpus_category_counts_match() -> None:
    cases = load_language_noise_corpus()
    validate_language_noise_corpus(cases)


def test_language_noise_corpus_categories_exact() -> None:
    cases = load_language_noise_corpus()
    counts = Counter(case.category for case in cases)
    assert dict(counts) == LANGUAGE_NOISE_CATEGORY_COUNTS


def test_language_noise_corpus_case_ids_are_unique() -> None:
    cases = load_language_noise_corpus()
    case_ids = [case.case_id for case in cases]
    assert len(case_ids) == len(set(case_ids))


def test_reviewed_corpus_total_is_within_target_range() -> None:
    total = reviewed_corpus_case_count()
    assert REVIEWED_CORPUS_MIN <= total <= REVIEWED_CORPUS_MAX
    assert total == CORE_CASE_COUNT + SAFETY_CASE_COUNT + LANGUAGE_NOISE_CASE_COUNT


def test_language_noise_corpus_expected_resolutions_match_resolver() -> None:
    cases = load_language_noise_corpus()
    verify_expected_resolutions(cases)


def test_language_noise_corpus_rows_are_eval_cases() -> None:
    cases = load_language_noise_corpus()
    assert all(isinstance(case, EvalCase) for case in cases)


def test_validate_language_noise_corpus_rejects_wrong_count() -> None:
    cases = load_language_noise_corpus()[:1]
    with pytest.raises(ValueError, match="case count"):
        validate_language_noise_corpus(cases)
