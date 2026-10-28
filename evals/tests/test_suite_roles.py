from __future__ import annotations

from evals.cases import load_all_cases, load_suite


def test_selection_regression_and_acceptance_suites_are_disjoint() -> None:
    dev = set(load_suite("dev"))
    regression = set(load_suite("regression"))
    acceptance = set(load_suite("promotion")) | set(load_suite("context"))
    assert dev and regression
    assert not dev & regression
    assert not (dev | regression) & acceptance
    assert dev | regression | acceptance == {case.id for case in load_all_cases()}
