from __future__ import annotations

import copy
import json
from collections import defaultdict

from evals import runner
from evals.cases import Case, fingerprint_cases, load_all_cases
from evals.runner import fingerprint_inputs, render_case


def _input_key(case: Case) -> str:
    rendered = render_case(case)
    return json.dumps(
        {"messages": rendered["messages"], "tools": rendered["tools"]},
        sort_keys=True,
    )


def _expectation_groups(cases: tuple[Case, ...] | list[Case]) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        expectation = json.dumps(
            [case.expected["response_type"], case.expected["calls"]], sort_keys=True
        )
        groups[_input_key(case)].add(expectation)
    return groups


def test_identical_rendered_inputs_share_one_expectation() -> None:
    conflicting = {
        key: expectations
        for key, expectations in _expectation_groups(load_all_cases()).items()
        if len(expectations) > 1
    }
    assert not conflicting


def test_input_hash_tracks_entity_contents_not_case_hash(monkeypatch) -> None:
    cases = [
        case
        for case in load_all_cases()
        if case.household == "grounding_porch_speaker_capable"
    ]
    assert cases
    original_home = runner.load_home

    baseline_input = fingerprint_inputs(cases)
    baseline_case = fingerprint_cases(cases)

    def renamed_home(home_id: str) -> dict:
        home = copy.deepcopy(original_home(home_id))
        for entity in home["entities"]:
            entity["name"] = f"{entity['name']} Annex"
        return home

    monkeypatch.setattr(runner, "load_home", renamed_home)

    assert fingerprint_inputs(cases) != baseline_input
    assert fingerprint_cases(cases) == baseline_case
