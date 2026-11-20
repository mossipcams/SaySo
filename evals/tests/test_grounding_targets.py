from __future__ import annotations

import pytest

from custom_components.sayso.core.grounding import (
    _area_noun_match,
    _matches_dom,
    is_grounded,
    labels,
    support,
    tokens,
)
from custom_components.sayso.core.vocabulary import DOMAIN_WORDS
from evals.cases import case_index, load_home
from evals.runner import _grounding_home

TARGET_IDS = (
    "realistic_eval_20260908_light_fan_settings_08",
    "realistic_eval_20260908_light_fan_settings_09",
    "realistic_eval_20260908_light_fan_settings_10",
    "realistic_eval_20260908_media_01",
    "realistic_eval_20260908_media_03",
    "realistic_eval_20260908_media_04",
    "realistic_eval_20260908_media_05",
    "realistic_eval_20260908_media_06",
    "realistic_eval_20260908_media_08",
    "realistic_eval_20260908_media_10",
    "realistic_eval_20260908_ordinary_03",
    "realistic_eval_20260908_ordinary_09",
    "realistic_eval_20260908_routines_vacuum_01",
    "realistic_eval_20260908_routines_vacuum_02",
    "realistic_eval_20261118_ordinary_04",
    "realistic_gen_1_light_fan_settings_014",
    "realistic_gen_1_media_006",
    "realistic_gen_1_media_007",
    "realistic_gen_1_routines_vacuum_004",
    "realistic_gen_1_routines_vacuum_005",
    "realistic_gen_2_ordinary_021",
    "realistic_gen_2_routines_vacuum_003",
    "realistic_gen_2_routines_vacuum_004",
    "realistic_gen_3_routines_vacuum_003",
    "realistic_gen_3_routines_vacuum_005",
    "realistic_gen_5_ordinary_004",
    "realistic_gen_5_ordinary_009",
    "realistic_gen_5_ordinary_016",
)

RESIDUAL_ID = "realistic_eval_20261118_ordinary_03"

AMBIGUITY_IDS = (
    "realistic_eval_20260908_ambiguity_01",
    "realistic_eval_20260908_ambiguity_02",
    "realistic_eval_20260908_ambiguity_03",
    "realistic_gen_1_ambiguity_001",
    "realistic_gen_1_ambiguity_002",
    "realistic_gen_1_ambiguity_003",
    "realistic_gen_2_ambiguity_001",
    "realistic_gen_3_ambiguity_001",
    "realistic_gen_4_ambiguity_001",
    "realistic_gen_4_ambiguity_002",
    "realistic_gen_4_ambiguity_003",
    "realistic_gen_4_ambiguity_004",
    "realistic_gen_5_ambiguity_001",
    "realistic_gen_5_ambiguity_002",
)


def _case_and_home(case_id: str):
    case = case_index()[case_id]
    return case, _grounding_home(load_home(case.household))


@pytest.mark.parametrize("case_id", TARGET_IDS)
def test_expected_call_is_grounded(case_id: str) -> None:
    case, home = _case_and_home(case_id)
    (call,) = case.expected["calls"]
    assert is_grounded(call["name"], call["arguments"], case.utterance, home) is True


def test_residual_librarian_tv_is_grounded() -> None:
    case, home = _case_and_home(RESIDUAL_ID)
    (call,) = case.expected["calls"]
    assert is_grounded(call["name"], call["arguments"], case.utterance, home) is True


@pytest.mark.parametrize("case_id", AMBIGUITY_IDS)
def test_ambiguity_utterance_grounds_no_new_candidate(case_id: str) -> None:
    case, home = _case_and_home(case_id)
    domains = {DOMAIN_WORDS[t] for t in tokens(case.utterance) if t in DOMAIN_WORDS}
    for entity in home.entities:
        if not any(_matches_dom(entity, dom) for dom in domains):
            continue
        grounded = is_grounded(
            "intent__HassTurnOn",
            {"domain": [entity.domain], "name": entity.name},
            case.utterance,
            home,
        )
        existing = any(
            support(label, case.utterance) >= 0.5 for label in labels(entity)
        ) or _area_noun_match(entity, case.utterance, home)
        assert not grounded or existing, entity.name
