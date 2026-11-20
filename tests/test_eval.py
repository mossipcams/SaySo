from __future__ import annotations

from collections import Counter

from evals.adapters.in_memory import InMemoryAdapter
from evals.cases import load_all_cases, load_suite, select_cases
from evals.outcomes import Outcome
from evals.runner import evaluate
from evals.scorer import summarize


def test_cases_have_unique_ids() -> None:
    ids = [case.id for case in load_all_cases()]
    assert ids
    assert len(ids) == len(set(ids))
    assert all("datasets" not in case_id for case_id in ids)


def test_cases_cover_required_behaviors() -> None:
    categories = {case.category for case in select_cases(suite="promotion")}
    assert {
        "ordinary",
        "status",
        "ambiguity",
        "unavailable",
        "multi_action",
    } <= categories


def test_smoke_and_promotion_are_id_filters() -> None:
    assert set(load_suite("smoke")) <= set(load_suite("promotion"))
    assert len(load_suite("smoke")) == 24
    assert len(load_suite("promotion")) == 530
    assert dict(Counter(case.category for case in select_cases(suite="promotion"))) == {
        "ordinary": 130,
        "light_fan_settings": 76,
        "climate": 19,
        "media": 51,
        "timers": 53,
        "routines_vacuum": 48,
        "status": 18,
        "aliases": 47,
        "multi_action": 46,
        "ambiguity": 14,
        "unavailable": 28,
    }


def test_shared_runner_scores_a_perfect_smoke_case() -> None:
    case = select_cases(case_id="realistic_eval_20260908_ordinary_01")[0]
    call = case.expected["calls"][0]
    args = ", ".join(f"{key}={value!r}" for key, value in call["arguments"].items())

    def ask(messages, tools):
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"[{call['name']}({args})]",
                    }
                }
            ]
        }

    result = evaluate([case], InMemoryAdapter(ask))[0]
    assert result.passed
    assert result.outcome == Outcome.VALID_TOOL_CALL
    assert summarize([result])["passed"] == 1
