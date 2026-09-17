"""Tests for the in-memory evaluation adapter."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evals.adapters.in_memory import InMemoryAdapter
from evals.cases import select_cases
from evals.outcomes import Outcome
from evals.runner import evaluate


def test_evaluate_checkpoint_passes_identical_tool_call_prediction() -> None:
    case = next(
        item
        for item in select_cases(tag="recipe-lock")
        if item.expected["calls"]
    )
    call = case.expected["calls"][0]
    args = ", ".join(f"{key}={value!r}" for key, value in call["arguments"].items())

    def ask(messages, tools):
        return {"choices": [{"message": {"role": "assistant", "content": f"[{call['name']}({args})]"}}]}

    results = evaluate([case], InMemoryAdapter(ask))
    assert len(results) == 1
    assert results[0].passed
    assert results[0].outcome in {Outcome.VALID_TOOL_CALL, Outcome.VALID_MULTI_TOOL_CALL}


def test_evaluate_checkpoint_malformed_output_is_not_a_pass() -> None:
    case = next(
        item
        for item in select_cases(tag="recipe-lock")
        if item.expected["response_type"] in {"clarification", "refusal"}
    )

    def ask(messages, tools):
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "[intent__HassTurnOn(name='Lamp'"}}
            ]
        }

    results = evaluate([case], InMemoryAdapter(ask))
    assert results[0].outcome == Outcome.MALFORMED_OUTPUT
    assert not results[0].passed
