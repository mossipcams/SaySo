"""Both adapters feed the shared scorer."""

from __future__ import annotations

from evals.adapters.endpoint import EndpointAdapter
from evals.adapters.in_memory import InMemoryAdapter
from evals.cases import select_cases
from evals.outcomes import Outcome
from evals.runner import evaluate
from evals.scorer import score, summarize


def test_in_memory_adapter_uses_shared_scorer() -> None:
    case = select_cases(case_id="realistic_eval_20260908_ordinary_01")[0]
    expected = case.expected["calls"]

    def ask(messages, tools):
        args = ", ".join(f"{k}={v!r}" for k, v in expected[0]["arguments"].items())
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"[{expected[0]['name']}({args})]",
                    }
                }
            ]
        }

    results = evaluate([case], InMemoryAdapter(ask))
    assert results[0].passed
    assert results[0].outcome == Outcome.VALID_TOOL_CALL
    assert summarize(results)["passed"] == 1


def test_malformed_in_memory_cannot_pass_as_abstention() -> None:
    case = select_cases(case_id="realistic_eval_20260908_ambiguity_01")[0]

    def ask(messages, tools):
        return {"choices": [{"message": {"role": "assistant", "content": "[intent__HassTurnOn(name='x'"}}]}

    result = evaluate([case], InMemoryAdapter(ask))[0]
    assert result.outcome == Outcome.MALFORMED_OUTPUT
    assert not result.passed


def test_endpoint_adapter_requires_a_server() -> None:
    try:
        EndpointAdapter("")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_adapters_share_the_score_function() -> None:
    assert InMemoryAdapter(lambda messages, tools: None).name == "in_memory"
    assert EndpointAdapter("http://127.0.0.1:8080").name == "endpoint"
    assert score.__module__ == "evals.scorer"
