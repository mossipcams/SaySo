from __future__ import annotations

from evals.outcomes import Expectation, Outcome, ResponseType, Turn
from evals.scorer import classify

WANT = {"name": "GetLiveContext", "arguments": {"name": "TV", "domain": ["media_player"]}}


def _classify(domain) -> Outcome:
    call = {"name": "GetLiveContext", "arguments": {"name": "TV", "domain": domain}}
    expectation = Expectation(category="status", response_type=ResponseType.STATUS, calls=(WANT,))
    return classify(expectation, Turn(raw={"calls": [call]}, calls=[call]))


def test_scalar_domain_matches_single_item_list() -> None:
    # The schema accepts domain as a string or a list for GetLiveContext.
    assert _classify("media_player") == Outcome.VALID_TOOL_CALL
    assert _classify(["media_player"]) == Outcome.VALID_TOOL_CALL


def test_different_domain_still_fails() -> None:
    assert _classify("sensor") == Outcome.UNEXPECTED_TOOL_CALL
