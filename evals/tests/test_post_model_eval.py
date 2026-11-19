from __future__ import annotations

import urllib.error
from typing import Any

from evals.cases import case_index, load_all_homes, select_cases
from evals.outcomes import QUERY_TOOLS, TODO_GET_ITEMS_TOOL, Outcome
from evals.runner import _action_tools, evaluate, production_catalog
from homeassistant.components.homeassistant import llm as ha_llm
from homeassistant.components.llm import llm as core_llm
from homeassistant.components.todo import llm as todo_llm
from homeassistant.helpers import llm
from custom_components.sayso.schema import _is_query_tool


def _content(calls: list[dict[str, Any]]) -> str:
    parts = []
    for call in calls:
        args = ", ".join(f"{k}={v!r}" for k, v in call["arguments"].items())
        parts.append(f"[{call['name']}({args})]")
    return " ".join(parts)


def _reply(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class _Stub:
    name = "stub"

    def __init__(self, first: Any) -> None:
        self.first = first
        self.requests: list[list[dict[str, Any]]] = []

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        self.requests.append(messages)
        index = len(self.requests) - 1
        if index == 0:
            return self.first(messages)
        return _reply("Done.")


def _expected_then_done(case_id: str) -> _Stub:
    case = case_index()[case_id]
    return _Stub(lambda messages: _reply(_content(case.expected["calls"])))


def _single(case_id: str):
    return select_cases(case_id=case_id)


def test_correction_recovers_invalid_arguments() -> None:
    case = _single("realistic_eval_20260908_ordinary_01")
    expected = case[0].expected["calls"][0]
    bad = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "bad1", "type": "function", "function": {"name": expected["name"], "arguments": '{"name": "Kitchen Counter Lights", "bogus": 1}'}}
    ]}}]}
    good = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "good1", "type": "function", "function": {"name": expected["name"], "arguments": '{"name": "Kitchen Counter Lights"}'}}
    ]}}]}
    replies = [bad, good, _reply("Done.")]

    class _Scripted(_Stub):
        def complete(self, messages, tools):
            self.requests.append(messages)
            return replies.pop(0)

    stub = _Scripted(None)
    result = evaluate(case, stub)[0]
    assert result.first_pass == Outcome.INVALID_ARGUMENTS
    assert result.outcome == Outcome.VALID_TOOL_CALL
    assert result.passed is True
    assert "correction_used" in result.info_flags
    assert "done_shortcut" in result.info_flags
    assert result.turn.raw == bad
    assert [call["arguments"] for call in result.turn.calls] == [{"name": "Kitchen Counter Lights"}]


def test_area_drop_removes_unspoken_area_and_is_reported() -> None:
    case = _single("realistic_eval_20260908_routines_vacuum_07")
    assert case[0].expected["calls"][0]["arguments"].get("area")
    result = evaluate(case, _expected_then_done(case[0].id))[0]
    assert "area_dropped" in result.info_flags
    assert result.turn.calls == [
        {"name": case[0].expected["calls"][0]["name"], "arguments": {"name": "Kitchen Robot Vacuum"}}
    ]
    assert result.first_pass == Outcome.VALID_TOOL_CALL


def test_grounding_clarify_is_reported_and_executes_nothing() -> None:
    case = _single("realistic_eval_20260908_media_01")
    result = evaluate(case, _expected_then_done(case[0].id))[0]
    assert "grounding_clarify" in result.info_flags
    assert result.turn.calls == []
    assert result.turn.text.startswith("Did you mean")
    assert result.outcome == Outcome.MISSING_TOOL_CALL
    assert result.first_pass == Outcome.VALID_TOOL_CALL


def test_done_shortcut_skips_follow_up_request() -> None:
    case = _single("realistic_eval_20260908_ordinary_01")
    stub = _expected_then_done(case[0].id)
    result = evaluate(case, stub)[0]
    assert result.turn.text == "Done."
    assert "done_shortcut" in result.info_flags
    assert len(stub.requests) == 1
    assert result.passed is True


def test_status_query_is_followed_by_a_model_request() -> None:
    case = _single("realistic_eval_20260908_status_01")
    stub = _expected_then_done(case[0].id)
    result = evaluate(case, stub)[0]
    assert len(stub.requests) == 2
    assert stub.requests[1][-1]["role"] == "tool"
    assert result.turn.text == "Done."
    assert "done_shortcut" not in result.info_flags
    assert [call["name"] for call in result.turn.calls] == ["homeassistant__GetLiveContext"]
    assert result.passed is True


def test_transport_error_sets_flag_and_fails_run() -> None:
    case = _single("realistic_eval_20260908_ordinary_01")

    def down(messages):
        raise urllib.error.URLError("connection refused")

    result = evaluate(case, _Stub(down))[0]
    assert "transport_error" in result.flags
    assert result.passed is False
    assert result.turn.raw == {"transport_error": "<urlopen error connection refused>"}


def test_parse_error_is_still_malformed_output() -> None:
    case = _single("realistic_eval_20260908_ordinary_01")
    result = evaluate(case, _Stub(lambda messages: {"choices": [{"message": {"role": "assistant", "content": None}}]}))[0]
    assert result.outcome == Outcome.MALFORMED_OUTPUT
    assert result.first_pass == Outcome.MALFORMED_OUTPUT
    assert result.turn.parse_error is not None
    assert "transport_error" not in result.flags


def _ha_tool(name: str) -> llm.Tool:
    if name == "homeassistant__GetLiveContext":
        return object.__new__(ha_llm.GetLiveContextTool)
    if name == "llm__GetDateTime":
        return object.__new__(core_llm.GetDateTimeTool)
    if name == TODO_GET_ITEMS_TOOL:
        return object.__new__(todo_llm.TodoGetItemsTool)
    tool = object.__new__(llm.IntentTool)
    tool.name = name.split("__", 1)[1]
    return tool


def test_action_tools_match_schema_query_classification() -> None:
    names: set[str] = set()
    for home in load_all_homes().values():
        names.update(tool["function"]["name"] for tool in production_catalog(home))
    queries = {name for name in names if _is_query_tool(_ha_tool(name))}
    assert queries == names & (QUERY_TOOLS | {TODO_GET_ITEMS_TOOL})
    assert _action_tools(names) == frozenset(names - queries)
