from __future__ import annotations

import json
from typing import Any
from urllib import request as urlrequest

import pytest

from sayso_contract import grammar as _grammar
from evals import runner
from evals.adapters.endpoint import EndpointAdapter
from evals.cases import select_cases
from evals.runner import evaluate, render_case

GrammarUnsupported = _grammar.GrammarUnsupported
build_grammar = _grammar.build_grammar

CASE_ID = "realistic_eval_20260908_ordinary_01"
MESSAGES = [{"role": "user", "content": "turn on the lamp"}]
GRAMMAR = "root ::= block | text\n"


class _Response:
    def read(self) -> bytes:
        return b"{}"

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _posted_payload(monkeypatch: pytest.MonkeyPatch, **request: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_urlopen(request_obj, timeout=None):
        captured["payload"] = json.loads(request_obj.data.decode("utf-8"))
        return _Response()

    monkeypatch.setattr(urlrequest, "urlopen", fake_urlopen)
    EndpointAdapter("http://127.0.0.1:9").complete(MESSAGES, [], **request)
    return captured["payload"]


def test_endpoint_sends_the_grammar_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _posted_payload(monkeypatch, grammar=GRAMMAR)

    assert payload["grammar"] == GRAMMAR


def test_endpoint_omits_the_grammar_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _posted_payload(monkeypatch)

    assert "grammar" not in payload


class _Recorder:
    name = "recorder"

    def __init__(self) -> None:
        self.grammars: list[str | None] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        grammar: str | None = None,
    ) -> Any:
        self.grammars.append(grammar)
        return {"choices": [{"message": {"role": "assistant", "content": "Sorry."}}]}


def _case():
    return select_cases(case_id=CASE_ID)


def test_grammar_is_built_from_the_rendered_tools() -> None:
    adapter = _Recorder()
    cases = _case()

    evaluate(cases, adapter, grammar_markers=True)

    expected = build_grammar(render_case(cases[0])["tools"], markers=True)
    assert adapter.grammars[0] == expected
    assert "<|tool_call_start|>" in expected


def test_bare_grammar_has_no_markers() -> None:
    adapter = _Recorder()

    evaluate(_case(), adapter, grammar_markers=False)

    assert adapter.grammars[0] is not None
    assert "<|tool_call_start|>" not in adapter.grammars[0]


def test_grammar_is_off_by_default() -> None:
    adapter = _Recorder()

    results = evaluate(_case(), adapter)

    assert set(adapter.grammars) == {None}
    assert results[0].grammar_skip_reason is None


def test_unsupported_grammar_fails_open_per_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unsupported(tools, *, markers):
        raise GrammarUnsupported("nested_object")

    monkeypatch.setattr(runner, "build_grammar", unsupported)
    adapter = _Recorder()

    results = evaluate(_case(), adapter, grammar_markers=True)

    assert set(adapter.grammars) == {None}
    assert results[0].grammar_skip_reason == "nested_object"
    assert results[0].outcome is not None
