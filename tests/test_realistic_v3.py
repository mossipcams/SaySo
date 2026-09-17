"""Realistic promotion cases: one production contract, explicit outcomes, enforced gates."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

import pytest
import voluptuous as vol
from homeassistant.helpers import llm

from custom_components.sayso import const
from custom_components.sayso.area_context import build_area_context, render_system_prompt
from custom_components.sayso.schema import compile_tools, emit_tools_source_json
from custom_components.sayso.tool_schema import compile_source_tools
from evals.adapters.in_memory import InMemoryAdapter
from evals.cases import ARCHIVE_DIR, load_gates, load_home, select_cases
from evals.outcomes import PRODUCTION_MAX_OUTPUT_TOKENS, PRODUCTION_TEMPERATURE, Outcome
from evals.runner import evaluate, production_catalog, render_case, training_path
from evals.scorer import check_gates, is_clarification, is_refusal, summarize
from sayso_contract import tool_contract

PROMOTION = select_cases(suite="promotion")
CASES = {case.id.removeprefix("realistic_eval_20260908_"): case for case in PROMOTION}
GATES = load_gates("promotion")
V2_ARCHIVE = ARCHIVE_DIR / "realistic_eval_20260908_v2.json"


def _body(content: str | None = None, tool_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}]}


def _python_calls(calls: list[dict[str, Any]]) -> str:
    return "[{}]".format(
        ", ".join(
            "{}({})".format(call["name"], ", ".join(f"{k}={v!r}" for k, v in call["arguments"].items()))
            for call in calls
        )
    )


def _run(short_id: str, completion: dict[str, Any]):
    case = CASES[short_id]
    rendered = render_case(case)
    from evals.outcomes import read_completion
    from evals.scorer import score

    turn = read_completion(completion, rendered["tools"], rendered["exposed_domains"])
    return score(case.id, rendered["expectation"], turn)


def test_promotion_keeps_120_cases_with_ten_per_category() -> None:
    counts = Counter(case.category for case in PROMOTION)
    assert sum(counts.values()) == 120
    assert len(counts) == 12
    assert set(counts.values()) == {10}
    assert len({case.id for case in PROMOTION}) == 120


def test_archived_v2_fixture_is_byte_identical_and_matches_utterances() -> None:
    v2 = json.loads(V2_ARCHIVE.read_text(encoding="utf-8"))
    digest = hashlib.sha256(V2_ARCHIVE.read_bytes()).hexdigest()
    assert digest == PROMOTION[0].provenance["derived_from"]["sha256"]
    v2_cases = [
        (case["metadata"]["candidate_id"], next(m["content"] for m in case["messages"] if m["role"] == "user"))
        for case in v2["cases"]
    ]
    assert v2_cases == [(case.id, case.utterance) for case in PROMOTION]


def test_cases_store_structured_inputs_not_rendered_prompts() -> None:
    from evals.cases import CASES_DIR

    text = (CASES_DIR / "realistic_v3.jsonl").read_text(encoding="utf-8")
    assert "You are in area" not in text
    assert "Static Context" not in text
    for case in PROMOTION:
        assert not {"messages", "context", "tool_names"} & set(case.as_dict())


def test_spoken_aliases_exist_in_the_household() -> None:
    for case in PROMOTION:
        entities = {e["name"]: e for e in load_home(case.household)["entities"]}
        for name, alias in case.expected["spoken_aliases"].items():
            assert alias in entities[name]["aliases"], (case.id, name, alias)


def test_eval_prompt_is_the_production_renderer_applied_to_the_ha_prompt() -> None:
    with training_path():
        from generators.context import serialize_context  # noqa: PLC0415

    for case in PROMOTION:
        home = load_home(case.household)
        rendered = render_case(case)["messages"][0]["content"]
        ha_prompt = rendered.rsplit("\nArea context:\n", 1)[0] + (
            f"\nYou are in area {home['sayso_entity_area']} (floor Main Floor) and all generic"
            " commands like 'turn on the lights' should target this area."
        )
        context = build_area_context(case.utterance, home["area_aliases"], home["sayso_entity_area"])
        assert rendered == render_system_prompt(ha_prompt, context)
        assert rendered == serialize_context(home, case.utterance)


class _Tool(llm.Tool):
    def __init__(self, name: str, parameters: vol.Schema, description: str) -> None:
        self.name = name
        self.parameters = parameters
        self.description = description

    async def async_call(self, hass, tool_input, llm_context):  # pragma: no cover
        return {}


def test_compiling_tool_source_matches_compiling_live_ha_tools() -> None:
    tools = [
        llm.NamespacedTool(
            "intent",
            _Tool(
                "HassTurnOn",
                vol.Schema({vol.Optional("name"): str, vol.Optional("domain"): [str], vol.Optional("area"): str}),
                "Turns on a device",
            ),
        ),
        llm.NamespacedTool(
            "light",
            _Tool("HassLightSet", vol.Schema({vol.Optional("brightness"): vol.All(int, vol.Range(0, 100))}), "Sets a light"),
        ),
    ]
    live = compile_tools(tools, custom_serializer=llm.selector_serializer)
    from_source = compile_source_tools(json.loads(emit_tools_source_json(tools, custom_serializer=llm.selector_serializer)))
    assert json.dumps(live) == json.dumps(from_source.tools)
    assert [tool["function"]["name"] for tool in live] == ["intent__HassTurnOn", "light__HassLightSet"]


def test_catalog_is_realistic_not_answer_conditioned() -> None:
    catalogs: dict[str, set[str]] = {}
    for case in PROMOTION:
        if case.unavailable:
            continue
        names = json.dumps(render_case(case)["tools"])
        catalogs.setdefault(case.household, set()).add(names)
    assert len(catalogs) == 5
    for household, variants in catalogs.items():
        assert len(variants) == 1, household
        assert len(json.loads(next(iter(variants)))) > 8


def test_unavailable_cases_remove_only_the_recorded_capability() -> None:
    for short_id, case in CASES.items():
        if case.category != "unavailable":
            assert case.unavailable is None
            continue
        normal = production_catalog(load_home(case.household))
        offered = render_case(case)["tools"]
        removed = set(case.unavailable["removed_tools"])
        assert [t for t in normal if t["function"]["name"] not in removed] == offered, short_id


def test_every_expected_call_is_valid_under_the_production_contract() -> None:
    for case in PROMOTION:
        rendered = render_case(case)
        calls = case.expected["calls"]
        assert not tool_contract.check_tool_calls(
            [(c["name"], c["arguments"]) for c in calls], rendered["tools"], rendered["exposed_domains"]
        ), case.id


def test_eval_constants_match_production() -> None:
    assert PRODUCTION_TEMPERATURE == const.DEFAULT_TEMPERATURE
    assert PRODUCTION_MAX_OUTPUT_TOKENS == const.DEFAULT_MAX_OUTPUT_TOKENS


def test_a_perfect_run_passes_every_case_and_every_gate() -> None:
    by_utterance = {case.utterance: case for case in PROMOTION}

    def ask(messages, tools):
        case = by_utterance[messages[1]["content"]]
        expected = case.expected
        if expected["calls"]:
            return _body(_python_calls(expected["calls"]))
        if expected["response_type"] == "clarification":
            return _body("Which one did you mean?")
        return _body("I can't do that with the available Home Assistant tools.")

    results = evaluate(PROMOTION, InMemoryAdapter(ask))
    summary = summarize(results)
    assert summary["passed"] == 120, [r.as_dict() for r in results if not r.passed][:3]
    assert check_gates(summary, GATES, expected_count=120) == []


def test_malformed_output_fails_instead_of_becoming_zero_calls() -> None:
    result = _run("ambiguity_01", _body("[intent__HassTurnOn(name='Living Room Floor Lamp'"))
    assert result.outcome == Outcome.MALFORMED_OUTPUT
    assert result.turn.calls == []
    assert not result.passed
    summary = summarize([result])
    assert summary["malformed_output_rate"] == 1.0
    assert summary["flags"]["malformed_treated_as_abstention"] == 0


def test_a_truncated_call_is_malformed_not_prose() -> None:
    result = _run("unavailable_03", _body("intent__HassTurnOn(name='Garage Heater'"))
    assert result.outcome == Outcome.MALFORMED_OUTPUT


def test_empty_output_is_malformed() -> None:
    assert _run("unavailable_03", _body("")).outcome == Outcome.MALFORMED_OUTPUT


def test_a_refusal_does_not_pass_an_ambiguity_case() -> None:
    result = _run("ambiguity_01", _body("I can't do that."))
    assert result.outcome == Outcome.WRONG_RESPONSE_TYPE
    assert not result.passed


def test_a_clarification_passes_an_ambiguity_case() -> None:
    result = _run("ambiguity_01", _body("Which reading light do you mean, the floor lamp or the table lamp?"))
    assert result.outcome == Outcome.VALID_CLARIFICATION
    assert result.passed


def test_a_clarification_does_not_pass_an_unavailable_case() -> None:
    result = _run("unavailable_01", _body("Which light did you mean?"))
    assert result.outcome == Outcome.WRONG_RESPONSE_TYPE
    assert not result.passed
    assert _run("unavailable_01", _body("I can't set brightness with the available tools.")).passed


def test_a_tool_call_on_an_unavailable_case_fails_its_gate() -> None:
    result = _run("unavailable_03", _body(_python_calls([{"name": "intent__HassTurnOn", "arguments": {"name": "Garage Heater"}}])))
    assert not result.passed
    assert "tool_call_on_unavailable" in result.flags
    assert check_gates(summarize([result]), GATES, expected_count=1)


def test_an_action_on_an_ambiguity_case_fails_its_gate() -> None:
    call = {"name": "intent__HassTurnOn", "arguments": {"name": "Living Room Floor Lamp", "domain": ["light"]}}
    result = _run("ambiguity_01", _body(_python_calls([call])))
    assert result.outcome == Outcome.UNEXPECTED_TOOL_CALL
    assert "action_on_ambiguity" in result.flags
    assert any("action_on_ambiguity" in failure for failure in check_gates(summarize([result]), GATES, expected_count=1))


def test_duplicate_names_across_areas_require_clarification() -> None:
    household = load_home(CASES["ambiguity_02"].household)
    candidates = CASES["ambiguity_02"].expected["clarification"]["candidates"]
    assert len(candidates) >= 2
    by_name = {e["name"]: e for e in household["entities"]}
    assert all("bedside lamp" in by_name[name]["aliases"] for name in candidates)
    call = {"name": "intent__HassTurnOff", "arguments": {"name": candidates[0], "domain": ["light"]}}
    assert not _run("ambiguity_02", _body(_python_calls([call]))).passed


def test_a_status_action_call_fails() -> None:
    call = {"name": "intent__HassTurnOn", "arguments": {"name": "Front Door Lock"}}
    result = _run("status_01", _body(_python_calls([call])))
    assert not result.passed
    assert "status_state_change" in result.flags
    expected = CASES["status_01"].expected["calls"]
    assert expected[0]["name"] == "homeassistant__GetLiveContext"
    assert _run("status_01", _body(_python_calls(expected))).passed


@pytest.mark.parametrize(
    "calls",
    [
        [
            {"name": "intent__HassTurnOff", "arguments": {"domain": ["light"], "name": "Kitchen Counter Lights"}},
            {"name": "intent__HassTurnOff", "arguments": {"domain": ["light"], "name": "Kitchen Sink Light"}},
            {"name": "intent__HassTurnOff", "arguments": {"domain": ["light"], "name": "Kitchen Ceiling Light"}},
        ],
        [{"name": "intent__HassTurnOff", "arguments": {"area": "Kitchen", "domain": ["light"]}}],
    ],
)
def test_an_exclusion_that_affects_the_excluded_entity_fails(calls: list[dict[str, Any]]) -> None:
    result = _run("exclusion_01", _body(_python_calls(calls)))
    assert not result.passed
    assert "excluded_entity_affected" in result.flags


def test_partial_multi_action_output_fails() -> None:
    expected = CASES["multi_action_01"].expected["calls"]
    result = _run("multi_action_01", _body(_python_calls(expected[:1])))
    assert result.outcome == Outcome.MISSING_TOOL_CALL
    assert not result.passed
    assert _run("multi_action_01", _body(_python_calls(list(reversed(expected))))).outcome == (
        Outcome.VALID_MULTI_TOOL_CALL
    )


def test_bare_legacy_tool_names_fail() -> None:
    bare = [{"name": "HassTurnOn", "arguments": {"domain": ["light"], "name": "Kitchen Counter Lights"}}]
    result = _run("ordinary_01", _body(_python_calls(bare)))
    assert result.outcome == Outcome.UNKNOWN_TOOL
    assert not result.passed
    structured = _body(tool_calls=[{"id": "c1", "type": "function", "function": {"name": "HassTurnOn", "arguments": json.dumps(bare[0]["arguments"])}}])
    assert _run("ordinary_01", structured).outcome == Outcome.UNKNOWN_TOOL


def test_an_invalid_domain_is_invalid_arguments() -> None:
    call = {"name": "intent__HassTurnOn", "arguments": {"domain": ["lights"], "name": "Kitchen Counter Lights"}}
    assert _run("ordinary_01", _body(_python_calls([call]))).outcome == Outcome.INVALID_ARGUMENTS
    missing_name = _body(tool_calls=[{"id": "c1", "type": "function", "function": {"name": "", "arguments": "{}"}}])
    assert _run("ordinary_01", missing_name).outcome == Outcome.MALFORMED_OUTPUT


def test_explicit_area_overrides_the_satellite_area() -> None:
    case = CASES["routines_vacuum_08"]
    assert render_case(case)["area_context"] == {
        "satellite_area": "Kitchen",
        "target_area": "Living Room",
        "target_area_source": "explicit",
    }
    satellite_area = {"name": "vacuum__HassVacuumCleanArea", "arguments": {"area": "Kitchen", "name": "Roomba"}}
    result = _run("routines_vacuum_08", _body(_python_calls([satellite_area])))
    assert not result.passed
    assert "wrong_target_area" in result.flags
    assert _run("routines_vacuum_08", _body(_python_calls(case.expected["calls"]))).passed


def test_an_implicit_area_request_resolves_to_the_satellite_area() -> None:
    areas = load_home("realistic_eval_household_1")["areas"]
    context = build_area_context("turn off the lights", areas, "Kitchen")
    assert (context.target_area, context.target_area_source) == ("Kitchen", "satellite")
    assert render_case(CASES["ambiguity_10"])["area_context"]["target_area_source"] == "satellite"


@pytest.mark.parametrize(
    ("text", "clarification", "refusal"),
    [
        ("Which device did you mean?", True, False),
        ("Do you mean the floor lamp or the table lamp?", True, False),
        ("I can't do that with the available Home Assistant tools.", False, True),
        ("There is no garage heater in this home.", False, True),
        ("The Garage has no lights available.", False, True),
        ("I can’t tell which one, which did you mean?", True, False),
        ("Done.", False, False),
        ("What would you like?", False, False),
        ("", False, False),
    ],
)
def test_response_classification_is_narrow(text: str, clarification: bool, refusal: bool) -> None:
    assert is_clarification(text) is clarification
    assert is_refusal(text) is refusal


def test_gate_config_is_versioned_and_covers_every_category() -> None:
    assert GATES["version"] == 1
    assert set(GATES["category_min_pass_rate"]) == {case.category for case in PROMOTION}
    assert set(GATES["zero_tolerance_flags"]) >= {
        "action_on_ambiguity",
        "tool_call_on_unavailable",
        "malformed_treated_as_abstention",
        "status_state_change",
        "excluded_entity_affected",
    }


def test_incomplete_run_cannot_pass_promotion() -> None:
    failures = check_gates(summarize([]), GATES, expected_count=120)
    assert any("incomplete run" in failure for failure in failures)
