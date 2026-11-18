from __future__ import annotations

import json
import re
from collections import Counter

import pytest

from evals import generate_realistic as gen
from evals.cases import (
    _read_jsonl,
    entity_names_for_tag,
    excluded_train_utterances,
    load_home,
    normalize_utterance,
    select_cases,
)
from evals.outcomes import Turn
from evals.runner import render_case
from evals.scorer import score

ROWS = [
    json.loads(line)
    for line in gen.OUTPUT.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
CASES = _read_jsonl(gen.OUTPUT)


def test_promotion_is_reviewed_plus_generated_with_exact_composition() -> None:
    promotion = select_cases(suite="promotion")
    generated = [case.id for case in promotion if "generated" in case.tags]
    assert generated == [row["id"] for row in ROWS]
    assert len(promotion) == 530
    assert dict(Counter(case.category for case in promotion)) == {
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


def test_regeneration_is_byte_identical() -> None:
    assert gen.render() == gen.OUTPUT.read_text(encoding="utf-8")


def test_generated_utterances_are_excluded_from_training() -> None:
    excluded = {normalize_utterance(text) for text in excluded_train_utterances()}
    assert {normalize_utterance(row["utterance"]) for row in ROWS} <= excluded


def test_generated_expected_entities_are_promotion_entities() -> None:
    names: set[str] = set()
    for row in ROWS:
        expected = row["expected"]
        names.update(expected["spoken_aliases"])
        for call in expected["calls"]:
            if "name" in call["arguments"] and "Timer" not in call["name"]:
                names.add(call["arguments"]["name"])
        if expected["clarification"]:
            names.update(expected["clarification"]["candidates"])
    assert names
    assert names <= entity_names_for_tag("promotion")


def test_check_flag_passes_on_committed_file() -> None:
    assert gen.main(["--check"]) == 0


def test_check_flag_fails_on_stale_file(tmp_path) -> None:
    stale = tmp_path / "stale.jsonl"
    stale.write_text("", encoding="utf-8")
    assert gen.main(["--check", "--output", str(stale)]) == 1


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row["id"])
def test_row_passes_generator_validation(row: dict) -> None:
    gen.validate_case(row)


def test_rows_parse_with_case_loader_schema() -> None:
    assert [case.as_dict() for case in CASES] == ROWS
    assert all(case.id.startswith("realistic_gen_") for case in CASES)
    assert len({case.id for case in CASES}) == len(CASES)


def test_composition_bounds() -> None:
    gen.check_composition(ROWS)
    counts = Counter(row["category"] for row in ROWS)
    assert 422 <= len(ROWS) <= 922
    assert set(counts) == set(gen.CATEGORIES)
    assert "exclusion" not in counts
    assert 0.02 <= counts["ambiguity"] / len(ROWS) <= 0.06
    assert 0.03 <= counts["unavailable"] / len(ROWS) <= 0.06
    assert counts["status"] / len(ROWS) <= 0.04


def test_vacuum_utterances_never_use_a_room_vacuum_or_name_it_twice() -> None:
    vacuum_rows = [
        row
        for row in ROWS
        if any(call["name"].startswith("vacuum__") for call in row["expected"]["calls"])
    ]
    assert vacuum_rows
    for row in vacuum_rows:
        home = load_home(row["household"])
        text = f" {normalize_utterance(row['utterance'])} "
        for area in home["areas"]:
            assert f" {normalize_utterance(area)} vacuum " not in text, row["id"]
        labels = sorted(
            {
                normalize_utterance(label)
                for entity in home["entities"]
                if entity["domain"] == "vacuum"
                for label in [entity["name"], *(entity.get("aliases") or [])]
            },
            key=len,
            reverse=True,
        )
        for label in labels:
            text = re.sub(rf"(?<= ){re.escape(label)}(?= )", "vacuumref", text)
        mentions = len(re.findall(r"(?<= )the (?:robot )?vacuum(?= )", text)) + text.count(
            "vacuumref"
        )
        assert mentions <= 1, row["id"]


def test_alias_rows_use_an_alias_different_from_the_entity_name() -> None:
    alias_rows = [row for row in ROWS if row["category"] == "aliases"]
    assert alias_rows
    for row in alias_rows:
        ((name, alias),) = row["expected"]["spoken_aliases"].items()
        assert alias.casefold() != name.casefold(), row["id"]


def test_no_near_duplicate_rows_beyond_stated_exceptions() -> None:
    seen: Counter = Counter()
    limits: dict = {}
    for row in ROWS:
        for key, limit in gen.near_keys(row, load_home(row["household"])):
            seen[(row["household"], key)] += 1
            limits[(row["household"], key)] = limit
    assert not [key for key, count in seen.items() if count > limits[key]]
    wide = {"climate__HassClimateSetTemperature", "intent__HassStartTimer", "clarify"}
    assert all(key[0] in wide for (_, key), limit in limits.items() if limit == 2)


def test_no_near_duplicate_of_a_literal_reviewed_case_in_the_household() -> None:
    reviewed = [
        json.loads(line)
        for line in gen.REVIEWED.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    counts: Counter = Counter()
    for case in reviewed:
        home = load_home(case["household"])
        if gen.literal(case, home):
            counts.update((case["household"], key) for key, _ in gen.near_keys(case, home))
    crowded = []
    for row in ROWS:
        for key, limit in gen.near_keys(row, load_home(row["household"])):
            counts[(row["household"], key)] += 1
            if counts[(row["household"], key)] > limit:
                crowded.append(row["id"])
    assert not crowded
    utterances = {(row["household"], row["utterance"]) for row in ROWS}
    assert (gen.household_id(2), "Switch on the bathroom lights.") not in utterances
    assert not {
        "Set the heating to 69 degrees.",
        "Start the welcome home routine.",
    } & {row["utterance"] for row in ROWS}


def test_misheard_reviewed_cases_do_not_block_literal_counterparts() -> None:
    reviewed = [
        json.loads(line)
        for line in gen.REVIEWED.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    misheard = {
        case["utterance"]
        for case in reviewed
        if not gen.literal(case, load_home(case["household"]))
    }
    assert {"Turn on the ribbon room lights.", "Turn off librarian TV."} <= misheard
    for number in gen.HOUSEHOLDS:
        home = load_home(gen.household_id(number))
        literal_keys = Counter(
            key
            for case in reviewed
            if case["household"] == gen.household_id(number)
            and case["utterance"] not in misheard
            for key, _ in gen.near_keys(case, home)
        )
        assert gen.reviewed_near(number) == literal_keys


def test_no_vacuum_with_the_construction() -> None:
    assert not [
        row["id"]
        for row in ROWS
        if re.search(r"\bwith the\b.*\bvacuum\b", row["utterance"], re.IGNORECASE)
    ]


def test_alias_equal_to_name_counts_as_the_name_form() -> None:
    home = load_home(gen.household_id(4))
    fan = next(entity for entity in home["entities"] if entity["name"] == "Bedroom Fan")
    assert "bedroom fan" in [alias.casefold() for alias in fan.get("aliases") or []]
    fan_rows = [
        row
        for row in ROWS
        if row["household"] == gen.household_id(4)
        and len(row["expected"]["calls"]) == 1
        and row["expected"]["calls"][0]["arguments"].get("name") == "Bedroom Fan"
    ]
    assert fan_rows
    keys = Counter()
    for row in fan_rows:
        ((key, limit),) = gen.near_keys(row, home)
        assert key[2] == "name", row["id"]
        keys[key] += 1
        assert keys[key] <= limit, row["id"]


def test_no_collision_with_reviewed_cases() -> None:
    reviewed = gen.reviewed_utterances()
    assert reviewed
    assert not [
        row["id"] for row in ROWS if normalize_utterance(row["utterance"]) in reviewed
    ]


def test_no_cross_household_repeat_with_same_answer() -> None:
    keys = Counter(gen.answer_key(row) for row in ROWS)
    assert not [key[0] for key, count in keys.items() if count > 1]
    by_utterance: dict[str, set[str]] = {}
    for row in ROWS:
        utterance, answer = gen.answer_key(row)
        by_utterance.setdefault(utterance, set()).add(answer)
    assert any(len(answers) > 1 for answers in by_utterance.values())


def test_composition_rejects_cross_household_repeat() -> None:
    taken = {r["utterance"] for r in ROWS if r["household"] == gen.household_id(2)}
    row = next(
        r
        for r in ROWS
        if r["household"] == gen.household_id(1) and r["utterance"] not in taken
    )
    copy = json.loads(json.dumps(row))
    copy["household"] = gen.household_id(2)
    copy["id"] = "realistic_gen_2_copy_001"
    with pytest.raises(gen.GenerationError, match="cross-household"):
        gen.check_composition([*ROWS, copy])


def test_validation_rejects_unresolved_name() -> None:
    row = json.loads(json.dumps(next(r for r in ROWS if r["category"] == "ordinary")))
    row["expected"]["calls"][0]["arguments"]["name"] = "Garage Heater"
    row["expected"]["calls"][0]["arguments"].pop("area", None)
    with pytest.raises(gen.GenerationError):
        gen.validate_case(row)


def _turn(case) -> Turn:
    expected = case.expected
    if expected["response_type"] == "clarification":
        names = " or ".join(expected["clarification"]["candidates"])
        return Turn(raw=None, text=f"Which one did you mean, {names}?")
    if expected["response_type"] == "refusal":
        return Turn(raw=None, text="Sorry, I don't have that device.")
    return Turn(raw=None, text="", calls=[dict(call) for call in expected["calls"]])


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_expected_calls_score_as_passed(case) -> None:
    expectation = render_case(case)["expectation"]
    result = score(case.id, expectation, _turn(case), home=load_home(case.household))
    assert result.passed, (result.outcome, result.flags)
