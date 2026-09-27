"""Tests for the v6 generator eval fixes (Fix 5: fail-closed dataset hygiene gate)."""

from __future__ import annotations

from pathlib import Path

import pytest

from generators.config import GeneratorConfig
from generators.deduplication import DuplicateTracker
from generators.pipeline import enforce_hygiene, run_generation

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def test_duplicate_tracker_rejects_second_identical_pair() -> None:
    tracker = DuplicateTracker(near_limit=3)
    spec = {"utterance": "turn on the light", "home": {"entities": []}, "semantic_id": "a"}
    assert tracker.would_reject(spec) is None
    tracker.record(spec)
    assert tracker.would_reject(spec) == "exact_duplicate_utterance"


@pytest.fixture(scope="module")
def smoke_result() -> dict:
    config = GeneratorConfig.from_yaml(ROOT / "configs" / "generation" / "smoke.yaml", repo_root=REPO)
    return run_generation(config)


def test_enforce_hygiene_raises_on_duplicated_rows(smoke_result) -> None:
    rows = smoke_result["rows"]
    padded = rows + rows[:1] * 5
    with pytest.raises(RuntimeError, match="dataset hygiene gate failed"):
        enforce_hygiene(padded, len(padded))


def test_smoke_hygiene_stats(smoke_result) -> None:
    hygiene = smoke_result["stats"]["hygiene"]
    assert hygiene["duplicate_rate"] <= 0.01
    assert hygiene["conflict_rate"] <= 0.001
    assert hygiene["invalid_rows"] == 0


def _core_tool_names() -> set[str]:
    from generators.planning import CORE_TOOL_NAMES
    from generators.tools import namespaced_tool_name

    return {namespaced_tool_name(name) for name in CORE_TOOL_NAMES}


def test_no_smoke_row_lacks_a_core_tool(smoke_result) -> None:
    """Home Assistant always offers the core tools, so every row must too."""
    core = _core_tool_names()
    for row in smoke_result["rows"]:
        offered = {t["function"]["name"] for t in row["tools"]}
        assert core <= offered, f"row missing core tools {sorted(core - offered)}"


def test_withholdable_operations_never_map_to_core_tools() -> None:
    """Unavailable rows withhold the operation's tool; a core tool is never withheld."""
    from generators.capability_registry import operation_spec
    from generators.planning import CORE_TOOL_NAMES, _WITHHOLDABLE_OPERATIONS

    for cap_name, op_name in _WITHHOLDABLE_OPERATIONS:
        op = operation_spec(cap_name, op_name)
        assert op is not None and op.tool_name not in CORE_TOOL_NAMES, (
            f"{cap_name}/{op_name} maps to a core tool"
        )


def test_decoy_removals_never_withhold_core_tools() -> None:
    import random

    from generators.homes import generate_home
    from generators.rendering import _decoy_removals

    core = _core_tool_names()
    home = generate_home(917, 16, random.Random(917))
    for index in range(64):
        spec = {
            "candidate_id": f"decoy_test_{index}",
            "home": home,
            "expected": {},
            "capability": "lights",
            "operation": "turn_on",
        }
        withheld = set(_decoy_removals(spec, []))
        assert not withheld & core, f"decoy withheld core tools {withheld & core}"


# --- Task 5a: real-home retries must not shift the attempt cadence -----------


def _exclusion_slot() -> dict:
    """A non-family lights/turn_on slot that hits the exclusion trigger at attempt 0."""
    return {
        "capability": "lights",
        "operation": "turn_on",
        "index": 0,
        "home_size": 16,
        "robustness": "ordinary",
    }


def _generate_row(attempt: int, retry: int):
    import random

    from generators.row_generation import generate_row

    return generate_row(
        _exclusion_slot(),
        GeneratorConfig(seed=1),
        random.Random(1),
        excluded=set(),
        dup_tracker=DuplicateTracker(near_limit=3),
        attempt=attempt,
        retry=retry,
    )


def test_retry_preserves_exclusion_cadence() -> None:
    """A real-home retry keeps the exclusion trigger on the loop's attempt counter."""
    row, reason = _generate_row(attempt=0, retry=1)
    assert reason is None, reason
    assert row["metadata"]["category"] == "exclusion"


def test_retry_draws_a_fresh_scenario() -> None:
    """A retry must not repeat the rejected scenario, so it gets a fresh seed."""
    base, base_reason = _generate_row(attempt=0, retry=0)
    retried, retry_reason = _generate_row(attempt=0, retry=1)
    assert base_reason is None, base_reason
    assert retry_reason is None, retry_reason
    assert base["messages"][1]["content"] != retried["messages"][1]["content"]


# --- Task 5a round 2: capped real-home runs keep the exclusion floor ---------


def test_capped_run_keeps_the_exclusion_floor() -> None:
    """Cap rejections must not eat the audit's >= 0.5% exclusion floor.

    A saturated real home rejects most real-home rows with
    ``real_home_entity_cap``; the cadence attempts (attempt % 125) that carry
    the exclusion floor must be redrawn instead of lost to those rejections.
    """
    result = run_generation(
        GeneratorConfig(
            count=1500,
            seed=515,
            real_home_path=ROOT / "fixtures" / "synthetic_reference_home.json",
            real_home_rate=0.25,
            paraphrase_enabled=False,
            real_home_entity_cap=3,
        )
    )
    audit = result["stats"]["quality_audit"]
    assert audit["exclusion_rows"] >= 0.005 * audit["rows"]


# --- Task 5b: hygiene datetime requirement and underpowered core-tool ops ----


def _row_calls_tool(row: dict, tool: str) -> bool:
    for message in row["messages"]:
        for call in message.get("tool_calls", []):
            if call.get("function", {}).get("name") == tool:
                return True
    return False


def test_hygiene_datetime_requirement_follows_coverage_config(smoke_result) -> None:
    """A run with no datetime supervision must not be forced to teach GetDateTime."""
    rows = [row for row in smoke_result["rows"] if not _row_calls_tool(row, "llm__GetDateTime")]
    assert rows
    # With zero GetDateTime positives the default gate still fails...
    with pytest.raises(RuntimeError, match="llm__GetDateTime"):
        enforce_hygiene(rows, len(rows))
    # ...but a run configured with get_datetime_positive_min=0 is exempt.
    enforce_hygiene(rows, len(rows), require_datetime=False)


def test_underpowered_operations_keep_core_tool_ops() -> None:
    """The unsupported family builds an incapable device, so core-tool ops stay eligible."""
    from generators.capability_registry import operation_spec
    from generators.planning import (
        CORE_TOOL_NAMES,
        _UNDERPOWERED_OPERATIONS,
        _WITHHOLDABLE_OPERATIONS,
    )

    def tools(pairs) -> set[str]:
        return {operation_spec(cap, op).tool_name for cap, op in pairs}

    assert tools(_UNDERPOWERED_OPERATIONS) & CORE_TOOL_NAMES, (
        "no core-tool operation is underpowered"
    )
    assert not tools(_WITHHOLDABLE_OPERATIONS) & CORE_TOOL_NAMES


# --- Task 5c: real-home repeats must not escape the exact-duplicate gate ----


def _real_home(entities: list[dict]) -> dict:
    return {"home_id": "real_home_train", "entities": entities}


def test_duplicate_tracker_rejects_real_home_repeat_despite_injected_entities() -> None:
    """Injected entities mutate the home's entity list per row. The tracker must
    key on the stable home identity, so the same (utterance, home) is rejected
    after the first occurrence even when the entity lists differ. Otherwise the
    real home emits the same generic utterance with different gold (v5b's
    0.42% conflicts) and the hygiene gate fails."""
    tracker = DuplicateTracker(near_limit=3)
    base = {"utterance": "switch off the hallway light", "home": _real_home([]), "semantic_id": "a"}
    # A later real-home row reuses the utterance but carries different injected
    # entities (build_scenario appends per row), so the entity lists differ.
    injected = {
        "utterance": "switch off the hallway light",
        "home": _real_home(
            [{"name": "Injected Distractor", "area": "Hallway", "domain": "light"}]
        ),
        "semantic_id": "b",
    }
    assert tracker.would_reject(base) is None
    tracker.record(base)
    assert tracker.would_reject(injected) == "exact_duplicate_utterance"


def test_duplicate_tracker_still_distinguishes_different_homes() -> None:
    """Keying on the home identity must not collapse distinct homes: the same
    utterance in a different home is a new pair, not a duplicate."""
    tracker = DuplicateTracker(near_limit=3)
    first = {"utterance": "switch off the hallway light", "home": _real_home([]), "semantic_id": "a"}
    other = {
        "utterance": "switch off the hallway light",
        "home": {"home_id": "real_home_holdout", "entities": []},
        "semantic_id": "b",
    }
    tracker.record(first)
    assert tracker.would_reject(other) is None


# --- Task 1a: nickname aliases, aliases draw, and the audit count ------------


def test_nickname_aliases_appear_in_generated_homes() -> None:
    """generate_home draws informal nickname aliases; none is an eval nickname,
    and nicknames are aliases only, never entity names."""
    import random

    from generators.homes import _NICKNAMES, generate_home

    eval_nicknames = {"prep lights", "sofa lamp", "my bedside light"}
    pool = {nickname.casefold() for nicknames in _NICKNAMES.values() for nickname in nicknames}
    assert pool
    assert not pool & eval_nicknames
    seen_nicknames: set[str] = set()
    for seed in range(40):
        home = generate_home(seed, 32, random.Random(seed))
        for entity in home["entities"]:
            assert entity["name"].casefold() not in pool
            for alias in entity["aliases"]:
                assert alias.casefold() not in eval_nicknames
                if alias.casefold() in pool:
                    seen_nicknames.add(alias.casefold())
    assert seen_nicknames, "no nickname alias was drawn in 40 homes of size 32"


def test_alias_distractor_rows_resolve_alias_to_canonical_with_sibling() -> None:
    """Alias rows speak an alias that is not the canonical name, gold still
    calls the canonical name, and the home always carries a same-domain,
    same-area near-miss sibling of the target so the model must disambiguate."""
    from generators.planning import _ALIASES_COMBINATIONS
    from generators.scenarios.core import build_scenario

    checked = 0
    for combo_index, (capability, operation) in enumerate(_ALIASES_COMBINATIONS):
        for block in range(25):
            scenario = build_scenario(
                index=combo_index * 1000 + block,
                seed=block * 7 + 3,
                capability=capability,
                operation=operation,
                home_size=16,
                targeting="individual",
                robustness="alias_distractor",
                family="aliases",
            )
            target = scenario["target_entity"]
            if target is None or not scenario.get("spoken_targets"):
                continue
            name, alias = next(iter(scenario["spoken_targets"].items()))
            assert name == target["name"]
            assert alias.casefold() != name.casefold()
            expected = scenario["expected"]
            assert expected["kind"] == "action"
            call = expected["calls"][0]
            assert call["arguments"].get("name") == name, (
                f"gold call must target the canonical name, got {call['arguments']}"
            )
            siblings = [
                entity
                for entity in scenario["home"]["entities"]
                if entity is not target
                and entity["domain"] == target["domain"]
                and entity["area"] == target["area"]
            ]
            assert siblings, (
                f"{capability}/{operation} block {block}: no same-domain, same-area "
                f"sibling of {name!r}"
            )
            checked += 1
    assert checked >= 50, f"only {checked} scenarios set spoken_targets"


def test_aliases_slots_cover_media_players_and_turn_off() -> None:
    """The aliases family draw must reach the v5b failure shapes: media_players
    targets and turn_off operations, not only lights x turn_on."""
    from generators.planning import build_plan

    plan = build_plan(200, seed=1, allocations={"aliases": 1.0})
    slots = [slot for slot in plan.slots if slot.family == "aliases"]
    assert slots
    capabilities = {slot.capability for slot in slots}
    operations = {slot.operation for slot in slots}
    assert "media_players" in capabilities
    assert "turn_off" in operations
    assert {"turn_on", "turn_off"} <= operations


# --- Fix 2: settings-request ambiguity for clarify/follow_up ----------------


def _settings_slot(family: str, capability: str, operation: str, index: int) -> dict:
    return {
        "index": index,
        "family": family,
        "robustness": "ambiguity",
        "capability": capability,
        "operation": operation,
        "tier": 1,
        "home_size": 16,
    }


def test_clarify_follow_up_draw_ambiguous_settings_operations() -> None:
    """v5b answered "dim the reading light to 30 percent" with HassLightSet
    despite two candidates, because clarify only covered on/off/status. About
    half of clarify/follow_up rows must draw a settings operation instead."""
    import random

    from generators.planning import AMBIGUOUS_SETTINGS, _pick_capability_operation

    expected = {(cap, op) for cap, op in AMBIGUOUS_SETTINGS}
    for family in ("clarify", "follow_up"):
        rng = random.Random(42)
        draws = [_pick_capability_operation(family, rng)[:2] for _ in range(400)]
        settings_draws = sum(1 for draw in draws if draw in expected)
        assert 0.35 < settings_draws / len(draws) < 0.65, (
            f"{family}: {settings_draws}/{len(draws)} drew an ambiguous settings op"
        )


def test_clarify_settings_rows_clarify_with_candidates() -> None:
    """Settings-op clarify rows stay a clarification turn: no tool call, and the
    question names the real candidates."""
    import random
    import re

    from generators.row_generation import generate_row

    combos = [
        ("lights", "set_brightness"),
        ("lights", "set_color"),
        ("lights", "set_color_temperature"),
        ("media_players", "volume_set"),
        ("fans", "set_speed"),
        ("climate", "set_temperature"),
    ]
    checked = 0
    for combo_index, (capability, operation) in enumerate(combos):
        for block in range(10):
            index = combo_index * 100 + block
            row, reason = generate_row(
                _settings_slot("clarify", capability, operation, index),
                GeneratorConfig(seed=2, stt_noise_rate=0.0),
                random.Random(2000 + index),
                excluded=set(),
                dup_tracker=DuplicateTracker(near_limit=3),
            )
            if reason is not None:
                continue
            assistant = row["messages"][-1]
            assert not assistant.get("tool_calls"), (capability, operation, assistant)
            content = assistant["content"]
            assert content.startswith("Did you mean"), (capability, operation, content)
            names = re.split(r", | or ", content[len("Did you mean "):-1])
            assert len(names) >= 2, (capability, operation, content)
            checked += 1
    assert checked >= 10, f"only {checked} clarify settings rows were accepted"


def test_follow_up_settings_gold_reuses_spoken_value() -> None:
    """The follow-up gold reuses the value spoken in the ambiguous first turn,
    so the value arguments of the gold call appear in that utterance."""
    import json
    import random

    from generators.row_generation import generate_row

    combos = [
        ("lights", "set_brightness"),
        ("lights", "set_color"),
        ("lights", "set_color_temperature"),
        ("media_players", "volume_set"),
        ("fans", "set_speed"),
        ("climate", "set_temperature"),
    ]
    skip = {"name", "area", "floor", "domain", "device_class"}
    checked = 0
    for combo_index, (capability, operation) in enumerate(combos):
        for block in range(10):
            index = combo_index * 100 + block
            row, reason = generate_row(
                _settings_slot("follow_up", capability, operation, index),
                GeneratorConfig(seed=3, stt_noise_rate=0.0),
                random.Random(3000 + index),
                excluded=set(),
                dup_tracker=DuplicateTracker(near_limit=3),
            )
            if reason is not None:
                continue
            messages = row["messages"]
            first_user = next(m["content"] for m in messages if m["role"] == "user")
            call = next(m for m in reversed(messages) if m.get("tool_calls"))["tool_calls"][0]
            arguments = json.loads(call["function"]["arguments"])
            for key, value in arguments.items():
                if key in skip:
                    continue
                spoken = str(value).casefold()
                assert spoken in first_user.casefold(), (
                    f"{capability}/{operation} block {block}: gold {key}={value!r} "
                    f"not spoken in {first_user!r}"
                )
            checked += 1
    assert checked >= 10, f"only {checked} follow_up settings rows were accepted"


# --- Fix 3: withheld-tool refusal with sibling tools offered -----------------


def _offered_tools(row: dict) -> set[str]:
    return {tool["function"]["name"] for tool in row["tools"]}


def _unavailable_rows(smoke_result: dict) -> list[dict]:
    return [
        row for row in smoke_result["rows"]
        if row["metadata"].get("family") == "unavailable"
    ]


def test_unavailable_rows_withhold_only_the_requested_tool(smoke_result) -> None:
    """Unavailable rows withhold exactly the requested op's tool, and the
    sibling tools v5b substituted to (light__HassLightSet,
    media_player__HassSetVolume) are offered as temptations unless withheld."""
    from generators.planning import CORE_TOOL_NAMES
    from generators.tools import namespaced_tool_name

    rows = _unavailable_rows(smoke_result)
    assert rows, "smoke run produced no unavailable rows"
    core = {namespaced_tool_name(name) for name in CORE_TOOL_NAMES}
    for row in rows:
        offered = _offered_tools(row)
        withheld = {
            namespaced_tool_name(tool)
            for tool in row["metadata"].get("unavailable_tools") or []
        }
        assert withheld, f"unavailable row has no withheld tool: {row['metadata']['candidate_id']}"
        assert not withheld & offered, f"withheld tool offered: {withheld & offered}"
        for sibling in ("light__HassLightSet", "media_player__HassSetVolume"):
            if sibling not in withheld:
                assert sibling in offered, (
                    f"substitution temptation {sibling} not offered: "
                    f"{row['metadata']['candidate_id']}"
                )
        assert core <= offered, f"row missing core tools {sorted(core - offered)}"


def test_non_refusal_rows_offer_the_requested_tool(smoke_result) -> None:
    """The refusal contrast only works while non-unavailable rows never withhold
    the requested operation's tool (decoy removals stay off it)."""
    from generators.coverage import expected_tool
    from generators.tools import namespaced_tool_name

    excluded_families = {"unavailable", "unsupported", "absence", "junk", "datetime", "area"}
    checked = 0
    violations = []
    for row in smoke_result["rows"]:
        meta = row["metadata"]
        family = meta.get("family")
        if family in excluded_families:
            continue
        capability = meta.get("capability")
        if capability in {"scripts", "timers"}:
            continue
        tool = expected_tool(capability, meta.get("operation") or "")
        if not tool:
            continue
        checked += 1
        if namespaced_tool_name(tool) not in _offered_tools(row):
            violations.append((
                meta.get("candidate_id"), family, capability,
                meta.get("operation"), tool,
            ))
    assert checked, "no non-refusal rows to check"
    assert not violations, f"rows withholding the requested tool: {violations}"


def test_audit_reports_by_withheld_tool(smoke_result) -> None:
    """The quality audit reports the withheld-tool distribution so
    substitution-prone pairs are covered."""
    report = smoke_result["stats"]["quality_audit"]
    assert "by_withheld_tool" in report
    assert report["by_withheld_tool"], (
        "smoke run has unavailable rows but the audit reports no withheld tools"
    )


# --- Fix 4: near-miss exclusion picks, scope metadata, and audit counts -----


def test_exclusion_rows_prefer_near_miss_same_area_siblings() -> None:
    """Same-area exclusion rows must exclude the in-area device that reads most
    like its neighbours: same area and domain as every target, never in gold,
    and the max name-token-overlap score among in-area entities."""
    from generators.scenarios.core import _near_miss_score, build_scenario

    checked = 0
    for index in range(64):
        scenario = build_scenario(
            index=index,
            seed=index * 13 + 7,
            capability="lights",
            operation="turn_off",
            home_size=64,
            targeting="multiple",
            robustness="exclusion",
            family="exclusion",
        )
        if not scenario.get("exclusion_scope"):
            continue
        home = scenario["home"]
        in_area = [
            e for e in home["entities"]
            if e["capability"] == "lights" and e["area"] == home["sayso_entity_area"]
        ]
        excluded = next(e for e in in_area if e["name"] in scenario["excluded_names"])
        targets = scenario["target_entities"]
        assert targets
        assert all(t["area"] == excluded["area"] and t["domain"] == excluded["domain"]
                   for t in targets), (index, excluded["name"], targets)
        gold_names = {call["arguments"].get("name") for call in scenario["expected"]["calls"]}
        assert excluded["name"] not in gold_names
        scores = {e["name"]: _near_miss_score(e, in_area) for e in in_area}
        assert scores[excluded["name"]] == max(scores.values()), (index, scores)
        checked += 1
    assert checked >= 10, f"only {checked} scenarios carried exclusion_scope"


def test_smoke_audit_counts_exclusion_scopes(smoke_result) -> None:
    """The audit splits exclusion rows into same-area scope vs named-list
    contrast, and every exclusion row lands in exactly one bucket."""
    audit = smoke_result["stats"]["quality_audit"]
    assert audit["exclusion_same_area"] >= 1
    assert audit["exclusion_same_area"] + audit["exclusion_named_list"] == audit["exclusion_rows"]
