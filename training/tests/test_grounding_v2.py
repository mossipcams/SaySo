"""Regression tests for the v2 grounding/discrimination corpus fixes.

These guard the two defects that shipped in the v1 40k corpus:

1. Grounding forcing was a one-shot latch, so the requested share was never
   delivered (3% requested, 0.22% achieved, reported as a pass).
2. No gate asserted the achieved share, so the shortfall was silent.

Plus the entity-discrimination category, which v1 did not have at all.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))
sys.path.insert(0, str(TRAINING_ROOT / "scripts"))

from generators.config import GeneratorConfig  # noqa: E402
from generators.grounding import (  # noqa: E402
    build_spec,
    required_training_variants,
    training_variants,
)
from generators.grounding import (  # noqa: E402
    grounding_capable_slot,
    grounding_pairs,
)
from generators.pipeline import run_generation  # noqa: E402
from generators.rates import enforce_rate_gate  # noqa: E402
from generators.scenarios.discrimination import (  # noqa: E402
    pick_discriminating_description,
    sibling_entities,
)
from generators.sampling import QuotaTracker  # noqa: E402
from generators.scenarios import build_scenario  # noqa: E402

COUNT = 1200


def _run(**overrides):
    return run_generation(
        GeneratorConfig(
            count=COUNT,
            seed=20260913,
            **overrides,
        )
    )


def test_grounding_reaches_its_ceiling_and_all_families():
    """v1 shipped 88 grounding rows out of 40k. v2 must deliver its ceiling."""
    report = _run(grounding_rate=0.028)["stats"]["grounding"]
    assert report["achieved_rate"] >= 0.020, report
    assert report["rows"] >= COUNT * 0.020, report
    # Every declared family must appear; the original bug left most unreachable.
    assert len(report["by_family"]) >= len(required_training_variants()), report


def test_grounding_catalogue_scales_to_a_full_size_corpus():
    """The v1 collapse was a fixed ceiling, not a scheduling bug.

    Every grounding variant is one fixed scenario (fixed home, fixed request), and
    ``DuplicateTracker`` accepts a scenario at most ``near_duplicate_limit``
    times. The catalogue size therefore caps grounding rows absolutely, whatever
    the corpus size: a 15-variant catalogue could never exceed ~120 rows, so 40k
    at 2.8% (1120 rows) shipped 136 and 2k at 2.8% (56 rows) looked healthy. The
    ceiling has to clear the largest request the defaults can make.
    """
    config = GeneratorConfig()
    ceiling = len(training_variants()) * config.near_duplicate_limit
    demanded = config.count * config.grounding_rate
    assert ceiling >= demanded, (
        f"grounding catalogue holds {len(training_variants())} scenarios, capping "
        f"grounding at {ceiling} rows, but the default config asks for {demanded:.0f}"
    )
    # Rows can only land on a slot of their own (capability, operation), so the
    # ceiling has to hold per pair, not just in total.
    per_pair = Counter((v["capability"], v["operation"]) for v in training_variants())
    for pair in grounding_pairs():
        assert per_pair[pair] * config.near_duplicate_limit >= demanded / len(grounding_pairs()), (
            f"{pair} has only {per_pair[pair]} variants"
        )


def test_every_generated_site_keeps_its_family_contrast():
    """The catalogue is mostly generated, so its contracts must be checked in bulk.

    `test_coverage_gates` asserts each family's contrast on hand-written
    arguments. 330 of the 345 scenarios are built by ``site_variants`` instead,
    and a family whose contrast quietly inverts is worse than a missing one: it
    ships confident wrong labels. This found the alias family targeting its own
    distractor.
    """
    by_prefix: dict[str, dict[str, tuple[dict, dict]]] = defaultdict(dict)
    for index, item in enumerate(training_variants()):
        prefix = item["phrasing_seed"]
        spec = build_spec(item, seed=20260913, index=index)
        by_prefix[prefix][item["family"][len(prefix) + 1:]] = (item, spec)

    def first_call(spec):
        calls = spec["expected"].get("calls") or []
        return calls[0] if calls else None

    def in_area(item, capability):
        area = item["home"]["sayso_entity_area"]
        return [e for e in item["home"]["entities"]
                if e["capability"] == capability and e["area"] == area]

    for prefix, members in sorted(by_prefix.items()):
        kind = prefix.split("_")[1]
        if kind == "media":
            assert set(members) == {"present", "renamed", "moved", "distractors"}, prefix
            # One request, four entity graphs: that is the whole point.
            assert len({spec["utterance"] for _, spec in members.values()}) == 1, prefix
            for label in ("present", "renamed", "distractors"):
                item, spec = members[label]
                players = in_area(item, "media_players")
                assert len(players) == 1, f"{prefix}_{label}"
                call = first_call(spec)
                assert spec["expected"]["kind"] == "action", f"{prefix}_{label}"
                assert call["arguments"]["name"] == players[0]["name"], f"{prefix}_{label}"
            moved = members["moved"][1]["expected"]
            assert moved["kind"] == "no_action" and moved["response"] == "area_unavailable", prefix
            assert not moved.get("calls"), prefix
        elif kind == "light":
            single, pair = members["single"][1], members["pair"][1]
            assert single["expected"]["kind"] == "action", prefix
            assert pair["expected"]["kind"] == "no_action", prefix
            assert pair["expected"]["response"] == "clarify", prefix
            assert single["utterance"] == pair["utterance"], prefix
        elif kind == "domain":
            item, present = members["present"]
            absent_item, absent = members["absent"]
            assert present["expected"]["kind"] == "action", prefix
            assert first_call(present)["arguments"]["name"] == in_area(item, "fans")[0]["name"], prefix
            assert absent["expected"]["kind"] == "no_action", prefix
            assert not any(e["capability"] == "fans" for e in absent_item["home"]["entities"]), prefix
        elif kind == "features":
            capable, incapable = members["capable"][1], members["incapable"][1]
            assert first_call(capable)["name"] == "HassSetVolume", prefix
            assert incapable["expected"]["kind"] == "no_action", prefix
            assert incapable["expected"]["response"] == "device_unsupported", prefix
            assert capable["utterance"] == incapable["utterance"], prefix
        elif kind == "alias":
            item, spec = members["alias"]
            aliased = [e for e in item["home"]["entities"] if len(e.get("aliases") or []) > 1]
            assert len(aliased) == 1, prefix
            # The request must use the alias and the label must resolve it back to
            # the canonical name -- not fire at the distractor sitting next to it.
            assert first_call(spec)["arguments"]["name"] == aliased[0]["name"], (
                f"{prefix}: alias row targets {first_call(spec)['arguments']['name']!r}, "
                f"not the aliased entity {aliased[0]['name']!r}"
            )
        elif kind == "named":
            # Issue #52: one request, a device whose name and area are stored
            # separately, and seven homes. Every contract here is one of the
            # issue's acceptance criteria.
            assert set(members) == {
                "bare", "room_named", "aliased", "speaker",
                "absent", "ambiguous", "irrelevant",
            }, prefix
            assert len({spec["utterance"] for _, spec in members.values()}) == 1, prefix

            def screen_named(item):
                """The eligible screen: a media player that is not the speaker."""
                return [e for e in item["home"]["entities"]
                        if e["capability"] == "media_players"
                        and e.get("device_class") == "tv"]

            for label in ("bare", "room_named", "aliased", "speaker", "irrelevant"):
                item, spec = members[label]
                screens = screen_named(item)
                assert len(screens) == 1, f"{prefix}_{label}"
                assert spec["expected"]["kind"] == "action", f"{prefix}_{label}"
                assert first_call(spec)["arguments"]["name"] == screens[0]["name"], (
                    f"{prefix}_{label}: targeted "
                    f"{first_call(spec)['arguments']['name']!r}, not {screens[0]['name']!r}"
                )
            # An irrelevant change must not move the answer.
            assert (
                first_call(members["bare"][1])["arguments"]
                == first_call(members["irrelevant"][1])["arguments"]
            ), prefix
            # A speaker in the room is never mistaken for the requested screen.
            speaker_item, speaker_spec = members["speaker"]
            speakers = [e for e in speaker_item["home"]["entities"]
                        if e.get("device_class") == "speaker"]
            assert speakers, prefix
            assert first_call(speaker_spec)["arguments"]["name"] != speakers[0]["name"], prefix
            # Real absence, and it must not claim the room has no media players
            # when it still has a speaker.
            absent = members["absent"][1]["expected"]
            assert absent["kind"] == "no_action", prefix
            assert absent["response"] == "device_absent", prefix
            assert not absent.get("calls"), prefix
            ambiguous = members["ambiguous"][1]["expected"]
            assert ambiguous["kind"] == "no_action", prefix
            assert ambiguous["response"] == "clarify", prefix
        else:
            raise AssertionError(f"{prefix}: unknown family kind {kind!r}")


def test_grounding_rows_come_from_grounding_capable_slots():
    """Forcing must place grounding rows where a variant actually exists.

    v1 called pick_variant on whatever slot came up, and only ~4% of slots can
    host a variant, which is why the requested share collapsed.
    """
    pairs = grounding_pairs()
    assert pairs, "grounding variants must declare (capability, operation) pairs"
    assert len(pairs) < 6, f"expected a small set of capable pairs, got {pairs}"

    quota = QuotaTracker(COUNT, 20260913, GeneratorConfig().tier_proportions)
    import random

    slot = grounding_capable_slot(quota, random.Random(1))
    assert slot is not None
    assert (slot["capability"], slot["operation"]) in pairs


def test_rate_gate_rejects_a_v1_style_shortfall():
    """The gate must fail loudly on the shortfall v1 shipped silently."""
    config = GeneratorConfig(count=COUNT)
    section = {
        "requested_rate": 0.028,
        "achieved_rate": 0.0022,  # what v1 actually delivered
        "rows": 88,
    }
    with pytest.raises(RuntimeError, match="grounding rate shortfall"):
        enforce_rate_gate(section, "grounding", config, COUNT, available_share=0.028)


def test_rate_gate_allows_normal_variance():
    """A run at the low end of measured variance must not fail the build."""
    config = GeneratorConfig(count=COUNT)
    section = {
        "requested_rate": 0.028,
        "achieved_rate": 0.024,  # low end of the measured 2.4-2.8% band
        "rows": int(COUNT * 0.024),
    }
    enforce_rate_gate(section, "grounding", config, COUNT, available_share=0.028)


def test_rate_gate_is_inert_for_small_runs_and_zero_requests():
    """A percentage is not assertable on a tiny run, and 0 means "not requested"."""
    config = GeneratorConfig(count=50)
    enforce_rate_gate(
        {"requested_rate": 0.5, "achieved_rate": 0.0, "rows": 0},
        "grounding",
        config,
        50,
    )
    config2 = GeneratorConfig(count=COUNT)
    enforce_rate_gate(
        {"requested_rate": 0.0, "achieved_rate": 0.0, "rows": 0},
        "grounding",
        config2,
        COUNT,
    )


def test_config_rejects_unrepresentable_rates():
    with pytest.raises(ValueError, match="grounding_rate"):
        GeneratorConfig(grounding_rate=1.5)
    with pytest.raises(ValueError, match="discrimination_rate"):
        GeneratorConfig(discrimination_rate=-0.1)
    with pytest.raises(ValueError, match="must be <= 1"):
        GeneratorConfig(grounding_rate=0.6, discrimination_rate=0.6)


def test_discrimination_rows_are_produced_and_describe_rather_than_name():
    """v1 had no description-based rows at all; v2 must produce them."""
    report = _run(grounding_rate=0.0, discrimination_rate=0.35)
    disc = report["stats"]["discrimination"]
    assert disc["rows"] > 0, (
        "no description-based rows were produced; the discrimination category "
        "is not wired into the pipeline"
    )

    described = [
        row for row in report["rows"] if row["metadata"].get("discrimination")
    ]
    assert described, "no row carries the discrimination flag"
    for row in described:
        utterance = row["messages"][1]["content"].lower()
        targets = row["metadata"].get("expected_target_names") or []
        for name in targets:
            assert name.lower() not in utterance, (
                f"discrimination row names its target {name!r}: {utterance!r}"
            )


def test_discriminating_description_is_unique_and_leaks_no_name():
    """A description must not be satisfiable by a sibling, else the row is noise."""
    produced = 0
    for index in range(120):
        scenario = build_scenario(
            index=index,
            seed=1234,
            capability="lights",
            operation="turn_on",
            home_size=32,
            targeting="individual",
            robustness="ordinary",
            attempt=index,
        )
        import random

        text = pick_discriminating_description(scenario, random.Random(index))
        if not text:
            continue
        produced += 1
        target = scenario.get("target_entity") or {}
        siblings = sibling_entities(scenario)
        lowered = text.lower()
        # The description must not contain any listed entity name.
        for entity in [target, *siblings]:
            for name in [entity.get("name"), *(entity.get("aliases") or ())]:
                if name:
                    assert name.lower() not in lowered, (
                        f"description {text!r} leaks entity name {name!r}"
                    )
    assert produced > 0, "no descriptions were produced for lights/turn_on"


def test_discrimination_refuses_when_no_sibling_exists():
    """With nothing to discriminate between, the row would teach nothing."""
    scenario = {
        "target_entity": {
            "entity_id": "light.only",
            "name": "Only Light",
            "aliases": ["Only Light"],
            "domain": "light",
            "area": "Closet",
        },
        "home": {
            "entities": [
                {
                    "entity_id": "light.only",
                    "name": "Only Light",
                    "aliases": ["Only Light"],
                    "domain": "light",
                    "area": "Closet",
                }
            ]
        },
    }
    import random

    assert pick_discriminating_description(scenario, random.Random(0)) is None


def test_generation_is_deterministic_for_v2_config():
    """Same seed must produce the same corpus, or a run is not reproducible."""
    first = _run(grounding_rate=0.028, discrimination_rate=0.35)
    second = _run(grounding_rate=0.028, discrimination_rate=0.35)
    assert first["stats"]["grounding"] == second["stats"]["grounding"]
    assert first["stats"]["discrimination"] == second["stats"]["discrimination"]
    assert [r["messages"][1]["content"] for r in first["rows"]] == [
        r["messages"][1]["content"] for r in second["rows"]
    ]


# --------------------------------------------------------------------------- #
# Issue #52: the production tool contract and the held-out TV cases.
# --------------------------------------------------------------------------- #


def test_no_training_wording_reproduces_a_held_out_eval_prompt():
    """The grounding catalogue must not say an eval prompt out loud.

    The named-device families say the device word with no area attached
    ("Turn on the tv"), so a site that draws "TV" reproduces the frozen issue #52
    eval prompt verbatim. Caught live: one site did.
    """
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    repo = root.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from evals.cases import cases_with_tag

    held_out = {
        case.utterance.strip().casefold().rstrip(".")
        for case in cases_with_tag("grounding")
    }
    spoken = {
        item["utterance"].strip().casefold().rstrip(".")
        for item in training_variants()
        if item.get("utterance")
    }
    assert not (spoken & held_out), f"training says held-out prompts: {sorted(spoken & held_out)}"


def test_namespaced_rows_keep_prompt_schema_and_label_in_one_contract():
    """A row renders in exactly one tool contract, everywhere it names a tool.

    Home Assistant 2026.9 sends `intent__HassTurnOn` in both the schema and the
    system prompt. A row that offers the namespaced schema but labels the bare
    name teaches the model to emit a tool that is not on offer.
    """
    from generators.labels import render_example
    from generators.tools import namespaced_tool_name

    variant = next(v for v in training_variants()
                   if v["phrasing_seed"].split("_")[1] == "named")
    spec = build_spec(variant, seed=20260914, index=0)
    spec["namespaced_tools"] = True
    spec["full_tool_catalog"] = True
    row = render_example(spec)

    offered = {tool["function"]["name"] for tool in row["tools"]}
    assert "intent__HassTurnOn" in offered or "intent__HassTurnOff" in offered
    for message in row["messages"]:
        for call in message.get("tool_calls") or []:
            assert call["function"]["name"] in offered, call["function"]["name"]
    prompt = row["messages"][0]["content"]
    assert namespaced_tool_name("GetLiveContext") in prompt
    assert "`GetLiveContext`" not in prompt
    # Production sends ~23 tools; the full catalogue must not be a sampled subset.
    assert len(row["tools"]) > 20


def test_namespaced_rows_still_count_towards_positive_coverage():
    """`intent__HassTurnOn` and `HassTurnOn` are one tool, not two.

    Comparing raw names made every namespaced row fail its positive quota, so the
    namespaced share silently shipped at a third of the requested rate.
    """
    from generators.coverage import classify_row
    from generators.labels import render_example

    variant = next(v for v in training_variants()
                   if v["phrasing_seed"].split("_")[1] == "named"
                   and v["family"].endswith("_bare"))
    spec = build_spec(variant, seed=20260914, index=0)
    bare = classify_row(render_example({**spec, "namespaced_tools": False}))
    named = classify_row(render_example({**spec, "namespaced_tools": True}))
    assert bare["positive"], "the bare-contract row is not positive; fixture is wrong"
    assert named["positive"], "namespaced row lost its positive coverage"
    assert bare["operation"] == named["operation"]
