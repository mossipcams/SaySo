"""Invariants for the v4 defect-driven corpus (issues #39, #94, #97).

Each test here fails if one of the three data defects the v4 recipe exists to
fix comes back.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from generators.config import GeneratorConfig
from generators.deduplication import DuplicateTracker
from generators.homes import generate_home, make_entity
from generators.pipeline import _run_generation, run_generation
from generators.planning import REAL_HOME_EXCLUDED_FAMILIES, build_plan, verify_feasible
from generators.row_generation import generate_row
from generators.stt_noise import apply_stt_noise
from generators.utterances import expand_utterance
from generators.validation import (
    _MIN_CHARS_PER_TOKEN,
    count_row_tokens,
    row_characters,
    validate_row,
)

TOKENIZER = "LiquidAI/LFM2.5-230M-Base"
REAL_HOME = REPO / "training" / "fixtures" / "real_home.json"
V4_RECIPE = REPO / "training" / "configs" / "generation" / "full_sft_v4.yaml"


def _row(family: str, seed: int = 20260922, **config_kwargs):
    """Generate one accepted row for ``family``, or None if every attempt fails."""
    config = GeneratorConfig(
        count=1,
        seed=seed,
        token_budget=8192,
        tokenizer_model=TOKENIZER,
        **config_kwargs,
    )
    rng = random.Random(seed)
    tracker = DuplicateTracker(near_limit=8)
    for attempt in range(60):
        slot = {
            "index": attempt,
            "family": family,
            "robustness": None,
            "capability": "lights",
            "operation": "turn_on",
            "tier": 1,
            "home_size": 16,
            "area_scenario": None,
        }
        row, _ = generate_row(
            slot, config, rng, excluded=set(), dup_tracker=tracker, attempt=attempt
        )
        if row is not None:
            return row
    return None


def test_count_row_tokens_counts_a_row_with_tool_calls():
    """Canonical rows keep ``arguments`` as a JSON string.

    The LFM2 chat template calls ``.items()`` on it, so templating raised and
    every caller silently fell back to a length/4 estimate; separately,
    transformers >= 5 returns a BatchEncoding whose ``len()`` is 2. Both made
    the token budget unenforceable, and all three shipped 40k manifests report
    ``{min: 0, p50: 0, max: 0}``.
    """
    row = _row("ordinary")
    assert row is not None, "no ordinary row generated"
    assert any(m.get("tool_calls") for m in row["messages"]), "need a tool-call row"

    tokens = count_row_tokens(row, model_name=TOKENIZER)

    # A production-shaped row carries a ~30-tool catalog and a full static
    # context: thousands of tokens, never 2 and never a char/4 estimate.
    assert tokens > 1000, f"token count collapsed to {tokens}"
    assert not row["metadata"].get("_token_length_estimated", False)


def test_cheap_token_bound_never_understates_the_real_count():
    """The budget gate skips the tokenizer using a character bound.

    Tokenizing is ~9 ms and generation validates hundreds of thousands of
    candidates, so the bound is what keeps a 40k build from taking two hours.
    It is only safe if it never reads low: a row it clears must genuinely fit.
    """
    # Families the lights/turn_on slot below can actually produce, spanning the
    # short (junk) and long (multi_action, exclusion) ends of the distribution.
    for family in ("ordinary", "settings", "multi_action", "exclusion", "junk"):
        row = _row(family)
        assert row is not None, f"no {family} row generated"
        bound = row_characters(row) / _MIN_CHARS_PER_TOKEN
        actual = count_row_tokens(row, model_name=TOKENIZER)
        assert bound >= actual, (
            f"{family}: bound {bound:.0f} understates {actual} real tokens"
        )


def test_junk_transcripts_answer_without_calling_a_tool():
    """Issue #97: garbled STT must not produce tool calls."""
    row = _row("junk")
    assert row is not None, "no junk row generated"

    assistant = [m for m in row["messages"] if m["role"] == "assistant"]
    assert assistant, "junk row has no assistant turn"
    assert not any(m.get("tool_calls") for m in row["messages"]), (
        "junk transcript produced a tool call"
    )
    assert assistant[-1]["content"] == "Sorry, I didn't catch that."
    assert row["tools"], "junk row must still offer the production catalog"


def test_bare_name_rate_produces_entities_without_their_area_prefix():
    """Issue #94: a name like ``TV`` in area ``Living Room`` must exist.

    Every synthetic name used to be ``f"{area} {role}"``, which taught the model
    to answer "the living room TV" with ``name="Living room TV"`` -- a string no
    Home Assistant registry holds, so the intent matcher raises MatchFailedError.
    """
    rng = random.Random(7)
    prefixed = bare = 0
    bare_with_area_alias = 0
    for index in range(12):
        home = generate_home(index, 32, rng, bare_name_rate=0.35)
        for entity in home["entities"]:
            area = entity["area"]
            if entity["name"].casefold().startswith(area.casefold()):
                prefixed += 1
                continue
            bare += 1
            if any(
                a.casefold().startswith(area.casefold())
                for a in entity.get("aliases") or []
            ):
                bare_with_area_alias += 1

    assert bare > 0 and prefixed > 0, (
        f"expected a mix, got bare={bare} prefixed={prefixed}"
    )
    assert 0.15 < bare / (bare + prefixed) < 0.60, (
        "bare-name share outside the useful band"
    )
    # A bare entity must not be handed the area-prefixed alias back: that would
    # put the concatenated string in the context and teach the defect again.
    assert bare_with_area_alias == 0, (
        f"{bare_with_area_alias} bare entities still carry an area-prefixed alias"
    )


def test_bare_tv_room_phrasing_keeps_canonical_gold_name():
    """Issue #94: room-qualified speech must not change ``name`` in tool args."""
    tv = make_entity(
        name="TV",
        capability="media_players",
        area="Living Room",
        floor="Main Floor",
        rng=random.Random(0),
    )
    spec = {
        "candidate_id": "tv94",
        "semantic_id": "tv94",
        "seed": 94,
        "category": "ordinary",
        "home": {
            "sayso_entity_area": "Living Room",
            "entities": [tv],
            "areas": ["Living Room"],
            "area_floors": {"Living Room": "Main Floor"},
        },
        "expected": {
            "kind": "action",
            "calls": [
                {
                    "name": "HassTurnOn",
                    "arguments": {"name": "TV", "domain": ["media_player"]},
                }
            ],
        },
        "target_names": ["TV"],
        "capability": "media_players",
        "operation": "turn_on",
    }
    utterance = expand_utterance(spec)
    assert "living room" in utterance.casefold()
    assert "tv" in utterance.casefold() or "t v" in utterance.casefold()
    assert spec["expected"]["calls"][0]["arguments"]["name"] == "TV"
    spec["utterance"] = utterance
    assert validate_row(spec) is None


def test_stt_tv_corruption_passes_validate_row():
    """Issue #102: letter-spaced TV in the utterance still validates against gold ``TV``."""
    tv = make_entity(
        name="TV",
        capability="media_players",
        area="Living Room",
        floor="Main Floor",
        rng=random.Random(1),
    )
    spec = {
        "candidate_id": "stt_tv_102",
        "home": {"entities": [tv], "areas": ["Living Room"], "area_floors": {"Living Room": "Main Floor"}},
        "expected": {
            "kind": "action",
            "calls": [
                {
                    "name": "HassTurnOn",
                    "arguments": {"name": "TV", "domain": ["media_player"]},
                }
            ],
        },
        "target_names": ["TV"],
        "capability": "media_players",
        "operation": "turn_on",
    }
    tv_kinds = {"letter_spaced_tv", "punctuated_tv", "phonetic_tv"}
    for source in ("Turn on the living room TV", "turn on the living room tv"):
        corrupted = ""
        kind: str | None = None
        for seed in range(200):
            corrupted, kind = apply_stt_noise(
                source,
                random.Random(seed),
                target_names=["TV"],
                force_transform=True,
            )
            if kind in tv_kinds:
                break
        assert kind in tv_kinds, f"no TV STT kind for {source!r} within 200 seeds (last={kind})"
        spec["utterance"] = corrupted
        assert validate_row(spec) is None


def test_bare_name_rate_zero_keeps_the_old_naming():
    """The default must not silently change corpora built from older recipes."""
    rng = random.Random(7)
    home = generate_home(0, 32, rng)
    named_by_area = [
        entity
        for entity in home["entities"]
        if entity["name"].casefold().startswith(entity["area"].casefold())
    ]
    assert len(named_by_area) > len(home["entities"]) // 2


def test_junk_rows_are_not_counted_against_capability_quotas():
    """A junk row teaches no capability, so it must not consume one's quota."""
    row = _row("junk")
    assert row is not None
    assert row["metadata"]["family"] == "junk"
    assert row["metadata"]["no_action_reason"] == "not_understood"


def _v4_config(**overrides) -> GeneratorConfig:
    cfg = GeneratorConfig.from_yaml(V4_RECIPE, repo_root=REPO)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_v4_plan_is_feasible_with_real_home_mixing() -> None:
    cfg = _v4_config()
    plan = build_plan(
        cfg.count,
        cfg.seed,
        cfg.allocations,
        area_distribution_path=cfg.area_distribution_path,
        near_duplicate_limit=cfg.near_duplicate_limit,
        datetime_required=cfg.get_datetime_positive_min,
        real_home_path=cfg.real_home_path,
        real_home_rate=cfg.real_home_rate,
    )
    assert plan.requested["unavailable"] > 0
    assert plan.requested["unsupported"] > 0


def test_verify_feasible_rejects_refusal_over_real_home_identity_without_exclusion() -> None:
    with pytest.raises(ValueError, match="real home supports at most"):
        verify_feasible(
            4000,
            {"ordinary": 0.5, "unavailable": 0.5},
            near_duplicate_limit=8,
            area_required={},
            grounding_variants=0,
            grounding_rate=0.0,
            real_home_path=REAL_HOME,
            real_home_rate=0.10,
            refusal_on_synthetic_only=False,
        )


def test_refusal_families_never_use_real_home_rows() -> None:
    cfg = _v4_config(
        count=240,
        seed=20260922,
        get_datetime_positive_min=0,
        area_distribution_path=None,
        output_path=ROOT / "datasets" / "_pytest_refusal_real_home.jsonl",
    )
    result = run_generation(cfg)
    assert len(result["rows"]) == cfg.count
    for row in result["rows"]:
        family = row["metadata"].get("family")
        if family in REAL_HOME_EXCLUDED_FAMILIES:
            assert not row["metadata"].get("real_home"), (
                f"{family} row incorrectly marked real_home"
            )


def test_ordinary_only_run_fills_with_balanced_real_home_mix() -> None:
    """Late-slot real-home pressure must not strand ordinary family quota."""
    base = _v4_config()
    plan = build_plan(
        base.count,
        base.seed,
        base.allocations,
        area_distribution_path=base.area_distribution_path,
        near_duplicate_limit=base.near_duplicate_limit,
        datetime_required=base.get_datetime_positive_min,
        real_home_path=base.real_home_path,
        real_home_rate=base.real_home_rate,
    )
    count = min(plan.requested["ordinary"], 320)
    cfg = _v4_config(
        count=count,
        seed=20260922,
        allocations={"ordinary": 1.0},
        get_datetime_positive_min=0,
        area_distribution_path=None,
        discrimination_rate=0.0,
        min_positive_per_operation=0,
        min_positive_per_tool=0,
        output_path=ROOT / "datasets" / "_pytest_ordinary_real_home.jsonl",
    )
    result = _run_generation(cfg)
    assert len(result["rows"]) == count
    assert all(row["metadata"]["family"] == "ordinary" for row in result["rows"])


def test_40k_shaped_refusal_leftover_fills_under_attempt_budget() -> None:
    """Unavailable + unsupported counts from the v4 recipe, isolated from easy families."""
    base = _v4_config()
    plan = build_plan(
        base.count,
        base.seed,
        base.allocations,
        area_distribution_path=base.area_distribution_path,
        near_duplicate_limit=base.near_duplicate_limit,
        datetime_required=base.get_datetime_positive_min,
    )
    count = plan.requested["unavailable"] + plan.requested["unsupported"]
    # Keep the v4 unavailable/unsupported ratio but stay under audit's 1k-row
    # diversity floor so this test only exercises allocation + attempt budget.
    probe_count = min(count, 800)
    scale = probe_count / count
    probe_unavailable = max(1, int(round(plan.requested["unavailable"] * scale)))
    probe_unsupported = max(1, probe_count - probe_unavailable)
    probe_count = probe_unavailable + probe_unsupported
    cfg = _v4_config(
        count=probe_count,
        seed=20260922,
        allocations={
            "unavailable": probe_unavailable / probe_count,
            "unsupported": probe_unsupported / probe_count,
        },
        get_datetime_positive_min=0,
        area_distribution_path=None,
        real_home_rate=0.0,
        discrimination_rate=0.0,
        min_positive_per_operation=0,
        min_positive_per_tool=0,
        output_path=ROOT / "datasets" / "_pytest_refusal_leftover.jsonl",
    )
    result = _run_generation(cfg)
    assert len(result["rows"]) == probe_count
    attempts = result["stats"]["attempts"]
    assert attempts < probe_count * cfg.max_attempts_multiplier, (
        f"refusal-only generation exhausted attempts ({attempts} >= "
        f"{probe_count * cfg.max_attempts_multiplier})"
    )
    families = {row["metadata"]["family"] for row in result["rows"]}
    assert families <= {"unavailable", "unsupported"}
    assert not any(row["metadata"].get("real_home") for row in result["rows"])


if __name__ == "__main__":  # pragma: no cover - manual run
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print(json.dumps({"checks": "passed"}))
