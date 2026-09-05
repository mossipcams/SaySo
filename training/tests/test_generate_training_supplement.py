"""Contract tests for corrective SFT and shadow eval supplement generators."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_synthetic_dataset import _normalized, load_user_utterances  # noqa: E402
from evals.lfm_python_parse import parse_lfm_python_tool_call  # noqa: E402
from evals.recipe_lock import locked_specs, quality_eval_user_prompts  # noqa: E402
from generate_training_supplement import (  # noqa: E402
    CORRECTIVE_MAX,
    CORRECTIVE_MIN,
    CORRECTIVE_TARGETS,
    SHADOW_MAX,
    SHADOW_MIN,
    build_corrective_examples,
    build_corrective_specs,
    build_shadow_examples,
    build_shadow_specs,
    corrective_category_counts,
    validate_corrective_specs,
    validate_shadow_specs,
)

HELDOUT = ROOT / "datasets" / "sayso_test_balanced.jsonl"
BASE_TRAIN = ROOT / "datasets" / "sayso_train_first_10000.jsonl"


def _recipe_lock_sets() -> tuple[set[str], set[str]]:
    utterances = {_normalized(text) for text in quality_eval_user_prompts()}
    entities = {entity["name"] for spec in locked_specs() for entity in spec["home"]["entities"]}
    return utterances, entities


def test_corrective_row_count_within_bounds() -> None:
    specs = build_corrective_specs(seed=20260905)
    assert CORRECTIVE_MIN <= len(specs) <= CORRECTIVE_MAX
    assert len(specs) == sum(CORRECTIVE_TARGETS.values())


def test_corrective_category_weights() -> None:
    counts = corrective_category_counts(build_corrective_specs(seed=20260905))
    assert counts == CORRECTIVE_TARGETS


def test_shadow_row_count_within_bounds() -> None:
    specs = build_shadow_specs(seed=20260905, count=125)
    assert SHADOW_MIN <= len(specs) <= SHADOW_MAX


def test_corrective_light_fan_rows_use_matching_tool_family() -> None:
    specs = build_corrective_specs(seed=20260905)
    for spec in specs:
        if spec["category"] != "light_fan_contrast":
            continue
        call = spec["expected"]["calls"][0]
        entity = next(entity for entity in spec["home"]["entities"] if entity["name"] == spec["target_names"][0])
        if entity["kind"] == "light":
            assert call["name"] == "HassLightSet"
            assert "brightness" in call["arguments"]
        else:
            assert call["name"] == "HassFanSetSpeed"
            assert "percentage" in call["arguments"]


def test_corrective_lock_polarity_rows_use_turn_on_for_lock() -> None:
    specs = build_corrective_specs(seed=20260905)
    for spec in specs:
        if spec["category"] != "lock_polarity":
            continue
        call = spec["expected"]["calls"][0]
        if spec["subcategory"] == "lock_on_unlock_off":
            assert call["name"] == "HassTurnOn"
            assert call["arguments"]["device_class"] == ["door"]
        else:
            assert call["name"] == "HassTurnOff"
            assert call["arguments"]["device_class"] == ["door"]


def test_generators_avoid_recipe_lock_entity_and_utterance_overlap() -> None:
    recipe_lock_utterances, recipe_lock_entities = _recipe_lock_sets()
    corrective = build_corrective_examples(seed=20260905, heldout_path=HELDOUT, base_train_path=BASE_TRAIN)
    shadow = build_shadow_examples(seed=20260905, heldout_path=HELDOUT, base_train_path=BASE_TRAIN)
    for row in corrective + shadow:
        user = next(message["content"] for message in row["messages"] if message["role"] == "user")
        assert _normalized(user) not in recipe_lock_utterances
    for spec in build_corrective_specs(seed=20260905) + build_shadow_specs(seed=20260905):
        for entity in spec["home"]["entities"]:
            assert entity["name"] not in recipe_lock_entities


def test_validate_corrective_specs_rejects_recipe_lock_utterance_overlap() -> None:
    recipe_lock_utterances, recipe_lock_entities = _recipe_lock_sets()
    specs = build_corrective_specs(seed=20260905)
    bad = dict(specs[0])
    bad["utterance"] = next(iter(quality_eval_user_prompts()))
    try:
        validate_corrective_specs(
            [bad],
            recipe_lock_utterances=recipe_lock_utterances,
            recipe_lock_entities=recipe_lock_entities,
            heldout_utterances=set(),
            check_counts=False,
        )
    except ValueError as error:
        assert "recipe-lock" in str(error)
    else:
        raise AssertionError("expected recipe-lock overlap rejection")


def test_validate_corrective_specs_rejects_wrong_light_fan_tool() -> None:
    recipe_lock_utterances, recipe_lock_entities = _recipe_lock_sets()
    bad = next(spec for spec in build_corrective_specs(seed=20260905) if spec["category"] == "light_fan_contrast")
    bad = dict(bad)
    bad["expected"] = {
        "kind": "action",
        "calls": [
            {
                "name": "HassFanSetSpeed",
                "arguments": {"name": bad["target_names"][0], "domain": ["fan"], "percentage": 50},
            }
        ],
    }
    bad["utterance"] = f"set {bad['target_names'][0]} to 50 percent"
    try:
        validate_corrective_specs(
            [bad],
            recipe_lock_utterances=recipe_lock_utterances,
            recipe_lock_entities=recipe_lock_entities,
            heldout_utterances=set(),
            check_counts=False,
        )
    except ValueError as error:
        assert "HassLightSet" in str(error)
    else:
        raise AssertionError("expected wrong light/fan tool rejection")


def test_validate_corrective_specs_rejects_wrong_lock_polarity() -> None:
    recipe_lock_utterances, recipe_lock_entities = _recipe_lock_sets()
    bad = next(spec for spec in build_corrective_specs(seed=20260905) if spec["category"] == "lock_polarity")
    bad = dict(bad)
    bad["subcategory"] = "lock_on_unlock_off"
    bad["expected"] = {
        "kind": "action",
        "calls": [
            {
                "name": "HassTurnOff",
                "arguments": {"name": bad["target_names"][0], "device_class": ["door"]},
            }
        ],
    }
    bad["utterance"] = f"lock {bad['target_names'][0]}"
    try:
        validate_corrective_specs(
            [bad],
            recipe_lock_utterances=recipe_lock_utterances,
            recipe_lock_entities=recipe_lock_entities,
            heldout_utterances=set(),
            check_counts=False,
        )
    except ValueError as error:
        assert "HassTurnOn" in str(error)
    else:
        raise AssertionError("expected wrong lock polarity rejection")


def test_validate_shadow_specs_rejects_out_of_range_size() -> None:
    recipe_lock_utterances, recipe_lock_entities = _recipe_lock_sets()
    specs = build_shadow_specs(seed=20260905, count=125)
    try:
        validate_shadow_specs(
            specs[:50],
            recipe_lock_utterances=recipe_lock_utterances,
            recipe_lock_entities=recipe_lock_entities,
            heldout_utterances=set(),
        )
    except ValueError as error:
        assert "shadow row count" in str(error)
    else:
        raise AssertionError("expected shadow size rejection")


def test_validate_corrective_specs_rejects_out_of_range_size() -> None:
    recipe_lock_utterances, recipe_lock_entities = _recipe_lock_sets()
    specs = build_corrective_specs(seed=20260905)
    try:
        validate_corrective_specs(
            specs[:100],
            recipe_lock_utterances=recipe_lock_utterances,
            recipe_lock_entities=recipe_lock_entities,
            heldout_utterances=set(),
        )
    except ValueError as error:
        assert "corrective row count" in str(error)
    else:
        raise AssertionError("expected corrective size rejection")


def test_parser_does_not_truncate_apostrophe_entity_names_from_labels() -> None:
    for name in ("O'Malley's Study Blinds", "Kids' Room Light", "Office Main Light"):
        rendered = parse_lfm_python_tool_call(f"HassTurnOff(name='{name}', domain=['light'])")
        assert rendered["arguments"]["name"] == name


def test_rendered_rows_match_sayso_jsonl_contract() -> None:
    rows = build_corrective_examples(seed=20260905, heldout_path=HELDOUT, base_train_path=BASE_TRAIN)
    for row in rows:
        blob = json.dumps(row, ensure_ascii=False)
        assert "<tool_call>" not in blob
        assert "evals/cases/" not in blob
        for message in row["messages"]:
            for call in message.get("tool_calls") or []:
                assert call["type"] == "function"
                assert isinstance(call["function"]["arguments"], str)
                assert isinstance(json.loads(call["function"]["arguments"]), dict)


def test_shadow_rows_cover_recipe_concepts() -> None:
    specs = build_shadow_specs(seed=20260905, count=125)
    categories = {spec["category"] for spec in specs}
    assert "clean_direct" in categories
    assert "conversational" in categories
    assert "entity_identity" in categories
    assert "multi_action_exclusion" in categories
    assert "stt_corrupted" in categories
    assert "status" in categories
    assert "ambiguity" in categories
    assert "unsupported_no_action" in categories
    assert "light_fan_contrast" in categories
    assert "lock_polarity" in categories
    tools = {call["name"] for spec in specs for call in spec["expected"].get("calls") or []}
    assert "HassLightSet" in tools
    assert "HassFanSetSpeed" in tools
    lock_names = {
        call["name"]
        for spec in specs
        if spec["category"] == "lock_polarity"
        for call in spec["expected"]["calls"]
    }
    assert lock_names == {"HassTurnOn", "HassTurnOff"}


def test_generators_avoid_base_train_and_heldout_overlap() -> None:
    if not BASE_TRAIN.is_file() or not HELDOUT.is_file():
        return
    excluded = {_normalized(text) for text in load_user_utterances(BASE_TRAIN)}
    excluded.update(_normalized(text) for text in load_user_utterances(HELDOUT))
    rows = build_corrective_examples(seed=20260905, heldout_path=HELDOUT, base_train_path=BASE_TRAIN)
    rows.extend(build_shadow_examples(seed=20260905, heldout_path=HELDOUT, base_train_path=BASE_TRAIN))
    overlap = []
    for row in rows:
        user = next(message["content"] for message in row["messages"] if message["role"] == "user")
        if _normalized(user) in excluded:
            overlap.append(user)
    assert overlap == []
