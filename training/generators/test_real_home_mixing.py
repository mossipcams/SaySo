"""Real-home mixing: what the manifest claims must be what the corpus contains.

Home Assistant is authoritative for exposure, names, aliases, areas and supported
features, so these checks run against a snapshot in the exporter's own shape. The
fixture is `synthetic_reference_home.json` -- clearly synthetic, and used because
this repository has no credentials for the live instance. A live refresh
(`scripts/fetch_ha_home.py`, exposure_source=assist_exposure) remains a
prerequisite for the home-specific recipe; `require_exposure_source` enforces it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from generators.capability_registry import entity_supports
from generators.config import GeneratorConfig
from generators.coverage import classify_row, expected_tool
from generators.pipeline import run_generation
from generators.real_home import (
    LIVE_EXPOSURE_SOURCES,
    TESTABLE_EXPOSURE_SOURCES,
    exposure_source,
    holdout_entity_names,
    load_real_home,
    require_exposure_source,
)

FIXTURE = TRAINING_ROOT / "fixtures" / "synthetic_reference_home.json"
LIVE_SNAPSHOT = TRAINING_ROOT / "fixtures" / "real_home.json"


_RUNS: dict[tuple, dict] = {}


def _mixed_run(**overrides):
    """Deterministic, so one run per distinct configuration is enough."""
    key = tuple(sorted(overrides.items()))
    if key not in _RUNS:
        _RUNS[key] = run_generation(
            GeneratorConfig(
                count=800,
                seed=515,
                real_home_path=FIXTURE,
                real_home_rate=0.25,
                paraphrase_enabled=False,
                **overrides,
            )
        )
    return _RUNS[key]


def test_the_reference_snapshot_carries_the_living_room_tv():
    home = json.loads(FIXTURE.read_text())
    tv = next(e for e in home["entities"] if e["entity_id"] == "media_player.living_room_tv")
    assert tv["name"] == "TV"
    assert tv["area"] == "Living Room"
    assert "the telly" in tv["aliases"]
    # Not every media player is the same device.
    speaker = next(e for e in home["entities"] if e["entity_id"] == "media_player.kitchen_speaker")
    assert not entity_supports(speaker, "media_players", "turn_on")
    assert entity_supports(speaker, "media_players", "volume_set")
    assert entity_supports(tv, "media_players", "turn_on")


def test_the_tv_survives_the_train_holdout_split():
    train = load_real_home(FIXTURE, split="train")
    assert any(e["entity_id"] == "media_player.living_room_tv" for e in train["entities"])
    assert "TV" not in holdout_entity_names(FIXTURE)


def test_mixing_produces_positive_tv_rows_for_its_supported_operations():
    rows = _mixed_run()["rows"]
    operations = set()
    for row in rows:
        facets = classify_row(row)
        if not facets["positive"]:
            continue
        wanted = expected_tool(facets["capability"], facets["operation"])
        for message in row["messages"]:
            for call in message.get("tool_calls") or []:
                if (
                    call["function"]["name"] == wanted
                    and json.loads(call["function"]["arguments"]).get("name") == "TV"
                ):
                    operations.add((facets["operation"], call["function"]["name"]))
    assert operations >= {
        ("turn_on", "HassTurnOn"),
        ("turn_off", "HassTurnOff"),
        ("pause", "HassMediaPause"),
        ("volume_set", "HassSetVolume"),
    }, operations


def test_the_manifest_records_requested_and_achieved_mixing():
    report = _mixed_run()["stats"]
    real = report["real_home"]
    assert real["requested_rate"] == 0.25
    assert real["achieved_rate"] == 0.25
    assert real["rows"] == 200
    assert real["rows"] == sum(
        1 for row in _mixed_run()["rows"] if row["metadata"]["real_home"]
    )
    assert real["target_counts"]


def _named_targets(row):
    return [
        name
        for message in row["messages"]
        for call in message.get("tool_calls") or []
        if (name := json.loads(call["function"]["arguments"]).get("name"))
    ]


def test_a_multi_target_row_is_one_real_home_row():
    result = _mixed_run()
    real_rows = [row for row in result["rows"] if row["metadata"]["real_home"]]
    assert any(len(_named_targets(row)) > 1 for row in real_rows), "no multi-target real rows"
    # One row is one row, however many entities it names.
    assert result["stats"]["real_home"]["rows"] == len(real_rows)
    assert result["stats"]["real_home"]["rows"] < sum(
        max(len(_named_targets(row)), 1) for row in real_rows
    )


def test_entity_counts_only_include_real_entities():
    result = _mixed_run()
    known = {e["name"] for e in load_real_home(FIXTURE, split="train")["entities"]}
    assert set(result["stats"]["real_home"]["target_counts"]) <= known


def test_held_out_entities_never_become_training_targets():
    result = _mixed_run()
    assert not holdout_entity_names(FIXTURE) & set(result["stats"]["real_home"]["target_counts"])


def test_an_explicit_cap_binds():
    result = _mixed_run(real_home_entity_cap=3)
    counts = result["stats"]["real_home"]["target_counts"]
    assert counts and max(counts.values()) <= 3
    assert result["stats"]["accepted"] == 800  # capping redistributes, never shrinks


def test_synthetic_only_is_an_explicit_override():
    config = GeneratorConfig(
        count=200, seed=7, real_home_path=FIXTURE, real_home_rate=0.25, synthetic_only=True
    )
    assert config.real_home_path is None and config.real_home_rate == 0.0
    assert "real_home" not in run_generation(config)["stats"]


def test_live_snapshot_requires_assist_exposure_and_stale_exports_fail(tmp_path):
    assert exposure_source(LIVE_SNAPSHOT) == "assist_exposure"
    require_exposure_source(LIVE_SNAPSHOT)
    stale = tmp_path / "stale_home.json"
    stale.write_text('{"exposure_source":"domain_filter"}')
    with pytest.raises(ValueError, match="exposure_source"):
        require_exposure_source(stale)
    with pytest.raises(ValueError, match="exposure_source"):
        require_exposure_source(stale, allowed=TESTABLE_EXPOSURE_SOURCES)
    # The synthetic fixture is testable but still not a live input.
    require_exposure_source(FIXTURE, allowed=TESTABLE_EXPOSURE_SOURCES)
    with pytest.raises(ValueError, match="exposure_source"):
        require_exposure_source(FIXTURE, allowed=LIVE_EXPOSURE_SOURCES)
