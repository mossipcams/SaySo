"""Tests for generation pipeline."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from generators.config import GeneratorConfig
from generators.validation import check_quality_eval_overlap
from generators.pipeline import run_generation

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SMOKE = ROOT / "configs" / "generation" / "smoke.yaml"


def _cfg(count: int = 100, seed: int = 99) -> GeneratorConfig:
    config = GeneratorConfig.from_yaml(SMOKE, repo_root=REPO)
    config.count = count
    config.seed = seed
    config.area_distribution_path = None
    return config


def test_smoke_generation_small_n() -> None:
    result = run_generation(_cfg(count=100))
    rows = result["rows"]
    assert len(rows) == 100
    assert result["stats"]["accepted"] == 100
    for row in rows:
        assert row["tools"]
        assert row["messages"][0]["role"] == "system"
        assert any(m["role"] == "user" for m in row["messages"])


def test_generation_is_deterministic() -> None:
    first = run_generation(_cfg(count=80, seed=123))["rows"]
    second = run_generation(_cfg(count=80, seed=123))["rows"]
    assert first == second


_DIGEST_SCRIPT = """
import hashlib, json, sys
from pathlib import Path
ROOT = Path(sys.argv[1])
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
from generators.config import GeneratorConfig
from generators.pipeline import run_generation

cfg = GeneratorConfig.from_yaml(ROOT / "configs/generation/smoke.yaml", repo_root=REPO)
cfg.count = 60
cfg.seed = 7
cfg.area_distribution_path = None
rows = run_generation(cfg)["rows"]
print(hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest())
"""


def _digest_under_hash_seed(hash_seed: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", _DIGEST_SCRIPT, str(ROOT)],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": hash_seed},
    )
    return result.stdout.strip()


def test_generation_is_reproducible_across_processes() -> None:
    assert _digest_under_hash_seed("0") == _digest_under_hash_seed("1")


def test_rows_use_production_catalog() -> None:
    """Every row offers the production exposure catalog, not an answer-first subset."""
    rows = run_generation(_cfg(count=80, seed=5))["rows"]
    sizes: set[int] = set()
    for row in rows:
        offered = {tool["function"]["name"] for tool in row["tools"]}
        called = {
            call["function"]["name"]
            for message in row["messages"]
            for call in (message.get("tool_calls") or [])
        }
        assert called <= offered, sorted(called - offered)
        assert row["metadata"].get("catalog_source") == "production_exposure"
        sizes.add(len(offered))
    assert max(sizes) > 12


def test_scripts_are_their_own_tools_and_absent_from_the_entity_overview() -> None:
    from generators.context import serialize_context
    from generators.tools import script_tool_name, script_tools

    home = {
        "sayso_entity_area": "Kitchen",
        "entities": [
            {
                "entity_id": "script.good_morning",
                "name": "Good Morning",
                "aliases": ["Sunrise"],
                "domain": "script",
                "area": "Kitchen",
                "floor": "Ground",
                "state": "off",
            },
            {
                "entity_id": "light.kitchen_light",
                "name": "Kitchen Light",
                "aliases": [],
                "domain": "light",
                "area": "Kitchen",
                "floor": "Ground",
                "state": "off",
            },
        ],
    }
    assert script_tool_name(home["entities"][0]) == "good_morning"
    tools = script_tools(home)
    assert [tool["function"]["name"] for tool in tools] == ["good_morning"]
    context = serialize_context(home)
    assert "Good Morning" not in context
    assert "Kitchen Light" in context


def test_generated_script_rows_call_the_script_tool_not_hass_turn_on() -> None:
    rows = run_generation(_cfg(count=120, seed=17))["rows"]
    script_rows = [r for r in rows if r["metadata"].get("capability") == "scripts"]
    if not script_rows:
        return
    for row in script_rows:
        offered = {tool["function"]["name"] for tool in row["tools"]}
        for message in row["messages"]:
            for call in message.get("tool_calls") or []:
                name = call["function"]["name"]
                assert name != "HassTurnOn"
                assert name in offered


def test_every_quality_eval_prompt_is_rejected_verbatim() -> None:
    from evals.cases import excluded_train_utterances

    for prompt in excluded_train_utterances():
        assert check_quality_eval_overlap(prompt), prompt
        assert check_quality_eval_overlap(prompt.upper()), prompt
