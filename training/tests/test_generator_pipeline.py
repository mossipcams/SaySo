"""Tests for generation pipeline."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from generators.config import GeneratorConfig
from generators.pipeline import run_generation

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_generation_small_n() -> None:
    config = GeneratorConfig(count=100, seed=99, paraphrase_enabled=False)
    result = run_generation(config)
    rows = result["rows"]
    assert len(rows) == 100
    assert result["stats"]["accepted"] == 100
    for row in rows:
        assert row["tools"]
        assert row["messages"][0]["role"] == "system"
        assert any(m["role"] == "user" for m in row["messages"])


def test_generation_is_deterministic() -> None:
    cfg = GeneratorConfig(count=50, seed=123)
    first = run_generation(cfg)["rows"]
    second = run_generation(cfg)["rows"]
    assert first == second


_DIGEST_SCRIPT = """
import hashlib, json, sys
sys.path.insert(0, sys.argv[1])
from generators.config import GeneratorConfig
from generators.pipeline import run_generation

rows = run_generation(GeneratorConfig(count=60, seed=7, paraphrase_enabled=False))["rows"]
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
    """Same seed must yield the same dataset regardless of str-hash randomization."""
    assert _digest_under_hash_seed("0") == _digest_under_hash_seed("1")


def test_offered_tools_are_compact_and_always_contain_the_called_tools() -> None:
    """A row must offer every tool its label calls, but not the whole 27-tool catalog."""
    rows = run_generation(GeneratorConfig(count=300, seed=5, paraphrase_enabled=False))["rows"]
    sizes: set[int] = set()
    tool_sets: set[tuple[str, ...]] = set()
    for row in rows:
        offered = {tool["function"]["name"] for tool in row["tools"]}
        called = {
            call["function"]["name"]
            for message in row["messages"]
            for call in (message.get("tool_calls") or [])
        }
        assert called <= offered, sorted(called - offered)
        sizes.add(len(offered))
        tool_sets.add(tuple(sorted(offered)))
    assert max(sizes) <= 12, sizes
    # distractors must vary, or the model never learns to select
    assert len(tool_sets) > 10, len(tool_sets)


def test_every_quality_eval_prompt_is_rejected_verbatim() -> None:
    """Punctuation must not defeat the guard: "joe's" and "joe s" both contaminate."""
    from evals.recipe_lock import quality_eval_user_prompts
    from evals.v3_quality import gold_user_prompts, shadow_user_prompts
    from generators.pipeline import _check_quality_eval_overlap

    for prompt in (*quality_eval_user_prompts(), *gold_user_prompts(), *shadow_user_prompts()):
        assert _check_quality_eval_overlap(prompt), prompt
        assert _check_quality_eval_overlap(prompt.upper()), prompt
