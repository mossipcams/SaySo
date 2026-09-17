"""Contract checks for the balanced synthetic test-data composer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from generators.config import GeneratorConfig
from generators.pipeline import run_generation

from generate_balanced_test_data import DEFAULT_COUNT


def _user_text(example: dict) -> str:
    content = next(
        message["content"]
        for message in example["messages"]
        if message["role"] == "user"
    )
    if isinstance(content, str):
        return content
    return content[0]["text"]


def _build_test_set(count: int, seed: int) -> list[dict]:
    config = GeneratorConfig.from_yaml(
        ROOT / "configs" / "generation" / "production.yaml",
        repo_root=REPO,
    )
    config.count = count
    config.seed = seed
    config.split = "test"
    return run_generation(config)["rows"]


def test_balanced_test_set_has_canonical_shape_and_is_deterministic() -> None:
    assert DEFAULT_COUNT == 2_500

    first = _build_test_set(count=100, seed=1042)
    second = _build_test_set(count=100, seed=1042)

    assert first == second
    assert len(first) == 100
    assert len({_user_text(example) for example in first}) == len(first)
    rendered_prompts = "\n".join(_user_text(example).casefold() for example in first)
    assert "you to is " not in rendered_prompts
    assert "you to are " not in rendered_prompts
    assert "can you let's " not in rendered_prompts

    for example in first:
        assert "evals/cases/" not in json.dumps(example)
        for message in example["messages"]:
            for call in message.get("tool_calls") or []:
                arguments = call["function"]["arguments"]
                assert isinstance(arguments, str)
                assert isinstance(json.loads(arguments), dict)
