"""Tests for Home-LLM → SaySo SFT generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sayso_server.control_plan import ControlPlan
from sayso_server.models import ENTITY_ID_PATTERN
from train.generator import convert_home_llm_jsonl, row_to_sft_example
from train.home_llm import parse_home_llm_row
from train.mapper import map_row_to_control_plan

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_maps_light_turn_on_home_llm_row_to_control_plan() -> None:
    row = parse_home_llm_row(_load_fixture("light_turn_on.json"))
    assert row is not None

    plan = map_row_to_control_plan(row)
    assert plan is not None
    validated = ControlPlan.model_validate(plan)
    assert validated.outcome == "action"
    assert validated.domain == "light"
    assert validated.state == "on"
    assert validated.targets == ["floor lamp"]

    example = row_to_sft_example(row)
    assert example is not None
    assert example.prompt.startswith("Reply with only one ControlPlan JSON object.")
    assert "turn on the floor lamp" in example.prompt
    assert "Floor Lamp" in example.prompt
    assert ENTITY_ID_PATTERN.search(example.prompt) is None

    completion = json.loads(example.completion)
    assert ControlPlan.model_validate(completion) == ControlPlan.model_validate(plan)


def test_drops_unsupported_timer_service_row() -> None:
    row = parse_home_llm_row(_load_fixture("timer_start.json"))
    assert row is not None
    assert map_row_to_control_plan(row) is None
    assert row_to_sft_example(row) is None


def test_does_not_emit_entity_ids_as_targets() -> None:
    row = parse_home_llm_row(_load_fixture("light_turn_on.json"))
    assert row is not None

    example = row_to_sft_example(row)
    assert example is not None

    completion = json.loads(example.completion)
    for field in ("targets", "include", "exclude"):
        for value in completion.get(field, []):
            assert ENTITY_ID_PATTERN.match(value) is None

    assert "light.floor_lamp" not in example.completion


def test_cli_converts_fixture_jsonl(tmp_path: Path) -> None:
    input_path = FIXTURES / "sample.jsonl"
    output_path = tmp_path / "sayso_train.jsonl"

    stats = convert_home_llm_jsonl(input_path, output_path)
    assert stats.input_rows == 2
    assert stats.kept_rows == 1
    assert stats.dropped_rows == 1

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert "prompt" in record
    assert "completion" in record
    ControlPlan.model_validate(json.loads(record["completion"]))


def test_rejects_eval_dataset_output_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="evals/datasets"):
        convert_home_llm_jsonl(
            FIXTURES / "sample.jsonl",
            tmp_path / "evals/datasets/train.jsonl",
        )
