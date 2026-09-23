"""LlamaFactory rendered view export and v5 ROCm recipe checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from adapters.lfm import (
    dataset_info_fragment,
    expand_canonical_row_to_views,
    render_supervised_turn,
    validate_llamafactory_full_config,
)
from adapters.schema import contains_chatml_tool_call_markers

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"
CONFIGS = ROOT / "configs"
EXPORT_SCRIPT = ROOT / "scripts" / "export_llamafactory_view.py"
SMOKE_CANONICAL = FIXTURES / "sayso_lfm_smoke.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _correction_row() -> dict:
    wrong_name = "Kitchen Wrong Light"
    correct_name = "Kitchen Counter Lights"
    return {
        "messages": [
            {"role": "system", "content": "You are SaySo.", "train_on_turn": False},
            {"role": "user", "content": "turn on kitchen counter lights", "train_on_turn": False},
            {
                "role": "assistant",
                "content": "",
                "train_on_turn": False,
                "tool_calls": [
                    {
                        "id": "call_wrong",
                        "type": "function",
                        "function": {
                            "name": "intent__HassTurnOn",
                            "arguments": json.dumps(
                                {"domain": ["light"], "name": wrong_name},
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": json.dumps({"code": "schema_mismatch", "message": "bad name"}),
                "train_on_turn": False,
                "tool_call_id": "call_wrong",
            },
            {
                "role": "assistant",
                "content": "",
                "train_on_turn": True,
                "tool_calls": [
                    {
                        "id": "call_right",
                        "type": "function",
                        "function": {
                            "name": "intent__HassTurnOn",
                            "arguments": json.dumps(
                                {"domain": ["light"], "name": correct_name},
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": json.dumps({"result": "Success"}),
                "train_on_turn": False,
                "tool_call_id": "call_right",
            },
            {"role": "assistant", "content": "Done.", "train_on_turn": True},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "intent__HassTurnOn",
                    "description": "Turn on",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "domain": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
            }
        ],
        "metadata": {"family": "correction"},
    }


@pytest.fixture(scope="module")
def smoke_row() -> dict:
    return _load_jsonl(SMOKE_CANONICAL)[0]


def test_smoke_fixture_yields_two_supervised_views(smoke_row: dict) -> None:
    views = expand_canonical_row_to_views(smoke_row, source_row_index=0)
    assert len(views) == 2


def test_first_supervised_view_contains_gold_tool_name(smoke_row: dict) -> None:
    views = expand_canonical_row_to_views(smoke_row)
    tool_view = views[0]
    assert "HassTurnOn" in tool_view["output"]
    assert "Living Room" in tool_view["output"]
    assert "<tool_call>" not in tool_view["instruction"]
    assert "<tool_call>" not in tool_view["output"]


def test_correction_wrong_call_stays_in_prompt_not_output() -> None:
    row = _correction_row()
    views = expand_canonical_row_to_views(row)
    assert len(views) == 2
    wrong_name = "Kitchen Wrong Light"
    correct_name = "Kitchen Counter Lights"
    assert wrong_name not in views[0]["output"]
    assert correct_name in views[0]["output"]
    assert wrong_name in views[1]["instruction"]
    assert wrong_name not in views[1]["output"]
    for view in views:
        assert not contains_chatml_tool_call_markers(view["instruction"])
        assert not contains_chatml_tool_call_markers(view["output"])


def test_render_supervised_turn_includes_native_end_marker(smoke_row: dict) -> None:
    rendered = render_supervised_turn(smoke_row, 2)
    assert rendered["output"].rstrip().endswith("<|im_end|>")


def test_dataset_info_fragment_maps_alpaca_columns() -> None:
    fragment = dataset_info_fragment("sayso_v5_rendered", "sayso_full_sft_v5_rendered.jsonl")
    entry = fragment["sayso_v5_rendered"]
    assert entry["formatting"] == "alpaca"
    assert entry["columns"]["prompt"] == "instruction"
    assert entry["columns"]["response"] == "output"


def test_export_cli_writes_rendered_jsonl_and_dataset_info(tmp_path: Path) -> None:
    out = tmp_path / "rendered.jsonl"
    info = tmp_path / "dataset_info.fragment.json"
    cmd = [
        sys.executable,
        str(EXPORT_SCRIPT),
        str(SMOKE_CANONICAL),
        "-o",
        str(out),
        "--dataset-name",
        "sayso_v5_rendered_smoke",
        "--dataset-info-out",
        str(info),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    payload = json.loads(info.read_text(encoding="utf-8"))
    assert "sayso_v5_rendered_smoke" in payload


def test_full_llamafactory_yaml_matches_v5_contract() -> None:
    with (CONFIGS / "lfm25-230m-full-24gb-rocm-llamafactory.yml").open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    validate_llamafactory_full_config(cfg)
    assert cfg["output_dir"] == "/srv/llm/runs/sayso-lfm-v5-full"
    assert cfg["dataset"] == "sayso_v5_rendered"


def test_smoke_llamafactory_yaml_is_full_recipe_with_max_steps() -> None:
    with (
        CONFIGS / "lfm25-230m-full-24gb-rocm-llamafactory-smoke.yml"
    ).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    validate_llamafactory_full_config(cfg)
    assert cfg["max_steps"] == 2
    assert cfg["dataset"] == "sayso_v5_rendered_smoke"


def test_smoke20_llamafactory_yaml_is_full_recipe_with_max_steps() -> None:
    with (
        CONFIGS / "lfm25-230m-full-24gb-rocm-llamafactory-smoke-20.yml"
    ).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    validate_llamafactory_full_config(cfg)
    assert cfg["max_steps"] == 20
    assert cfg["dataset"] == "sayso_v5_rendered_smoke"
    assert cfg["output_dir"] == "/srv/llm/runs/sayso-lfm-v5-smoke-20"
