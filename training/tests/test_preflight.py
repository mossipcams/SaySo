"""Synthetic checks for the fast corpus gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from preflight import inspect_dataset  # noqa: E402
from adapters.schema import v2_openai_tools  # noqa: E402


def _row(*, positive: bool = True, family: str = "ordinary", utterance: str = "Pause the TV") -> dict:
    schemas = {tool["function"]["name"]: tool for tool in v2_openai_tools()}
    names = ("GetDateTime", "GetLiveContext", "HassTurnOn", "HassTurnOff", "HassMediaPause")
    tools = []
    for name in names:
        namespace = "llm" if name == "GetDateTime" else "homeassistant" if name == "GetLiveContext" else "media_player" if name == "HassMediaPause" else "intent"
        tool = json.loads(json.dumps(schemas[name]))
        tool["function"]["name"] = f"{namespace}__{name}"
        tools.append(tool)
    messages = [{"role": "user", "content": utterance}, {"role": "assistant", "content": "", "train_on_turn": True}]
    meta = {"family": family, "category": family, "capability": "media_players", "home_id": "fixture", "area_context": {}, "expected_target_names": ["Living Room TV"]}
    if positive:
        messages[-1]["tool_calls"] = [{"id": "call_1", "type": "function", "function": {"name": "media_player__HassMediaPause", "arguments": '{"name":"Living Room TV","domain":["media_player"]}'}}]
    else:
        meta["no_action_reason"] = "area_unavailable"
    return {"messages": messages, "tools": tools, "metadata": meta}


def _run(tmp_path: Path, rows: list[dict], **overrides):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    config = {
        "requirements": {
            "minimum_rows": 1,
            "minimum_positive_by_capability": {"media_players": 1},
            "minimum_positive_by_tool": {"media_player__HassMediaPause": 1},
            "minimum_family_examples": {},
            "required_positive_tools": ["media_player__HassMediaPause"],
        },
        "limits": {
            "max_negative_ratio_by_capability": {"media_players": 2.0},
            "max_duplicate_rate": 0.5,
            "max_conflict_rate": 0.5,
            "max_distribution_shift": 1.0,
        },
    }
    for section, values in overrides.items():
        config[section].update(values)
    return inspect_dataset(dataset, config, {"dataset": {"family_shares": {"ordinary": 1.0}}})


def test_valid_dataset_passes(tmp_path: Path) -> None:
    _, errors = _run(tmp_path, [_row()])
    assert errors == []


def test_missing_positive_coverage_fails(tmp_path: Path) -> None:
    _, errors = _run(tmp_path, [_row(positive=False)])
    assert any("positive examples for media_players" in error for error in errors)


def test_excessive_negative_ratio_fails(tmp_path: Path) -> None:
    _, errors = _run(tmp_path, [_row(), _row(positive=False)], limits={"max_negative_ratio_by_capability": {"media_players": 0.5}})
    assert any("negative/positive ratio" in error for error in errors)


def test_malformed_tool_call_fails(tmp_path: Path) -> None:
    row = _row()
    row["messages"][-1]["tool_calls"][0]["function"]["arguments"] = "{"
    _, errors = _run(tmp_path, [row])
    assert any("invalid tool call" in error for error in errors)


def test_malformed_training_row_fails(tmp_path: Path) -> None:
    report, errors = _run(tmp_path, [{"messages": "bad"}])
    assert report["invalid_rows"] == 1
    assert any("messages/tools/metadata have invalid types" in error for error in errors)


def test_unknown_target_fails(tmp_path: Path) -> None:
    row = _row()
    row["messages"][-1]["tool_calls"][0]["function"]["arguments"] = '{"name":"Unknown TV","domain":["media_player"]}'
    _, errors = _run(tmp_path, [row])
    assert any("unknown target" in error for error in errors)


def test_duplicate_examples_fail_above_limit(tmp_path: Path) -> None:
    _, errors = _run(tmp_path, [_row(), _row()], limits={"max_duplicate_rate": 0.1})
    assert any("duplicate rate" in error for error in errors)


def test_conflicting_examples_fail_above_limit(tmp_path: Path) -> None:
    first, second = _row(), _row()
    second["messages"][-1]["tool_calls"][0]["function"]["arguments"] = '{"name":"Different TV","domain":["media_player"]}'
    second["metadata"]["expected_target_names"] = ["Different TV"]
    _, errors = _run(tmp_path, [first, second], limits={"max_conflict_rate": 0.1})
    assert any("conflicting example rate" in error for error in errors)


def test_distribution_drift_fails(tmp_path: Path) -> None:
    _, errors = _run(tmp_path, [_row(family="clarify")], limits={"max_distribution_shift": 0.2})
    assert any("distribution shift" in error for error in errors)
