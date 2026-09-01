"""Tests for benchmark run configuration (M1)."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from evals.config import (
    BENCHMARK_CONFIG_RECORD_KIND,
    DEFAULT_MODEL_ID,
    BenchmarkConfig,
    config_sidecar_path,
    load_benchmark_config,
    parse_benchmark_config,
    write_benchmark_config_header,
    write_config_sidecar,
)
from evals.metrics import EvalRecord
from evals.runner import load_output_case_ids, run_benchmark
from evals.schema import EvalCase


def test_default_model_id_matches_server_lfm_checkpoint() -> None:
    assert DEFAULT_MODEL_ID == "mlx-community/LFM2.5-230M-OptiQ-4bit"
    config = BenchmarkConfig()
    assert config.model_id == DEFAULT_MODEL_ID


def test_benchmark_config_defaults() -> None:
    config = BenchmarkConfig()
    assert config.quantization == "4bit"
    assert config.runtime == "fake"
    assert config.revision is None
    assert config.seed == 0
    assert config.warmup_count == 0
    assert config.cold_start is True


def test_benchmark_config_is_frozen() -> None:
    config = BenchmarkConfig()
    with pytest.raises(FrozenInstanceError):
        config.seed = 42  # type: ignore[misc]


def test_benchmark_config_custom_fields() -> None:
    config = BenchmarkConfig(
        model_id="custom/model",
        quantization="8bit",
        runtime="mlx",
        revision="abc123",
        seed=7,
        warmup_count=3,
        cold_start=False,
    )
    assert config.model_id == "custom/model"
    assert config.quantization == "8bit"
    assert config.runtime == "mlx"
    assert config.revision == "abc123"
    assert config.seed == 7
    assert config.warmup_count == 3
    assert config.cold_start is False


def test_parse_benchmark_config_roundtrip() -> None:
    config = BenchmarkConfig(seed=11, warmup_count=2, revision="rev-1")
    payload = parse_benchmark_config(config)
    restored = BenchmarkConfig.from_payload(payload)
    assert restored == config


def test_write_benchmark_config_header_creates_first_jsonl_line(tmp_path: Path) -> None:
    output = tmp_path / "run.jsonl"
    config = BenchmarkConfig(seed=5, warmup_count=1)

    write_benchmark_config_header(output, config)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    header = json.loads(lines[0])
    assert header["record_kind"] == BENCHMARK_CONFIG_RECORD_KIND
    assert header["model_id"] == DEFAULT_MODEL_ID
    assert header["quantization"] == "4bit"
    assert header["runtime"] == "fake"
    assert header["seed"] == 5
    assert header["warmup_count"] == 1
    assert header["cold_start"] is True


def test_write_benchmark_config_header_is_idempotent(tmp_path: Path) -> None:
    output = tmp_path / "run.jsonl"
    first = BenchmarkConfig(seed=1)
    second = BenchmarkConfig(seed=99)

    write_benchmark_config_header(output, first)
    write_benchmark_config_header(output, second)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["seed"] == 1


def test_load_benchmark_config_from_jsonl_header(tmp_path: Path) -> None:
    output = tmp_path / "run.jsonl"
    expected = BenchmarkConfig(seed=42, warmup_count=2, runtime="mlx")
    write_benchmark_config_header(output, expected)

    loaded = load_benchmark_config(output)
    assert loaded == expected


def test_config_sidecar_path_and_write(tmp_path: Path) -> None:
    output = tmp_path / "reports" / "core.jsonl"
    sidecar = config_sidecar_path(output)
    assert sidecar == tmp_path / "reports" / "core.config.json"

    config = BenchmarkConfig(seed=3, revision="deadbeef")
    write_config_sidecar(output, config)

    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["model_id"] == DEFAULT_MODEL_ID
    assert payload["seed"] == 3
    assert payload["revision"] == "deadbeef"


def test_load_benchmark_config_prefers_jsonl_header_over_sidecar(tmp_path: Path) -> None:
    output = tmp_path / "run.jsonl"
    header_config = BenchmarkConfig(seed=1)
    sidecar_config = BenchmarkConfig(seed=2)
    write_benchmark_config_header(output, header_config)
    write_config_sidecar(output, sidecar_config)

    assert load_benchmark_config(output) == header_config


def test_load_benchmark_config_from_sidecar_when_no_header(tmp_path: Path) -> None:
    output = tmp_path / "run.jsonl"
    expected = BenchmarkConfig(seed=8)
    write_config_sidecar(output, expected)

    assert load_benchmark_config(output) == expected


def _action_case(case_id: str) -> EvalCase:
    return EvalCase.model_validate(
        {
            "case_id": case_id,
            "category": "simple_control",
            "home": "eval-home",
            "origin": "area_living_room",
            "turns": ["Turn off the lights"],
            "expected_control_plan": {
                "outcome": "action",
                "intent": "turn off the lights",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "state": "off",
            },
            "expected_candidate_entities": ["light.living_room_ceiling"],
            "expected_resolved_entities": ["light.living_room_ceiling"],
            "expected_outcome": "valid_action",
            "execution_allowed": True,
        },
    )


def test_run_benchmark_persists_config_header_with_seed_and_warmup(tmp_path: Path) -> None:
    output = tmp_path / "run.jsonl"
    case = _action_case("cfg-001")

    summary = run_benchmark([case], output, seed=17, warmup_count=2)

    assert summary.warmup_runs == 2
    lines = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert lines[0]["record_kind"] == BENCHMARK_CONFIG_RECORD_KIND
    assert lines[0]["model_id"] == DEFAULT_MODEL_ID
    assert lines[0]["seed"] == 17
    assert lines[0]["warmup_count"] == 2
    assert lines[1]["case_id"] == "cfg-001"
    EvalRecord.model_validate(lines[1])


def test_load_output_case_ids_skips_benchmark_config_header(tmp_path: Path) -> None:
    output = tmp_path / "run.jsonl"
    config = BenchmarkConfig()
    write_benchmark_config_header(output, config)
    with output.open("a", encoding="utf-8") as out:
        out.write(
            json.dumps({"case_id": "done-001", "ha_executed": False}, sort_keys=True)
            + "\n",
        )

    assert load_output_case_ids(output) == {"done-001"}


def test_run_benchmark_resume_does_not_duplicate_config_header(tmp_path: Path) -> None:
    cases = [_action_case("resume-cfg-001"), _action_case("resume-cfg-002")]
    output = tmp_path / "run.jsonl"

    run_benchmark(cases[:1], output, seed=3)
    run_benchmark(cases, output, seed=99)

    lines = output.read_text(encoding="utf-8").splitlines()
    config_lines = [
        line for line in lines if json.loads(line).get("record_kind") == BENCHMARK_CONFIG_RECORD_KIND
    ]
    assert len(config_lines) == 1
    assert json.loads(config_lines[0])["seed"] == 3


def test_run_benchmark_uses_config_seed_and_warmup_over_kwargs(tmp_path: Path) -> None:
    case = _action_case("cfg-seed-001")
    output = tmp_path / "run.jsonl"
    config = BenchmarkConfig(seed=42, warmup_count=1)

    summary = run_benchmark(
        [case],
        output,
        config=config,
        seed=99,
        warmup_count=5,
    )

    assert summary.warmup_runs == 1
    lines = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert lines[0]["seed"] == 42
    assert lines[0]["warmup_count"] == 1
    assert "cold_start" not in lines[1]
