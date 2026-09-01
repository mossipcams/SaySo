"""CLI tests for python -m evals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evals.config import BENCHMARK_CONFIG_RECORD_KIND, is_benchmark_config_line
from evals.__main__ import build_parser, check_expansion_gate, load_corpus_cases, main, parse_allowlist
from evals.executor import controller_dry_run_executor
from evals.mlx_executor import MLX_EVAL_ENV_VAR, controller_mlx_executor
from evals.runner import BenchmarkRunResult
from evals.schema import EvalCase


def _tiny_case_payload(case_id: str = "cli-001") -> dict[str, Any]:
    return {
        "case_id": case_id,
        "category": "simple_control",
        "home": "eval-home",
        "origin": "area_living_room",
        "turns": ["Turn off the ceiling lights"],
        "expected_control_plan": {
            "outcome": "action",
            "intent": "turn off the ceiling lights",
            "domain": "light",
            "targets": ["ceiling lights"],
            "state": "off",
        },
        "expected_candidate_entities": ["light.living_room_ceiling"],
        "expected_resolved_entities": ["light.living_room_ceiling"],
        "expected_outcome": "valid_action",
        "execution_allowed": True,
    }


def _tiny_case(case_id: str = "cli-001") -> EvalCase:
    return EvalCase.model_validate(_tiny_case_payload(case_id))


def test_build_parser_requires_corpus_and_output() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(["--corpus", "core", "--output", "/tmp/out.jsonl"])
    assert args.corpus == "core"
    assert args.output == Path("/tmp/out.jsonl")
    assert args.execute is False
    assert args.allowlist == ""
    assert args.warmup == 0


def test_parse_allowlist_splits_and_strips() -> None:
    assert parse_allowlist("") == []
    assert parse_allowlist(" light.a , light.b ") == ["light.a", "light.b"]


def test_main_calls_run_benchmark_with_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _tiny_case()
    output = tmp_path / "out.jsonl"
    captured: dict[str, object] = {}

    def fake_load(corpus: str) -> list[EvalCase]:
        assert corpus == "core"
        return [case]

    def fake_run_benchmark(
        cases: list[EvalCase] | str | Path,
        output_path: str | Path,
        executor=None,
        *,
        execute: bool = False,
        entity_allowlist=(),
        **kwargs: object,
    ) -> BenchmarkRunResult:
        captured["cases"] = cases
        captured["output_path"] = Path(output_path)
        captured["executor"] = executor
        captured["execute"] = execute
        captured["entity_allowlist"] = list(entity_allowlist)
        return BenchmarkRunResult(scored=1, skipped=0, warmup_runs=0, errors=0)

    monkeypatch.setattr("evals.__main__.load_corpus_cases", fake_load)
    monkeypatch.setattr("evals.__main__.run_benchmark", fake_run_benchmark)
    monkeypatch.setattr("evals.__main__.resolve_eval_executor", lambda: controller_dry_run_executor)

    rc = main(["--corpus", "core", "--output", str(output)])

    assert rc == 0
    assert captured["cases"] == [case]
    assert captured["output_path"] == output
    assert captured["executor"] is controller_dry_run_executor
    assert captured["execute"] is False
    assert captured["entity_allowlist"] == []


def test_build_parser_accepts_warmup() -> None:
    args = build_parser().parse_args(
        ["--corpus", "core", "--output", "/tmp/out.jsonl", "--warmup", "5"],
    )
    assert args.warmup == 5


def test_main_passes_fake_runtime_when_mlx_unset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "out.jsonl"
    captured: dict[str, object] = {}

    monkeypatch.delenv(MLX_EVAL_ENV_VAR, raising=False)
    monkeypatch.setattr(
        "evals.__main__.load_corpus_cases",
        lambda corpus: [_tiny_case("fake-runtime-001")],
    )

    def fake_run_benchmark(
        cases: list[EvalCase] | str | Path,
        output_path: str | Path,
        executor=None,
        *,
        config=None,
        seed: int = 0,
        warmup_count: int = 0,
        execute: bool = False,
        entity_allowlist=(),
        **kwargs: object,
    ) -> BenchmarkRunResult:
        captured["config"] = config
        captured["seed"] = seed
        captured["warmup_count"] = warmup_count
        captured["executor"] = executor
        return BenchmarkRunResult(scored=1, skipped=0, warmup_runs=0, errors=0)

    monkeypatch.setattr("evals.__main__.run_benchmark", fake_run_benchmark)

    rc = main(["--corpus", "core", "--output", str(output)])

    assert rc == 0
    assert captured["executor"] is controller_dry_run_executor
    assert captured["config"].runtime == "fake"
    assert captured["seed"] == captured["config"].seed
    assert captured["warmup_count"] == captured["config"].warmup_count


def test_main_passes_mlx_runtime_when_mlx_executor_resolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "out.jsonl"
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "evals.__main__.load_corpus_cases",
        lambda corpus: [_tiny_case("mlx-runtime-001")],
    )
    monkeypatch.setattr(
        "evals.__main__.resolve_eval_executor",
        lambda: controller_mlx_executor,
    )

    def fake_run_benchmark(
        cases: list[EvalCase] | str | Path,
        output_path: str | Path,
        executor=None,
        *,
        config=None,
        seed: int = 0,
        warmup_count: int = 0,
        **kwargs: object,
    ) -> BenchmarkRunResult:
        captured["config"] = config
        captured["seed"] = seed
        captured["warmup_count"] = warmup_count
        captured["executor"] = executor
        return BenchmarkRunResult(scored=1, skipped=0, warmup_runs=0, errors=0)

    monkeypatch.setattr("evals.__main__.run_benchmark", fake_run_benchmark)

    rc = main(["--corpus", "core", "--output", str(output), "--warmup", "2"])

    assert rc == 0
    assert captured["executor"] is controller_mlx_executor
    assert captured["config"].runtime == "mlx"
    assert captured["config"].warmup_count == 2
    assert captured["seed"] == captured["config"].seed
    assert captured["warmup_count"] == 2
    assert captured["warmup_count"] == captured["config"].warmup_count


def test_main_warmup_appears_in_jsonl_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "results.jsonl"

    monkeypatch.delenv(MLX_EVAL_ENV_VAR, raising=False)
    monkeypatch.setattr(
        "evals.__main__.load_corpus_cases",
        lambda corpus: [_tiny_case("warmup-header-001")],
    )

    rc = main(["--corpus", "core", "--output", str(output), "--warmup", "3"])

    assert rc == 0
    lines = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert lines[0]["record_kind"] == BENCHMARK_CONFIG_RECORD_KIND
    assert lines[0]["runtime"] == "fake"
    assert lines[0]["warmup_count"] == 3
    record_lines = [line for line in lines if not is_benchmark_config_line(line)]
    assert len(record_lines) == 1
    assert record_lines[0]["case_id"] == "warmup-header-001"


def test_main_passes_execute_and_allowlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "out.jsonl"
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "evals.__main__.load_corpus_cases",
        lambda corpus: [_tiny_case(f"{corpus}-001")],
    )

    def fake_run_benchmark(
        cases: list[EvalCase] | str | Path,
        output_path: str | Path,
        executor=None,
        *,
        execute: bool = False,
        entity_allowlist=(),
        **kwargs: object,
    ) -> BenchmarkRunResult:
        captured["execute"] = execute
        captured["entity_allowlist"] = list(entity_allowlist)
        return BenchmarkRunResult(scored=1, skipped=0, warmup_runs=0, errors=0)

    monkeypatch.setattr("evals.__main__.run_benchmark", fake_run_benchmark)
    monkeypatch.setattr("evals.__main__.resolve_eval_executor", lambda: controller_dry_run_executor)

    rc = main(
        [
            "--corpus",
            "safety",
            "--output",
            str(output),
            "--execute",
            "--allowlist",
            "light.a,light.b",
        ],
    )

    assert rc == 0
    assert captured["execute"] is True
    assert captured["entity_allowlist"] == ["light.a", "light.b"]


def test_load_corpus_cases_all_concatenates_in_stable_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    def make_loader(name: str):
        def loader() -> list[EvalCase]:
            order.append(name)
            return [_tiny_case(f"{name}-001")]

        return loader

    monkeypatch.setattr("evals.__main__.load_core_corpus", make_loader("core"))
    monkeypatch.setattr("evals.__main__.load_safety_corpus", make_loader("safety"))
    monkeypatch.setattr(
        "evals.__main__.load_language_noise_corpus",
        make_loader("language_noise"),
    )
    monkeypatch.setattr("evals.__main__.load_followup_corpus", make_loader("followup"))

    cases = load_corpus_cases("all")

    assert order == ["core", "safety", "language_noise", "followup"]
    assert [case.case_id for case in cases] == [
        "core-001",
        "safety-001",
        "language_noise-001",
        "followup-001",
    ]


def test_main_integration_with_tiny_jsonl_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_path = tmp_path / "tiny.jsonl"
    corpus_path.write_text(
        json.dumps(_tiny_case_payload("tiny-001")) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "results.jsonl"

    monkeypatch.setattr(
        "evals.__main__.load_corpus_cases",
        lambda corpus: [_tiny_case("tiny-001")] if corpus == "core" else [],
    )

    rc = main(["--corpus", "core", "--output", str(output)])

    assert rc == 0
    lines = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record_lines = [line for line in lines if not is_benchmark_config_line(line)]
    assert len(record_lines) == 1
    assert record_lines[0]["case_id"] == "tiny-001"
    assert record_lines[0]["ha_executed"] is False
    assert record_lines[0]["recorded_control_plan"] is not None


def test_build_parser_accepts_check_gate() -> None:
    args = build_parser().parse_args(
        ["--corpus", "core", "--output", "/tmp/out.jsonl", "--check-gate"],
    )
    assert args.check_gate is True


def test_main_check_gate_exits_1_when_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "out.jsonl"
    output.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        "evals.__main__.load_corpus_cases",
        lambda corpus: [_tiny_case("blocked-001")],
    )
    monkeypatch.setattr(
        "evals.__main__.check_expansion_gate",
        lambda cases, output_path: 1,
    )

    rc = main(["--corpus", "core", "--output", str(output), "--check-gate"])

    assert rc == 1


def test_main_check_gate_exits_0_when_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "out.jsonl"
    output.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        "evals.__main__.load_corpus_cases",
        lambda corpus: [_tiny_case("allowed-001")],
    )
    monkeypatch.setattr(
        "evals.__main__.check_expansion_gate",
        lambda cases, output_path: 0,
    )

    rc = main(["--corpus", "core", "--output", str(output), "--check-gate"])

    assert rc == 0


def test_check_expansion_gate_integration_blocks_zero_latency_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "out.jsonl"
    output.write_text(
        json.dumps(
            {
                "case_id": "tiny-001",
                "recorded_control_plan": _tiny_case_payload()["expected_control_plan"],
                "recorded_candidate_entities": ["light.living_room_ceiling"],
                "recorded_resolved_entities": ["light.living_room_ceiling"],
                "ha_executed": False,
                "cold_start": True,
            },
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "evals.__main__.load_corpus_cases",
        lambda corpus: [_tiny_case("tiny-001")],
    )

    rc = check_expansion_gate([_tiny_case("tiny-001")], output)

    assert rc == 1

