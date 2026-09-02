"""Benchmark run configuration persisted with eval JSONL outputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ID = "mlx-community/LFM2.5-230M-OptiQ-4bit"
HOME_LLM_270M_MODEL_ID = "acon96/Home-FunctionGemma-270m"
COMPARISON_BASELINE_RUNTIME = "external"
BENCHMARK_CONFIG_RECORD_KIND = "benchmark_config"


@dataclass(frozen=True)
class BenchmarkConfig:
    model_id: str = DEFAULT_MODEL_ID
    quantization: str = "4bit"
    runtime: str = "fake"
    revision: str | None = None
    seed: int = 0
    warmup_count: int = 0
    cold_start: bool = True

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> BenchmarkConfig:
        return cls(
            model_id=str(payload["model_id"]),
            quantization=str(payload["quantization"]),
            runtime=str(payload["runtime"]),
            revision=None if payload.get("revision") is None else str(payload["revision"]),
            seed=int(payload["seed"]),
            warmup_count=int(payload["warmup_count"]),
            cold_start=bool(payload["cold_start"]),
        )


def parse_benchmark_config(config: BenchmarkConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["record_kind"] = BENCHMARK_CONFIG_RECORD_KIND
    return payload


def config_sidecar_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    return path.with_name(f"{path.stem}.config.json")


def _config_payload_without_record_kind(config: BenchmarkConfig) -> dict[str, Any]:
    return asdict(config)


def write_config_sidecar(output_path: str | Path, config: BenchmarkConfig) -> Path:
    sidecar = config_sidecar_path(output_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps(_config_payload_without_record_kind(config), sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return sidecar


def write_benchmark_config_header(output_path: str | Path, config: BenchmarkConfig) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8").strip():
        return
    path.write_text(
        json.dumps(parse_benchmark_config(config), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_jsonl_header(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if payload.get("record_kind") == BENCHMARK_CONFIG_RECORD_KIND:
            return payload
        return None
    return None


def load_benchmark_config(output_path: str | Path) -> BenchmarkConfig | None:
    path = Path(output_path)
    header = _read_jsonl_header(path)
    if header is not None:
        return BenchmarkConfig.from_payload(header)

    sidecar = config_sidecar_path(path)
    if not sidecar.exists():
        return None
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    return BenchmarkConfig.from_payload(payload)


def is_benchmark_config_line(payload: dict[str, Any]) -> bool:
    return payload.get("record_kind") == BENCHMARK_CONFIG_RECORD_KIND


def comparison_baseline_benchmark_config() -> BenchmarkConfig:
    """Home-LLM 270M comparison slot; adapter not wired in-tree yet."""
    return BenchmarkConfig(
        model_id=HOME_LLM_270M_MODEL_ID,
        runtime=COMPARISON_BASELINE_RUNTIME,
        revision=None,
    )


def sayso_comparison_benchmark_config() -> BenchmarkConfig:
    """SaySo side of the Home-LLM comparison benchmark."""
    return BenchmarkConfig()
