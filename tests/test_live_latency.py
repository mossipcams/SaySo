"""Live llama.cpp latency measurement. Not a second eval runner."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from custom_components.sayso.client import LlamaCppClient, build_chat_completions_payload


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight


def compute_latency_percentiles(latencies_ms: list[float]) -> dict[str, float]:
    ordered = sorted(latencies_ms)
    return {"p50": _percentile(ordered, 0.5), "p95": _percentile(ordered, 0.95)}


def build_live_latency_report(
    *,
    ttft_samples_ms: list[float],
    end_to_end_samples_ms: list[float],
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    return {
        "warmups": warmups,
        "repetitions": repetitions,
        "ttft_ms": compute_latency_percentiles(ttft_samples_ms),
        "end_to_end_ms": compute_latency_percentiles(end_to_end_samples_ms),
    }


@dataclass(frozen=True, slots=True)
class LiveLatencyConfig:
    warmups: int = 1
    repetitions: int = 3


@dataclass(frozen=True, slots=True)
class LiveLatencySample:
    ttft_ms: float
    end_to_end_ms: float


async def measure_live_latency_once(
    client: LlamaCppClient,
    payload: dict[str, Any],
    *,
    execute_completion: Callable[[], Awaitable[Any]],
) -> LiveLatencySample:
    ttft_ms = await client.probe_ttft_ms(payload)
    started = time.perf_counter()
    await execute_completion()
    return LiveLatencySample(ttft_ms=ttft_ms, end_to_end_ms=(time.perf_counter() - started) * 1000.0)


async def run_live_latency_benchmark(
    client: LlamaCppClient,
    *,
    messages: list[dict[str, Any]],
    model: str,
    config: LiveLatencyConfig | None = None,
    execute_completion: Callable[[], Awaitable[Any]] | None = None,
) -> dict[str, Any]:
    settings = config or LiveLatencyConfig()
    payload = build_chat_completions_payload(messages, model=model)

    async def _default_execute_completion() -> Any:
        return await client.chat_completion(messages, model=model)

    completion_runner = execute_completion or _default_execute_completion
    for _ in range(settings.warmups):
        await measure_live_latency_once(client, payload, execute_completion=completion_runner)
    ttft_samples_ms: list[float] = []
    end_to_end_samples_ms: list[float] = []
    for _ in range(settings.repetitions):
        sample = await measure_live_latency_once(client, payload, execute_completion=completion_runner)
        ttft_samples_ms.append(sample.ttft_ms)
        end_to_end_samples_ms.append(sample.end_to_end_ms)
    return build_live_latency_report(
        ttft_samples_ms=ttft_samples_ms,
        end_to_end_samples_ms=end_to_end_samples_ms,
        warmups=settings.warmups,
        repetitions=settings.repetitions,
    )


class _FakeSseStream:
    def __init__(self, *, token_delay_s: float) -> None:
        self._token_delay_s = token_delay_s
        self._lines = [
            b'data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}\n',
            b"data: [DONE]\n",
        ]
        self._index = 0
        self.first_token_sent_at: float | None = None

    async def readline(self) -> bytes:
        if self._index >= len(self._lines):
            return b""
        if self._index == 0:
            await asyncio.sleep(self._token_delay_s)
            self.first_token_sent_at = time.perf_counter()
        line = self._lines[self._index]
        self._index += 1
        return line


def _configure_stream_post(
    mock_session: aiohttp.ClientSession,
    *,
    token_delay_s: float,
) -> tuple[list[float], _FakeSseStream]:
    post_started_at: list[float] = []
    stream = _FakeSseStream(token_delay_s=token_delay_s)

    def _post(*_args: Any, **_kwargs: Any) -> AsyncMock:
        post_started_at.append(time.perf_counter())
        response = AsyncMock()
        response.status = 200
        response.content = stream
        context_manager = AsyncMock()
        context_manager.__aenter__ = AsyncMock(return_value=response)
        context_manager.__aexit__ = AsyncMock(return_value=False)
        return context_manager

    mock_session.post = MagicMock(side_effect=_post)
    return post_started_at, stream


def _configure_routing_post(
    mock_session: aiohttp.ClientSession,
    *,
    token_delay_s: float,
    completion_delay_s: float,
) -> dict[str, int]:
    counts = {"stream": 0, "completion": 0}

    def _post(*_args: Any, **kwargs: Any) -> AsyncMock:
        response = AsyncMock()
        response.status = 200
        if kwargs.get("json", {}).get("stream") is True:
            counts["stream"] += 1
            response.content = _FakeSseStream(token_delay_s=token_delay_s)
            context_manager = AsyncMock()
            context_manager.__aenter__ = AsyncMock(return_value=response)
            context_manager.__aexit__ = AsyncMock(return_value=False)
            return context_manager
        counts["completion"] += 1

        async def _json(**_kwargs: Any) -> dict[str, Any]:
            await asyncio.sleep(completion_delay_s)
            return {
                "choices": [
                    {"message": {"role": "assistant", "content": "The living room light is on."}}
                ]
            }

        response.json = AsyncMock(side_effect=_json)
        context_manager = AsyncMock()
        context_manager.__aenter__ = AsyncMock(return_value=response)
        context_manager.__aexit__ = AsyncMock(return_value=False)
        return context_manager

    mock_session.post = MagicMock(side_effect=_post)
    return counts


async def test_probe_ttft_starts_before_post_and_ends_on_first_token(
    mock_session: aiohttp.ClientSession,
) -> None:
    token_delay_s = 0.05
    post_started_at, stream = _configure_stream_post(mock_session, token_delay_s=token_delay_s)
    payload = build_chat_completions_payload(
        [{"role": "user", "content": "turn on the living room light"}],
        model="test-model",
    )
    client = LlamaCppClient(mock_session, "http://127.0.0.1:8080/v1", timeout=30)
    ttft_ms = await client.probe_ttft_ms(payload)
    assert len(post_started_at) == 1
    assert stream.first_token_sent_at is not None
    server_ttft_ms = (stream.first_token_sent_at - post_started_at[0]) * 1000.0
    assert ttft_ms == pytest.approx(server_ttft_ms, rel=0.25, abs=15.0)
    assert mock_session.post.call_args.kwargs["json"]["stream"] is True
    assert "stream" not in payload


async def test_measure_live_latency_once_ends_after_final_ha_result(
    mock_session: aiohttp.ClientSession,
) -> None:
    token_delay_s = 0.02
    completion_delay_s = 0.04
    ha_delay_s = 0.03
    _configure_routing_post(
        mock_session, token_delay_s=token_delay_s, completion_delay_s=completion_delay_s
    )
    payload = build_chat_completions_payload(
        [{"role": "user", "content": "turn on the living room light"}],
        model="test-model",
    )
    client = LlamaCppClient(mock_session, "http://127.0.0.1:8080/v1", timeout=30)

    async def execute_completion() -> str:
        await client.chat_completion(
            payload["messages"],
            model=payload["model"],
            temperature=payload["temperature"],
            max_tokens=payload["max_tokens"],
        )
        await asyncio.sleep(ha_delay_s)
        return "ha-complete"

    sample = await measure_live_latency_once(client, payload, execute_completion=execute_completion)
    min_e2e_ms = (completion_delay_s + ha_delay_s) * 1000.0
    assert sample.end_to_end_ms >= min_e2e_ms * 0.8
    assert sample.end_to_end_ms > sample.ttft_ms


async def test_run_live_latency_benchmark_records_warmups_repetitions_median_p95(
    mock_session: aiohttp.ClientSession,
) -> None:
    token_delay_s = 0.01
    completion_delay_s = 0.02
    ha_delay_s = 0.01
    counts = _configure_routing_post(
        mock_session, token_delay_s=token_delay_s, completion_delay_s=completion_delay_s
    )
    client = LlamaCppClient(mock_session, "http://127.0.0.1:8080/v1", timeout=30)

    async def execute_completion() -> str:
        await client.chat_completion(
            [{"role": "user", "content": "turn on the living room light"}],
            model="test-model",
        )
        await asyncio.sleep(ha_delay_s)
        return "ha-complete"

    report = await run_live_latency_benchmark(
        client,
        messages=[{"role": "user", "content": "turn on the living room light"}],
        model="test-model",
        config=LiveLatencyConfig(warmups=1, repetitions=3),
        execute_completion=execute_completion,
    )
    assert report["warmups"] == 1
    assert report["repetitions"] == 3
    assert set(report["ttft_ms"]) == {"p50", "p95"}
    assert set(report["end_to_end_ms"]) == {"p50", "p95"}
    assert counts["stream"] == 4
    assert counts["completion"] == 4


def test_build_live_latency_report_is_deterministic() -> None:
    report = build_live_latency_report(
        ttft_samples_ms=[40.0, 50.0, 60.0, 70.0, 80.0],
        end_to_end_samples_ms=[120.0, 140.0, 160.0, 180.0, 200.0],
        warmups=1,
        repetitions=5,
    )
    assert report == {
        "warmups": 1,
        "repetitions": 5,
        "ttft_ms": {"p50": 60.0, "p95": 78.0},
        "end_to_end_ms": {"p50": 160.0, "p95": 196.0},
    }
