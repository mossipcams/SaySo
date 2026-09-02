"""Opt-in live latency measurement against llama.cpp."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from custom_components.sayso.client import LlamaCppClient, build_chat_completions_payload

from evals.metrics import build_live_latency_report


@dataclass(frozen=True, slots=True)
class LiveLatencyConfig:
    """Warmup and repetition settings for live latency runs."""

    warmups: int = 1
    repetitions: int = 3


@dataclass(frozen=True, slots=True)
class LiveLatencySample:
    """One live latency measurement."""

    ttft_ms: float
    end_to_end_ms: float


async def measure_live_latency_once(
    client: LlamaCppClient,
    payload: dict[str, Any],
    *,
    execute_completion: Callable[[], Awaitable[Any]],
) -> LiveLatencySample:
    """Measure TTFT via streaming probe and end-to-end through HA completion."""
    ttft_ms = await client.probe_ttft_ms(payload)

    end_to_end_start = time.perf_counter()
    await execute_completion()
    end_to_end_ms = (time.perf_counter() - end_to_end_start) * 1000.0

    return LiveLatencySample(ttft_ms=ttft_ms, end_to_end_ms=end_to_end_ms)


async def run_live_latency_benchmark(
    client: LlamaCppClient,
    *,
    messages: list[dict[str, Any]],
    model: str,
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0,
    max_tokens: int = 160,
    config: LiveLatencyConfig | None = None,
    execute_completion: Callable[[], Awaitable[Any]] | None = None,
) -> dict[str, Any]:
    """Run warmups and repetitions, returning median/p95 TTFT and end-to-end latency."""
    settings = config or LiveLatencyConfig()
    payload = build_chat_completions_payload(
        messages,
        model=model,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    async def _default_execute_completion() -> Any:
        return await client.chat_completion(
            messages,
            model=model,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    completion_runner = execute_completion or _default_execute_completion

    for _ in range(settings.warmups):
        await measure_live_latency_once(
            client,
            payload,
            execute_completion=completion_runner,
        )

    ttft_samples_ms: list[float] = []
    end_to_end_samples_ms: list[float] = []
    for _ in range(settings.repetitions):
        sample = await measure_live_latency_once(
            client,
            payload,
            execute_completion=completion_runner,
        )
        ttft_samples_ms.append(sample.ttft_ms)
        end_to_end_samples_ms.append(sample.end_to_end_ms)

    return build_live_latency_report(
        ttft_samples_ms=ttft_samples_ms,
        end_to_end_samples_ms=end_to_end_samples_ms,
        warmups=settings.warmups,
        repetitions=settings.repetitions,
    )
