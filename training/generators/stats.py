"""Generation statistics aggregation."""

from __future__ import annotations

from collections import Counter
from typing import Any


def empty_stats() -> dict[str, Any]:
    return {
        "accepted": 0,
        "rejected": 0,
        "rejection_reasons": Counter(),
        "by_tier": Counter(),
        "by_capability": Counter(),
        "by_operation": Counter(),
        "by_home_size": Counter(),
        "by_difficulty": Counter(),
        "stt_corrupted": 0,
        "paraphrased": 0,
        "deterministic": 0,
        "unique_semantic_ids": 0,
    }


def record_accept(stats: dict[str, Any], row: dict[str, Any]) -> None:
    stats["accepted"] += 1
    meta = row.get("metadata") or {}
    stats["by_tier"][meta.get("tier")] += 1
    stats["by_capability"][meta.get("capability")] += 1
    stats["by_operation"][meta.get("operation")] += 1
    stats["by_home_size"][meta.get("home_size")] += 1
    stats["by_difficulty"][meta.get("category", "ordinary")] += 1
    if meta.get("stt_corruption"):
        stats["stt_corrupted"] += 1
    if meta.get("paraphrase_source"):
        stats["paraphrased"] += 1
    else:
        stats["deterministic"] += 1


def record_reject(stats: dict[str, Any], reason: str) -> None:
    stats["rejected"] += 1
    stats["rejection_reasons"][reason] += 1


def finalize_stats(
    stats: dict[str, Any],
    semantic_ids: set[str],
    *,
    quota_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stats["unique_semantic_ids"] = len(semantic_ids)
    if quota_summary is not None:
        stats["quota"] = quota_summary
    return {
        k: (dict(v) if isinstance(v, Counter) else v)
        for k, v in stats.items()
    }
