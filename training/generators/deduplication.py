"""Duplicate detection across splits and utterances."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

_NORMALIZE = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize(text: str) -> str:
    return " ".join(_NORMALIZE.sub(" ", text.casefold()).split())


def utterance_hash(utterance: str) -> str:
    return hashlib.sha256(_normalize(utterance).encode()).hexdigest()


def context_hash(home: dict[str, Any]) -> str:
    payload = json.dumps(
        [{"name": e["name"], "area": e["area"]} for e in home.get("entities", [])],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def pair_hash(utterance: str, home: dict[str, Any]) -> str:
    return hashlib.sha256(f"{utterance_hash(utterance)}:{context_hash(home)}".encode()).hexdigest()


class DuplicateTracker:
    def __init__(self, near_limit: int = 3) -> None:
        self.near_limit = near_limit
        self._pair_counts: dict[str, int] = {}
        self._semantic_counts: dict[str, int] = {}

    def occurrences(self, spec: dict[str, Any]) -> int:
        return self._pair_counts.get(pair_hash(spec.get("utterance") or "", spec.get("home", {})), 0)

    def would_reject(self, spec: dict[str, Any]) -> str | None:
        ph = pair_hash(spec.get("utterance") or "", spec.get("home", {}))
        if self._pair_counts.get(ph, 0) >= self.near_limit:
            return "exact_duplicate_utterance"
        sem = spec.get("semantic_id") or spec.get("candidate_id")
        if self._semantic_counts.get(sem, 0) >= self.near_limit:
            return "duplicate_semantic_id"
        return None

    def record(self, spec: dict[str, Any]) -> None:
        ph = pair_hash(spec.get("utterance") or "", spec.get("home", {}))
        self._pair_counts[ph] = self._pair_counts.get(ph, 0) + 1
        sem = spec.get("semantic_id") or spec.get("candidate_id")
        self._semantic_counts[sem] = self._semantic_counts.get(sem, 0) + 1

__all__ = [
    "DuplicateTracker",
    "context_hash",
    "pair_hash",
    "utterance_hash",
]
