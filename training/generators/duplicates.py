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
    """Track exact utterance/home pairs and semantic IDs per split."""

    def __init__(self, near_limit: int = 3) -> None:
        # near_limit applies to identical utterance+home pairs, not utterance text alone.
        self.near_limit = near_limit
        self._pair_counts: dict[str, int] = {}
        self._semantic_ids: set[str] = set()

    def would_reject(self, spec: dict[str, Any]) -> str | None:
        utterance = spec.get("utterance") or ""
        ph = pair_hash(utterance, spec.get("home", {}))
        if self._pair_counts.get(ph, 0) >= self.near_limit:
            return "exact_duplicate_utterance"
        sem = spec.get("semantic_id") or spec.get("candidate_id")
        if sem in self._semantic_ids:
            return "duplicate_semantic_id"
        return None

    def record(self, spec: dict[str, Any]) -> None:
        utterance = spec.get("utterance") or ""
        ph = pair_hash(utterance, spec.get("home", {}))
        self._pair_counts[ph] = self._pair_counts.get(ph, 0) + 1
        sem = spec.get("semantic_id") or spec.get("candidate_id")
        self._semantic_ids.add(sem)
