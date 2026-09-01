"""Token normalization for candidate retrieval."""

from __future__ import annotations

import re

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def normalize_tokens(text: str) -> list[str]:
    """Lowercase text and return alphanumeric tokens."""
    return _TOKEN_PATTERN.findall(text.lower())


def normalize_labels(labels: list[str]) -> set[str]:
    """Normalize names and aliases into comparable token sets."""
    tokens: set[str] = set()
    for label in labels:
        tokens.update(normalize_tokens(label))
    return tokens
