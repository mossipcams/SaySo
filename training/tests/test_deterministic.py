"""Deterministic generation checks."""

from __future__ import annotations

import io
import json
from pathlib import Path

from adapters.home_llm_v2 import convert_jsonl_stream

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_same_seed_same_output_count() -> None:
    lines = (FIXTURES / "home_llm_v2_example.jsonl").read_text(encoding="utf-8").strip().splitlines()
    buf_a = io.StringIO()
    buf_b = io.StringIO()
    stats_a, count_a = convert_jsonl_stream(iter(lines), seed=1234, output=buf_a)
    stats_b, count_b = convert_jsonl_stream(iter(lines), seed=1234, output=buf_b)
    assert count_a == count_b
    assert buf_a.getvalue().splitlines() == buf_b.getvalue().splitlines()
    assert stats_a.counts == stats_b.counts
