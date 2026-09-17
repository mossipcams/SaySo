"""One behavioral execution pipeline.

For each case: build the production area context from the household and
utterance (never from the expected answer), render the production system
prompt, compile the household's production tool catalog (minus only the
capability an unavailable case withholds), ask the model through the
configured adapter, parse with the production parser, validate with the
production contract check, and score with the shared scorer.

Evaluation never executes Home Assistant actions.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import urllib.error
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from evals.cases import (
    Case,
    fingerprint_cases,
    load_gates,
    load_home,
    production_contract_fingerprint,
    select_cases,
)
from evals.outcomes import (
    PRODUCTION_MAX_OUTPUT_TOKENS,
    PRODUCTION_TEMPERATURE,
    CaseResult,
    Expectation,
    ResponseType,
    read_completion,
)
from evals.scorer import check_gates, score, summarize

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"


class ModelAdapter(Protocol):
    name: str

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any: ...


@contextmanager
def training_path():
    """Expose ``training/generators`` without shadowing the repo ``tests`` package."""
    path = str(ROOT / "training")
    added = path not in sys.path
    if added:
        sys.path.append(path)
    try:
        yield
    finally:
        if added:
            sys.path.remove(path)


def production_catalog(home: dict[str, Any], *, removed_tools: list[str] | None = None) -> list[dict[str, Any]]:
    with training_path():
        from generators.tools import production_catalog as _production_catalog  # noqa: PLC0415

    return _production_catalog(home, removed_tools=removed_tools or [])


def render_case(case: Case) -> dict[str, Any]:
    """Everything the model and the scorer see for one case."""
    with training_path():
        from generators.context import area_context_for, serialize_context  # noqa: PLC0415

    home = load_home(case.household)
    removed = (case.unavailable or {}).get("removed_tools") or []
    expected = case.expected
    area = area_context_for(home, case.utterance)
    return {
        "messages": [
            {"role": "system", "content": serialize_context(home, case.utterance)},
            {"role": "user", "content": case.utterance},
        ],
        "tools": production_catalog(home, removed_tools=removed),
        "exposed_domains": frozenset(entity["domain"] for entity in home["entities"]),
        "area_context": area.as_dict(),
        "expectation": Expectation(
            category=case.category,
            response_type=ResponseType(expected["response_type"]),
            calls=tuple(expected["calls"]),
            forbidden_entities=tuple(expected["forbidden_entities"]),
            target_area=area.target_area,
        ),
    }


def evaluate(cases: list[Case], adapter: ModelAdapter) -> list[CaseResult]:
    """Score every case through the shared parser, validator, and scorer."""
    results: list[CaseResult] = []
    for case in cases:
        rendered = render_case(case)
        try:
            raw = adapter.complete(rendered["messages"], rendered["tools"])
            transport_error = None
        except (urllib.error.URLError, TimeoutError, OSError, ConnectionError) as err:
            raw = {"transport_error": str(err)}
            transport_error = str(err)
        turn = read_completion(raw, rendered["tools"], rendered["exposed_domains"])
        result = score(case.id, rendered["expectation"], turn)
        if transport_error:
            result.flags.append("transport_error")
        results.append(result)
    return results


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_run(
    *,
    cases: list[Case],
    results: list[CaseResult],
    adapter: ModelAdapter,
    suite: str | None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    dest = RESULTS_DIR / run_id
    dest.mkdir(parents=True, exist_ok=True)
    summary = summarize(results)
    gates = load_gates(suite) if suite else None
    failures = check_gates(summary, gates, expected_count=len(cases)) if gates else []
    metadata = {
        "run_id": run_id,
        "suite": suite,
        "adapter": adapter.name,
        "checkpoint": getattr(adapter, "model", None) or (extra_metadata or {}).get("checkpoint"),
        "code_revision": _git_revision(),
        "case_hash": fingerprint_cases(cases),
        "suite_hash": hashlib.sha256("\n".join(case.id for case in cases).encode()).hexdigest(),
        "production_contract_fingerprint": production_contract_fingerprint(),
        "decoding": {
            "temperature": PRODUCTION_TEMPERATURE,
            "max_output_tokens": PRODUCTION_MAX_OUTPUT_TOKENS,
        },
        "created_at": datetime.now(UTC).isoformat(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    if hasattr(adapter, "server"):
        metadata["server"] = adapter.server
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {
        "summary": summary,
        "promotion": {"passed": not failures, "failures": failures} if gates else None,
    }
    (dest / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    outcomes = dest / "outcomes.jsonl"
    with outcomes.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.as_dict(), ensure_ascii=False) + "\n")
    raw_dir = dest / "raw"
    raw_dir.mkdir()
    for result in results:
        (raw_dir / f"{result.case_id}.json").write_text(
            json.dumps(result.turn.raw, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return dest
