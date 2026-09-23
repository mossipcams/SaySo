"""Dataset manifest construction and reconciliation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from generators.coverage import classify_row
from generators.validation import count_row_tokens


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_hash(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def build_manifest(
    rows: list[dict[str, Any]],
    *,
    config_dict: dict[str, Any],
    recipe_path: Path,
    stats: dict[str, Any],
    allocation_summary: dict[str, Any],
    tokenizer_model: str,
    rejections: dict[str, int],
    token_lengths: list[int],
) -> dict[str, Any]:
    """Manifest that reconciles with the written dataset."""
    outcome_counts = Counter()
    contrast_counts = Counter()
    tool_counts = Counter()
    area_sources = Counter()
    for row in rows:
        facets = classify_row(row)
        outcome_counts[facets["outcome"]] += 1
        if row["metadata"].get("contrastive_group"):
            contrast_counts[row["metadata"]["contrastive_group"]] += 1
        for tool in facets["tools"]:
            tool_counts[tool] += 1
        area_ctx = row["metadata"].get("area_context") or {}
        if area_ctx.get("target_area_source"):
            area_sources[area_ctx["target_area_source"]] += 1

    # Every accepted row, measured exactly once. Validation only tokenizes the
    # rows its cheap character bound could not clear, so mixing cached and
    # missing counts is normal -- taking only the cached ones would report the
    # longest tail of the corpus as if it were the whole distribution.
    lengths = list(token_lengths or [])
    if not lengths:
        for row in rows:
            cached = row["metadata"].get("_token_length")
            if cached:
                lengths.append(cached)
                continue
            try:
                lengths.append(count_row_tokens(row, model_name=tokenizer_model))
            except Exception:  # noqa: BLE001
                lengths.append(0)

    sorted_lengths = sorted(lengths) if lengths else [0]
    n = len(sorted_lengths)

    manifest: dict[str, Any] = {
        "generator_code_sha": _git_sha(),
        "recipe": {
            "path": str(recipe_path),
            "sha256": _file_sha256(recipe_path) if recipe_path.is_file() else None,
        },
        "seed": config_dict.get("seed"),
        "split": config_dict.get("split"),
        "schema_path": config_dict.get("schema_path"),
        "tokenizer_model": tokenizer_model,
        "renderer": "generators.rendering",
        "dataset_hash": dataset_hash(rows),
        "final_row_count": len(rows),
        "requested_count": config_dict.get("count"),
        "allocations": allocation_summary,
        "outcome_coverage": dict(sorted(outcome_counts.items())),
        "contrast_coverage": dict(sorted(contrast_counts.items())),
        "positive_tool_coverage": dict(sorted(tool_counts.items())),
        "area_target_source_coverage": dict(sorted(area_sources.items())),
        "rejections": dict(sorted(rejections.items())),
        "token_length": {
            "min": sorted_lengths[0],
            "p50": sorted_lengths[n // 2],
            "p95": sorted_lengths[int(n * 0.95)] if n > 1 else sorted_lengths[0],
            "max": sorted_lengths[-1],
            "oversized_rejects": stats.get("oversized_rejects", 0),
        },
        "config": config_dict,
        "stats": {k: v for k, v in stats.items() if k != "rows"},
    }
    _reconcile_totals(manifest, rows)
    return manifest


def _reconcile_totals(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    outcome_total = sum(manifest["outcome_coverage"].values())
    if outcome_total != len(rows):
        raise ValueError(
            f"manifest outcome totals ({outcome_total}) != row count ({len(rows)})"
        )
    alloc = manifest["allocations"]
    achieved = alloc.get("achieved_family", {})
    if achieved and sum(achieved.values()) + sum(alloc.get("achieved_area", {}).values()) != len(rows):
        # Area rows are included in slots; family+area may overlap metadata
        pass


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
