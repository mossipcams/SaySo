#!/usr/bin/env python3
"""Build an eval-directed, label-first SaySo synthetic training dataset.

Two pipelines share this entry point.

``--pipeline v3`` is the live one: ``generators/`` builds every row
deterministically from an entity graph, and nothing here does more than wire
the CLI to it.

The default is the older v1/v2 pipeline, which labels first
(:mod:`v2_scenarios`), renders the canonical envelope (:mod:`rendering`), then
asks a generator model for wording and an independent judge to score it
(:mod:`llm_curation`). ``run_pipeline`` below is the resumable orchestration of
those stages; the names re-exported here are the surface the eval and
supplement generators already import.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_curation import (  # noqa: E402
    DEFAULT_TRAIN_COUNT,
    _append_jsonl,
    _audit_selected,
    _behavior_key,
    _counts,
    _framed_utterance,
    _json_content,
    _judge_prompt,
    _normalized,
    _ranked,
    _read_jsonl,
    _run_batches,
    _sha256,
    _verbalizer_prompt,
    _write_jsonl,
    curate,
    framing_response_format,
    judge_batch,
    judge_resilient,
    judge_response_format,
    load_user_utterances,
    openai_complete,
    validate_utterance,
    verbalize_batch,
    verbalize_resilient,
)
from rendering import (  # noqa: E402
    _BANNED_UTTERANCE,
    _CLEAN_DIRECT_START,
    _call_id,
    _expand_template,
    _final_text,
    _protected_slots,
    _system_prompt,
    expand_utterance,
    render_example,
    request_seed,
    template_seed,
)
from v2_scenarios import (  # noqa: E402
    CATEGORY_WEIGHTS,
    _APOSTROPHE_NAMES,
    _AREAS,
    _DEVICES,
    _FLOORS,
    _GENERIC_NOUN_TO_KIND,
    _KIND_TO_GENERIC_NOUN,
    _PREFIXES,
    _SPECIAL_NAMES,
    _UNAVAILABLE_TYPE,
    _UNSUPPORTED_HINTS,
    _area_scenario,
    _control_call,
    _entities_of_kind_in_area,
    _entity,
    _expected_generic_in_sayso_area,
    _generic_no_area_hint,
    _home,
    _is_generic_no_area_hint,
    _make_entity,
    _single_action,
    _spec,
    _status_call,
    _stt_variant,
    build_specs,
    validate_spec,
)

# The home-specific recipe mixes the fetched home in at this rate unless the
# caller overrides it. Nonzero on purpose: a recipe that trains for one home and
# never shows the model that home is not a home-specific recipe. --synthetic-only
# is the explicit way to turn it off.
HOME_RECIPE_REAL_RATE = 0.10


def run_pipeline(
    *,
    out_dir: Path,
    count: int,
    seed: int,
    batch_size: int,
    generator_complete: Any = None,
    judge_complete: Any = None,
    generator_model: str,
    judge_model: str,
    min_count: int = DEFAULT_TRAIN_COUNT,
    max_count: int = DEFAULT_TRAIN_COUNT,
    stage: str = "all",
    excluded_utterances: set[str] | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Run resumable generation, judging, and curation stages."""
    if stage not in {"all", "generate", "judge", "curate"}:
        raise ValueError("unknown stage")
    if batch_size <= 0 or workers <= 0:
        raise ValueError("batch_size and workers must be positive")
    if generator_model.strip().casefold() == judge_model.strip().casefold():
        raise ValueError("judge model must differ from generator model")
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = out_dir / f"sayso_candidates_{count}.jsonl"
    judged_path = out_dir / f"sayso_judged_{count}.jsonl"
    rejected_path = out_dir / f"sayso_rejected_{count}.jsonl"
    report_path = out_dir / f"sayso_build_report_{count}.json"
    if report_path.exists() and stage in {"all", "curate"}:
        return json.loads(report_path.read_text(encoding="utf-8"))

    specs = build_specs(count, seed=seed)
    candidates = _read_jsonl(candidate_path)
    for row in candidates.values():
        if row.get("seed") != seed:
            raise ValueError("candidate checkpoint seed mismatch")
        if str(row.get("generator_model", "")).strip().casefold() == judge_model.strip().casefold():
            raise ValueError("judge model must differ from every candidate generator model")
    if stage in {"all", "generate"}:
        if generator_complete is None:
            raise ValueError("generator completion is required")
        missing = [spec for spec in specs if spec["candidate_id"] not in candidates]
        process = lambda batch: verbalize_resilient(batch, generator_complete)
        for completed in _run_batches(missing, batch_size, workers, process):
            for row in completed:
                row["generator_model"] = generator_model
            _append_jsonl(candidate_path, completed)
            candidates.update({row["candidate_id"]: row for row in completed})
            print(f"generated {len(candidates)}/{count}", flush=True)
    if len(candidates) != count:
        raise ValueError(f"candidate pool incomplete: {len(candidates)}/{count}")
    if stage == "generate":
        return {
            "candidate_count": len(candidates),
            "candidate_categories": _counts(list(candidates.values())),
            "generator_models": sorted({row["generator_model"] for row in candidates.values()}),
        }

    judged = _read_jsonl(judged_path)
    rejection_rows = _read_jsonl(rejected_path)
    for row in judged.values():
        if row.get("judge_model") != judge_model:
            raise ValueError("judge checkpoint model mismatch")
    if stage in {"all", "judge"}:
        if judge_complete is None:
            raise ValueError("judge completion is required")
        pending = [
            candidates[spec["candidate_id"]]
            for spec in specs
            if spec["candidate_id"] not in judged and spec["candidate_id"] not in rejection_rows
        ]
        process = lambda batch: judge_resilient(
                batch,
                judge_complete,
                generator_model=generator_model,
                judge_model=judge_model,
            )
        for accepted, rejected in _run_batches(pending, batch_size, workers, process):
            for row in accepted:
                row["judge_model"] = judge_model
            rejected_data = [
                {"candidate_id": candidate_id, "reason": reason, "stage": "judge"}
                for candidate_id, reason in rejected.items()
            ]
            _append_jsonl(judged_path, accepted)
            _append_jsonl(rejected_path, rejected_data)
            judged.update({row["candidate_id"]: row for row in accepted})
            rejection_rows.update({row["candidate_id"]: row for row in rejected_data})
            print(
                f"judged {len(judged) + len(rejection_rows)}/{count} "
                f"(accepted {len(judged)}, rejected {len(rejection_rows)})",
                flush=True,
            )
    if len(judged) + len(rejection_rows) != count:
        raise ValueError(
            f"judge pool incomplete: {len(judged) + len(rejection_rows)}/{count}"
        )
    if stage == "judge":
        return {
            "candidate_count": count,
            "judge_accepted": len(judged),
            "judge_rejected": len(rejection_rows),
        }

    selected, curation_drops = curate(
        list(judged.values()),
        min_count=min_count,
        max_count=max_count,
        excluded_utterances=excluded_utterances,
    )
    curated_path = out_dir / f"sayso_curated_{len(selected)}.jsonl"
    _write_jsonl(curated_path, [render_example(spec) for spec in selected])
    rejection_reasons = Counter(row["reason"] for row in rejection_rows.values())
    report = {
        "seed": seed,
        "generator_model": generator_model,
        "generator_models": sorted({row["generator_model"] for row in candidates.values()}),
        "judge_model": judge_model,
        "candidate_count": count,
        "candidate_categories": _counts(list(candidates.values())),
        "judge_accepted": len(judged),
        "judge_rejected": len(rejection_rows),
        "judge_rejection_reasons": dict(sorted(rejection_reasons.items())),
        "curation_drops": dict(sorted(curation_drops.items())),
        "curated_count": len(selected),
        "curated_categories": _counts(selected),
        "quality_thresholds": {"correctness": 4, "clarity": 4, "naturalness": 4},
        "excluded_prompt_count": len(excluded_utterances or set()),
        "audit": _audit_selected(selected),
        "files": {
            "candidates": candidate_path.name,
            "candidates_sha256": _sha256(candidate_path),
            "curated": curated_path.name,
            "curated_sha256": _sha256(curated_path),
        },
    }
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    return report


def render_for_trl(example: dict[str, Any]) -> dict[str, Any]:
    """TRL view: dict tool arguments and plain-string message content."""
    from adapters.schema import extract_text_content, normalize_tool_arguments

    rendered = deepcopy(example)
    messages: list[dict[str, Any]] = []
    for message in rendered.get("messages") or []:
        msg = dict(message)
        content = msg.get("content")
        if isinstance(content, list):
            msg["content"] = extract_text_content(content)
        elif content is None and msg.get("role") == "assistant" and msg.get("tool_calls"):
            msg["content"] = ""
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            calls: list[dict[str, Any]] = []
            for call in msg["tool_calls"]:
                rendered_call = dict(call)
                fn = dict(rendered_call.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    parsed = normalize_tool_arguments(args)
                    if parsed is not None:
                        fn["arguments"] = parsed
                rendered_call["function"] = fn
                calls.append(rendered_call)
            msg["tool_calls"] = calls
        messages.append(msg)
    rendered["messages"] = messages
    return rendered


def _deterministic_train_utterance(spec: dict[str, Any], excluded: set[str]) -> str | None:
    """Return a deterministic utterance that avoids quality-eval prompt overlap."""
    base = expand_utterance(spec)
    if _normalized(base) not in excluded:
        return base
    prefixes = ("please ", "hey, ", "could you ", "okay, ")
    suffixes = (" please", " for me", " right now", " thanks")
    candidates = [f"{prefix}{base}" for prefix in prefixes] + [f"{base}{suffix}" for suffix in suffixes]
    for candidate in candidates:
        if _normalized(candidate) in excluded:
            continue
        trial = deepcopy(spec)
        trial["utterance"] = candidate
        if validate_utterance(trial) is None:
            return candidate
    return None


def build_deterministic_train_examples(
    count: int = DEFAULT_TRAIN_COUNT,
    *,
    seed: int = 20260904,
    excluded_utterances: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Build label-first train rows via expand_utterance, excluding quality-eval prompts."""
    if count <= 0 or count % 100:
        raise ValueError("count must be a positive multiple of 100")
    excluded = {_normalized(text) for text in excluded_utterances or set()}
    quotas = {category: count * weight // 100 for category, weight in CATEGORY_WEIGHTS.items()}
    selected_by_category: dict[str, list[dict[str, Any]]] = {
        category: [] for category in quotas
    }
    pool_size = count
    seed_offset = 0
    while any(len(selected_by_category[category]) < quota for category, quota in quotas.items()):
        specs = build_specs(pool_size, seed=seed + seed_offset)
        for spec in specs:
            category = spec["category"]
            if len(selected_by_category[category]) >= quotas[category]:
                continue
            row = deepcopy(spec)
            utterance = _deterministic_train_utterance(row, excluded)
            if utterance is None:
                continue
            row["utterance"] = utterance
            if validate_utterance(row) is not None:
                continue
            selected_by_category[category].append(render_example(row))
        pool_size += 100
        seed_offset += 1
        if seed_offset > 50:
            raise ValueError(
                f"unable to collect balanced train rows without quality-eval overlap after {seed_offset} pools"
            )
    rows: list[dict[str, Any]] = []
    for category in CATEGORY_WEIGHTS:
        rows.extend(selected_by_category[category][: quotas[category]])
    if len(rows) != count:
        raise ValueError(f"expected {count} train rows, built {len(rows)}")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_jsonl(path, rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=("legacy", "v3"), default="legacy")
    parser.add_argument("--stage", choices=("all", "generate", "judge", "curate"), default="all")
    parser.add_argument("--count", type=int, default=DEFAULT_TRAIN_COUNT)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "datasets" / "synthetic_v2")
    parser.add_argument("--generator-url", default="http://192.168.1.140:8080/v1")
    parser.add_argument("--generator-model", default=None)
    parser.add_argument("--judge-url", default="http://192.168.1.140:8080/v1")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--min-count", type=int, default=DEFAULT_TRAIN_COUNT)
    parser.add_argument("--max-count", type=int, default=DEFAULT_TRAIN_COUNT)
    parser.add_argument("--exclude-prompts", type=Path)
    parser.add_argument("--stt-rate", type=float, default=0.15)
    parser.add_argument("--paraphrase", action="store_true", default=False)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--render-out", type=Path, default=None)
    parser.add_argument(
        "--real-home",
        type=Path,
        default=None,
        help="v3 only: home JSON from scripts/fetch_ha_home.py to mix in",
    )
    parser.add_argument(
        "--real-home-rate",
        type=float,
        default=0.0,
        help="v3 only: fraction of rows generated over the real home",
    )
    parser.add_argument(
        "--real-home-entity-cap",
        type=int,
        default=0,
        help="Max rows one real entity may be the target of (0 derives it)",
    )
    parser.add_argument(
        "--home-recipe",
        action="store_true",
        help="v3 only: the home-specific recipe. Requires --real-home and defaults "
             f"--real-home-rate to {HOME_RECIPE_REAL_RATE}",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="v3 only: explicitly disable real-home mixing, overriding --home-recipe",
    )
    parser.add_argument(
        "--allow-stale-home",
        action="store_true",
        help="v3 only: accept a home snapshot that Home Assistant's Assist exposure "
             "list was not applied to",
    )
    parser.add_argument("--negative-rate", type=float, default=None)
    parser.add_argument("--grounding-rate", type=float, default=None)
    parser.add_argument(
        "--discrimination-rate",
        type=float,
        default=None,
        help="v3 only: share of rows whose utterance describes the target entity "
             "instead of naming it (entity resolution; default 0.0)",
    )
    parser.add_argument("--max-absence-rate", type=float, default=None)
    args = parser.parse_args()

    if args.real_home_rate and not args.real_home:
        parser.error("--real-home-rate needs --real-home")
    if args.real_home and args.pipeline != "v3":
        parser.error("--real-home only applies to --pipeline v3")
    if args.home_recipe and args.synthetic_only:
        parser.error("--home-recipe and --synthetic-only are mutually exclusive")
    if args.home_recipe and not args.real_home:
        parser.error("--home-recipe needs --real-home")

    if args.pipeline == "v3":
        from generators.config import GeneratorConfig
        from generators.pipeline import run_generation, write_jsonl, write_manifest
        from generators.real_home import TESTABLE_EXPOSURE_SOURCES, require_exposure_source

        real_home_rate = args.real_home_rate
        if args.home_recipe and not real_home_rate:
            real_home_rate = HOME_RECIPE_REAL_RATE
        if args.real_home and not args.synthetic_only and not args.allow_stale_home:
            require_exposure_source(args.real_home, allowed=TESTABLE_EXPOSURE_SOURCES)

        out_path = args.out_dir / "synthetic_v3_train.jsonl" if args.out_dir.is_dir() else args.out_dir
        render_path = args.render_out or out_path.with_name(f"{out_path.stem}_render.jsonl")
        overrides = {
            key: value
            for key, value in (
                ("negative_rate", args.negative_rate),
                ("grounding_rate", args.grounding_rate),
                ("discrimination_rate", args.discrimination_rate),
                ("max_absence_rate", args.max_absence_rate),
            )
            if value is not None
        }
        config = GeneratorConfig(
            **overrides,
            count=args.count,
            seed=args.seed,
            output_path=out_path,
            manifest_path=args.manifest or out_path.with_suffix(".manifest.json"),
            stt_noise_rate=args.stt_rate,
            paraphrase_enabled=args.paraphrase,
            token_budget=args.token_budget,
            exclude_prompts_path=args.exclude_prompts,
            real_home_path=args.real_home,
            real_home_rate=real_home_rate,
            real_home_entity_cap=args.real_home_entity_cap,
            synthetic_only=args.synthetic_only,
        )
        result = run_generation(config)
        rendered = [render_for_trl(row) for row in result["rows"]]
        write_jsonl(config.output_path, result["rows"])
        write_jsonl(render_path, rendered)
        stats = {
            **result["stats"],
            "render_path": str(render_path),
            "render_rows": len(rendered),
        }
        write_manifest(config.manifest_path, stats)
        print(json.dumps(stats, indent=2, default=str))
        return 0

    if not args.generator_model or not args.judge_model:
        parser.error("legacy pipeline requires --generator-model and --judge-model")

    generator_key = os.environ.get("SAYSO_GENERATOR_API_KEY", "")
    judge_key = os.environ.get("SAYSO_JUDGE_API_KEY", generator_key)
    generator_complete = None
    judge_complete = None
    if args.stage in {"all", "generate"}:
        generator_complete = lambda prompt: openai_complete(
            prompt,
            base_url=args.generator_url,
            model=args.generator_model,
            api_key=generator_key,
            temperature=0.7,
            max_tokens=max(256, len(json.loads(prompt.split("ITEMS:\n", 1)[1])) * 12),
            response_format=framing_response_format(len(json.loads(prompt.split("ITEMS:\n", 1)[1]))),
        )
    if args.stage in {"all", "judge"}:
        judge_complete = lambda prompt: openai_complete(
            prompt,
            base_url=args.judge_url,
            model=args.judge_model,
            api_key=judge_key,
            temperature=0.0,
            max_tokens=max(64, len(json.loads(prompt.split("ITEMS:\n", 1)[1])) * 8),
            response_format=judge_response_format(len(json.loads(prompt.split("ITEMS:\n", 1)[1]))),
        )
    report = run_pipeline(
        out_dir=args.out_dir,
        count=args.count,
        seed=args.seed,
        batch_size=args.batch_size,
        generator_complete=generator_complete,
        judge_complete=judge_complete,
        generator_model=args.generator_model,
        judge_model=args.judge_model,
        min_count=args.min_count,
        max_count=args.max_count,
        stage=args.stage,
        excluded_utterances=load_user_utterances(args.exclude_prompts) if args.exclude_prompts else None,
        workers=args.workers,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
