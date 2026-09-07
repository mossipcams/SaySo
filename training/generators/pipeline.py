"""Main deterministic generation pipeline."""

from __future__ import annotations

import json
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from generators.capability_registry import CAPABILITIES, registry_summary
from generators.config import GeneratorConfig
from generators.duplicates import DuplicateTracker
from generators.labels import render_example, scenario_to_spec
from generators.gold import target_names_from_expected
from generators.homes import make_entity, _ENTITY_TEMPLATES
from generators.tools import build_call_for_operation
from generators.paraphrase import load_paraphraser
from generators.sampling import QuotaTracker
from generators.scenarios import build_scenario, pick_robustness, pick_targeting
from generators.stats import empty_stats, finalize_stats, record_accept, record_reject
from generators.stt_noise import apply_stt_noise
from generators.utterances import expand_utterance, request_seed_from_spec, vary_training_utterance
from generators.validate import validate_row


def _unique_no_action_hint(spec: dict[str, Any], rng: random.Random) -> str:
    """Describe the actual blocked request, never an unrelated random action."""
    requested = spec["expected"].get("requested")
    if requested:
        return request_seed_from_spec({
            **spec, "expected": requested,
            "target_names": target_names_from_expected(requested),
        })
    capability = spec["capability"]
    area = spec["home"]["sayso_entity_area"]
    entity = None
    if capability != "timers":
        entity = make_entity(
            name=f"the {area} {_ENTITY_TEMPLATES[capability][0].lower()}",
            capability=capability, area=area, floor="Main Floor", rng=rng,
        )
    call = build_call_for_operation(entity, capability, spec["operation"], rng, area=area)
    return request_seed_from_spec({
        "expected": {"kind": "action", "calls": [call]},
        "target_names": [entity["name"]] if entity else [""],
    })


def _ambiguous_hint(spec: dict[str, Any], rng: random.Random) -> str:
    area = spec.get("home", {}).get("sayso_entity_area", "kitchen")
    cap = spec.get("capability", "lights")
    nouns = {
        "lights": "light",
        "fans": "fan",
        "switches": "outlet",
        "covers": "blinds",
        "locks": "door",
    }
    noun = nouns.get(cap, "device")
    return f"turn on the {area.casefold()} {noun}"


def _load_excluded_prompts(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    excluded: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            for msg in row.get("messages", []):
                if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                    excluded.add(msg["content"].casefold())
        except json.JSONDecodeError:
            continue
    return excluded


def _normalize_prompt(text: str) -> str:
    """Match excluded_train_prompts(): punctuation-insensitive, so "joe's" == "joe s"."""
    prompt = " ".join(re.findall(r"[a-z0-9]+", text.casefold()))
    prompt = re.sub(r"^(?:(?:please|can you|could you|tell me) )+", "", prompt)
    return re.sub(r"(?: for me)+$", "", prompt)


@lru_cache(maxsize=1)
def _quality_eval_prompts() -> frozenset[str]:
    """Normalized gold, shadow, and recipe-lock prompts, none of which may be trained on."""
    try:
        from evals.v3_quality import excluded_train_prompts

        return frozenset(_normalize_prompt(prompt) for prompt in excluded_train_prompts())
    except ImportError:
        try:
            from evals.recipe_lock import locked_specs

            return frozenset(_normalize_prompt(spec["utterance"]) for spec in locked_specs())
        except ImportError:
            return frozenset()


def _check_quality_eval_overlap(utterance: str) -> bool:
    """Reject contamination from golden, shadow, and recipe-lock eval utterances."""
    return _normalize_prompt(utterance) in _quality_eval_prompts()


def generate_row(
    slot: dict[str, Any],
    config: GeneratorConfig,
    rng: random.Random,
    *,
    excluded: set[str],
    dup_tracker: DuplicateTracker,
    attempt: int = 0,
    stt_remaining: int = 0,
    rows_remaining: int = 1,
) -> tuple[dict[str, Any] | None, str | None]:
    capability = slot["capability"]
    operation = slot["operation"]
    cap = CAPABILITIES[capability]
    robustness = pick_robustness(rng, config.ordinary_rate)
    targeting = pick_targeting(cap, rng, robustness)
    home_size = slot["home_size"]
    if robustness == "large_home":
        home_size = max(home_size, 64)

    scenario = build_scenario(
        index=slot["index"] + attempt * 10000,
        seed=config.seed ^ (attempt << 16),
        capability=capability,
        operation=operation,
        home_size=home_size,
        targeting=targeting,
        robustness=robustness,
        split=config.split,
        attempt=attempt,
    )

    spec = scenario_to_spec(scenario)
    expected = spec.get("expected") or {}
    if expected.get("kind") == "no_action":
        spec["request_hint"] = _unique_no_action_hint(spec, rng)
    elif robustness == "ambiguity":
        spec["request_hint"] = _ambiguous_hint(spec, rng)
    spec["utterance"] = expand_utterance({**spec, "category": "clean_direct"})
    if expected.get("kind") in {"action", "status"} and spec.get("target_names"):
        primary = spec["target_names"][0]
        if primary.casefold() not in spec["utterance"].casefold():
            spec["utterance"] = request_seed_from_spec(spec)

    # Check before style/noise transforms too: variants of held-out requests stay held out.
    if _check_quality_eval_overlap(spec["utterance"]):
        return None, "quality_eval_overlap"

    if (
        stt_remaining > 0
        and rows_remaining > 0
        and rng.random() < stt_remaining / rows_remaining
    ):
        corrupted, kind = apply_stt_noise(
            spec["utterance"],
            rng,
            target_names=spec.get("target_names"),
            force_transform=True,
        )
        if kind:
            trial = dict(spec)
            trial["utterance"] = corrupted
            trial["stt_corruption"] = kind
            if validate_row(trial, token_budget=config.token_budget) is None:
                spec = trial

    # Apply the same casing distribution to every label, including refusals.
    utterance = vary_training_utterance(spec["utterance"], rng)
    spec["utterance"] = (
        utterance.lower() if rng.random() < 0.5
        else utterance[:1].upper() + utterance[1:]
    )
    if spec["utterance"].casefold() in excluded:
        return None, "excluded_prompt"
    if _check_quality_eval_overlap(spec["utterance"]):
        return None, "quality_eval_overlap"

    reason = validate_row(spec, token_budget=config.token_budget)
    if reason:
        return None, reason

    reject = dup_tracker.would_reject(spec)
    if reject:
        return None, reject

    try:
        row = render_example(spec)
    except ValueError as exc:
        return None, str(exc)

    dup_tracker.record(spec)
    return row, None


def run_generation(config: GeneratorConfig) -> dict[str, Any]:
    """Generate accepted training rows up to config.count."""
    rng = random.Random(config.seed)
    quota = QuotaTracker(config.count, config.seed, config.tier_proportions)

    excluded = _load_excluded_prompts(config.exclude_prompts_path)
    dup_tracker = DuplicateTracker(near_limit=config.near_duplicate_limit)
    stats = empty_stats()
    accepted: list[dict[str, Any]] = []
    semantic_ids: set[str] = set()
    attempts = 0
    max_attempts = config.max_attempts()

    load_paraphraser(config.paraphrase_enabled)
    stt_target = int(round(config.count * config.stt_noise_rate))

    while not quota.is_complete() and attempts < max_attempts:
        slot = quota.next_slot()
        row, reason = generate_row(
            slot,
            config,
            rng,
            excluded=excluded,
            dup_tracker=dup_tracker,
            attempt=attempts,
            stt_remaining=max(0, stt_target - stats["stt_corrupted"]),
            rows_remaining=max(1, config.count - quota.accepted_total()),
        )
        attempts += 1
        if row is None:
            record_reject(stats, reason or "unknown")
            continue
        sem = row.get("metadata", {}).get("semantic_id")
        if sem:
            semantic_ids.add(sem)
        quota.record_accept(row)
        accepted.append(row)
        record_accept(stats, row)

    if not quota.is_complete():
        raise RuntimeError(
            f"failed to meet accepted-row quota: accepted {quota.accepted_total()}/{config.count} "
            f"after {attempts} attempts; shortfall={quota.shortfall()}; "
            f"rejections={dict(stats['rejection_reasons'])}"
        )

    quota.verify_complete()
    report = finalize_stats(stats, semantic_ids, quota_summary=quota.summary())
    report["requested_stt_rate"] = config.stt_noise_rate
    report["achieved_stt_rate"] = round(stats["stt_corrupted"] / max(stats["accepted"], 1), 4)
    report["requested"] = config.count
    report["attempts"] = attempts
    report["config"] = config.to_dict()
    report["registry"] = registry_summary()
    from generators.audit import audit_rows

    report["quality_audit"] = audit_rows(accepted, expected_count=config.count)
    return {"rows": accepted, "stats": report}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {k: v for k, v in report.items() if k != "rows"}
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
