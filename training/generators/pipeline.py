"""Main deterministic generation pipeline."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from generators.capability_registry import CAPABILITIES, registry_summary
from generators.config import GeneratorConfig
from generators.duplicates import DuplicateTracker
from generators.labels import render_example, scenario_to_spec
from generators.paraphrase import load_paraphraser
from generators.real_home import derive_entity_cap, load_real_home
from generators.sampling import QuotaTracker
from generators.scenarios import build_scenario, pick_robustness, pick_targeting
from generators.stats import empty_stats, finalize_stats, record_accept, record_reject
from generators.stt_noise import apply_stt_noise
from generators.utterances import expand_utterance, request_seed_from_spec
from generators.validate import validate_row


def _unique_no_action_hint(spec: dict[str, Any], rng: random.Random) -> str:
    """Generate no-action hints that avoid recipe-lock golden utterances."""
    home = spec.get("home", {})
    area = home.get("sayso_entity_area", "room")
    templates = (
        f"set the {area} thermostat to {rng.randint(60, 75)} degrees",
        f"start the robot vacuum in the {area}",
        f"play jazz in the {area}",
        f"add eggs to the {area} shopping list",
        f"run the {area} goodnight scene",
        f"start a {rng.randint(5, 20)} minute {area} timer",
    )
    return rng.choice(templates)


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
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


@lru_cache(maxsize=1)
def _quality_eval_prompts() -> frozenset[str]:
    """Normalized gold, shadow, and recipe-lock prompts, none of which may be trained on."""
    try:
        from evals.v3_quality import excluded_train_prompts

        return frozenset(excluded_train_prompts())
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
    real_home_targets: Counter[str] | None = None,
    real_home_entity_cap: int = 0,
) -> tuple[dict[str, Any] | None, str | None]:
    capability = slot["capability"]
    operation = slot["operation"]
    cap = CAPABILITIES[capability]
    robustness = pick_robustness(rng, config.ordinary_rate)
    targeting = pick_targeting(cap, rng, robustness)
    home_size = slot["home_size"]
    if robustness == "large_home":
        home_size = max(home_size, 64)

    # ponytail: deep-copied per row because build_scenario mutates the home.
    # Cheap next to utterance expansion; cache the split lists if it ever isn't.
    home = None
    if config.real_home_path and rng.random() < config.real_home_rate:
        home = load_real_home(config.real_home_path, split="train")

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
        home=home,
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

    # A rejected real-home row is retried, and the retry re-rolls the real/synthetic
    # draw, so capping converts surplus rows for one entity into synthetic rows
    # rather than shrinking the corpus.
    targets = spec.get("target_names") or []
    if home is not None and real_home_entity_cap and real_home_targets is not None:
        if any(real_home_targets[name] >= real_home_entity_cap for name in targets):
            return None, "real_home_entity_cap"

    try:
        row = render_example(spec)
    except ValueError as exc:
        return None, str(exc)

    dup_tracker.record(spec)
    if home is not None and real_home_targets is not None:
        real_home_targets.update(targets)
    return row, None


def run_generation(config: GeneratorConfig) -> dict[str, Any]:
    """Generate accepted training rows up to config.count."""
    rng = random.Random(config.seed)
    quota = QuotaTracker(config.count, config.seed, config.tier_proportions)

    excluded = _load_excluded_prompts(config.exclude_prompts_path)
    dup_tracker = DuplicateTracker(near_limit=config.near_duplicate_limit)
    real_home_targets: Counter[str] = Counter()
    real_home_entity_cap = config.real_home_entity_cap
    if config.real_home_path and not real_home_entity_cap:
        real_home_entity_cap = derive_entity_cap(
            config.count,
            config.real_home_rate,
            len(load_real_home(config.real_home_path, split="train")["entities"]),
        )
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
            real_home_targets=real_home_targets,
            real_home_entity_cap=real_home_entity_cap,
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
    if config.real_home_path:
        report["real_home"] = {
            "entity_cap": real_home_entity_cap,
            "rows": sum(real_home_targets.values()),
            "entities_used": len(real_home_targets),
            "most_common": real_home_targets.most_common(5),
        }
    report["registry"] = registry_summary()
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
