"""Main deterministic generation pipeline."""

from __future__ import annotations

import copy
import json
import random
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from generators.capability_registry import (
    CAPABILITIES,
    SupportLevel,
    operation_spec,
    registry_summary,
)
from generators.config import GeneratorConfig
from generators.coverage import classify_row
from generators.duplicates import DuplicateTracker, pair_hash
from generators.grounding import pick_variant, required_training_variants
from generators.labels import render_example, scenario_to_spec
from generators.gold import target_names_from_expected
from generators.homes import make_entity, _ENTITY_TEMPLATES
from generators.tools import build_call_for_operation
from generators.paraphrase import load_paraphraser
from generators.real_home import derive_entity_cap, load_real_home
from generators.sampling import QuotaTracker
from generators.scenarios import build_scenario, pick_robustness, pick_targeting
from generators.stats import empty_stats, finalize_stats, record_accept, record_reject
from generators.stt_noise import apply_stt_noise
from generators.utterances import (
    apply_generic_wording,
    expand_utterance,
    request_seed_from_spec,
    vary_training_utterance,
)
from generators.validate import validate_row


# Operations Home Assistant supplies no tool for. They have no call to render, so
# the request they refuse has to be written out; the refusal itself still comes
# from the registry (SupportLevel.UNAVAILABLE), not from this table.
_UNAVAILABLE_REQUESTS: dict[tuple[str, str], str] = {
    ("lawn_mowers", "control"): "start mowing the lawn with {name}",
    ("todo_lists", "control"): "add milk to {name}",
    ("buttons", "control"): "press {name}",
    ("covers", "set_position"): "set {name} to 40 percent open",
    ("scripts", "query_state"): "what is the status of {name}",
}


# Requiring every grounding contrast family needs slack, not parity. A family
# lands only when a slot of its capability/operation comes up *and* that bucket
# still has room for its outcome; the refusal families (`moved`, `incapable`)
# additionally compete for a bucket's small negative allowance. At parity a
# 400-row run fails closed on a gate it was never large enough to meet.
GROUNDING_FAMILY_SLACK = 2


def _unique_no_action_hint(spec: dict[str, Any], rng: random.Random) -> str:
    """Describe the actual blocked request, never an unrelated random action."""
    requested = spec["expected"].get("requested")
    if requested:
        return request_seed_from_spec({
            **spec, "expected": requested,
            "target_names": target_names_from_expected(requested),
        })
    capability = spec["capability"]
    operation = spec["operation"]
    area = spec["home"]["sayso_entity_area"]
    entity = None
    if capability != "timers":
        entity = make_entity(
            name=f"the {area} {'routine' if capability == 'scripts' else _ENTITY_TEMPLATES[capability][0].lower()}",
            capability=capability, area=area, floor="Main Floor", rng=rng,
        )
    template = _UNAVAILABLE_REQUESTS.get((capability, operation))
    if template:
        spec["linguistics"] = [{"source": "sayso_fallback", "intent": f"{capability}.{operation}"}]
        return template.format(name=entity["name"] if entity else "it")
    call = build_call_for_operation(entity, capability, operation, rng, area=area)
    return request_seed_from_spec({
        "expected": {"kind": "action", "calls": [call]},
        "target_names": [entity["name"]] if entity else [""],
        "phrasing_seed": spec.get("phrasing_seed"),
    })


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

        prompts = {_normalize_prompt(prompt) for prompt in excluded_train_prompts()}
    except ImportError:
        try:
            from evals.recipe_lock import locked_specs

            prompts = {_normalize_prompt(spec["utterance"]) for spec in locked_specs()}
        except ImportError:
            prompts = set()

    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "realistic_eval_20260908_v2.json"
    for case in json.loads(fixture.read_text(encoding="utf-8"))["cases"]:
        prompts.update(_normalize_prompt(message["content"]) for message in case["messages"]
                       if message["role"] == "user")
    return frozenset(prompts)


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
    real_home_names: frozenset[str] = frozenset(),
    real_home_usage: Counter[str] | None = None,
    quota: Any | None = None,
    grounding_required: bool = False,
    real_home_selected: bool | None = None,
    grounding_variants: list[dict[str, Any]] | None = None,
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
    real_home_row = False
    grounding_family = None
    inject_missing = True
    select_real_home = (
        rng.random() < config.real_home_rate
        if real_home_selected is None else real_home_selected
    )
    # Rotate even while forcing the missing families: two of them can share one
    # capability/operation pool, and a fixed index would keep retrying the first
    # until it lands, leaving the second unreachable.
    variant = (
        pick_variant(capability, operation, slot["index"] + attempt, variants=grounding_variants)
        if grounding_variants else None
    )
    if variant is not None:
        # A still-missing contrast family outranks the real-home draw: real rows
        # are a rate that self-corrects across the run, a missing family is a gate.
        pass
    elif config.real_home_path and select_real_home:
        home = load_real_home(config.real_home_path, split="train")
        real_home_row = True
    elif config.grounding_rate and (grounding_required or rng.random() < config.grounding_rate):
        variant = pick_variant(capability, operation, slot["index"] + attempt)
    if variant is not None:
        home = copy.deepcopy(variant["home"])
        targeting = variant["targeting"]
        robustness = variant["robustness"]
        grounding_family = variant["family"]
        inject_missing = variant.get("inject_missing", True)

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
        inject_missing=inject_missing,
        # Only the real home reuses its entities across rows, so only it needs the
        # least-used tie-break; synthetic homes are fresh each row.
        target_usage=real_home_usage if real_home_row else None,
    )
    if grounding_family:
        scenario["phrasing_seed"] = variant["phrasing_seed"]

    spec = scenario_to_spec(scenario)
    expected = spec.get("expected") or {}
    if expected.get("kind") == "no_action":
        spec["request_hint"] = _unique_no_action_hint(spec, rng)
    elif robustness == "ambiguity":
        apply_generic_wording(spec)
    spec["utterance"] = expand_utterance(spec)

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

    # One semantic scenario may appear more than once, so the row id is the
    # scenario, the utterance+home pair, and how many rows already share that
    # pair. Deterministic because generation is sequential. semantic_id keeps
    # identifying the scenario itself.
    spec["candidate_id"] = (
        f"{spec['semantic_id']}_{pair_hash(spec['utterance'], spec['home'])[:8]}"
        f"_{dup_tracker.occurrences(spec)}"
    )

    # A rejected real-home row is retried, and the retry re-rolls the real/synthetic
    # draw, so capping converts surplus rows for one entity into synthetic rows
    # rather than shrinking the corpus.
    targets = spec.get("target_names") or []
    if real_home_row and real_home_entity_cap and real_home_targets is not None:
        if any(real_home_targets[name] >= real_home_entity_cap for name in targets):
            return None, "real_home_entity_cap"

    try:
        row = render_example(spec)
    except ValueError as exc:
        return None, str(exc)

    row["metadata"]["real_home"] = real_home_row
    row["metadata"]["grounding_family"] = grounding_family
    # Ask the quota before recording the row anywhere: a bucket that is already
    # full must not consume the duplicate tracker's budget for the next attempt.
    if quota is not None:
        reason = quota.wants(classify_row(row))
        if reason:
            return None, reason
    dup_tracker.record(spec)
    if real_home_row and real_home_targets is not None:
        real = [name for name in targets if not real_home_names or name in real_home_names]
        real_home_targets.update(real)
        if real_home_usage is not None:
            real_home_usage.update(real)
    return row, None


def run_generation(config: GeneratorConfig) -> dict[str, Any]:
    """Generate accepted training rows up to config.count."""
    rng = random.Random(config.seed)
    quota = QuotaTracker(
        config.count, config.seed, config.tier_proportions, negative_rate=config.negative_rate
    )

    excluded = _load_excluded_prompts(config.exclude_prompts_path)
    dup_tracker = DuplicateTracker(near_limit=config.near_duplicate_limit)
    real_home_targets: Counter[str] = Counter()
    # Per operation, so "the TV has had three turn_on rows" cannot crowd out its
    # first pause row. The flat counter above still backs the entity cap.
    real_home_usage: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    real_home_names: frozenset[str] = frozenset()
    real_home_entity_cap = config.real_home_entity_cap
    if config.real_home_path:
        real_entities = load_real_home(config.real_home_path, split="train")["entities"]
        # build_scenario injects an entity for a capability the real home lacks;
        # that name is synthetic, so it must not inflate the real-entity counts.
        real_home_names = frozenset(entity["name"] for entity in real_entities)
        if not real_home_entity_cap:
            real_home_entity_cap = derive_entity_cap(
                config.count, config.real_home_rate, len(real_entities)
            )
    stats = empty_stats()
    accepted: list[dict[str, Any]] = []
    semantic_ids: set[str] = set()
    real_home_rows = 0
    grounding_rows: Counter[str] = Counter()
    required_grounding = required_training_variants()
    grounding_missing = (
        {variant["family"]: variant for variant in required_grounding}
        if config.count * config.grounding_rate
        >= GROUNDING_FAMILY_SLACK * len(required_grounding)
        else {}
    )
    real_home_target = int(round(config.count * config.real_home_rate))
    balance_real_home = bool(
        config.real_home_path and config.real_home_rate and config.real_home_entity_cap == 0
    )
    attempts = 0
    max_attempts = config.max_attempts()

    load_paraphraser(config.paraphrase_enabled)
    stt_target = int(round(config.count * config.stt_noise_rate))

    while not quota.is_complete() and attempts < max_attempts:
        slot = quota.next_slot()
        rows_remaining = max(1, config.count - quota.accepted_total())
        real_home_selected = None
        if balance_real_home:
            real_home_remaining = max(0, real_home_target - real_home_rows)
            real_home_selected = rng.random() < real_home_remaining / rows_remaining
        row, reason = generate_row(
            slot,
            config,
            rng,
            excluded=excluded,
            dup_tracker=dup_tracker,
            attempt=attempts,
            stt_remaining=max(0, stt_target - stats["stt_corrupted"]),
            rows_remaining=rows_remaining,
            real_home_targets=real_home_targets,
            real_home_entity_cap=real_home_entity_cap,
            real_home_names=real_home_names,
            real_home_usage=real_home_usage[(slot["capability"], slot["operation"])],
            quota=quota,
            grounding_required=config.grounding_rate > 0 and not grounding_rows,
            real_home_selected=real_home_selected,
            grounding_variants=list(grounding_missing.values()) if grounding_missing else None,
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
        real_home_rows += bool(row["metadata"].get("real_home"))
        if row["metadata"].get("grounding_family"):
            family = row["metadata"]["grounding_family"]
            grounding_rows[family] += 1
            grounding_missing.pop(family, None)

    if not quota.is_complete():
        raise RuntimeError(
            f"failed to meet accepted-row quota: accepted {quota.accepted_total()}/{config.count} "
            f"after {attempts} attempts; shortfall={quota.shortfall()}; "
            f"missing_grounding={sorted(grounding_missing)}; "
            f"rejections={dict(stats['rejection_reasons'])}"
        )

    quota.verify_complete()
    if grounding_missing:
        raise RuntimeError(
            f"missing required grounding families: {sorted(grounding_missing)}"
        )
    # Balance within each label so small refusal samples cannot acquire a case shortcut by chance.
    for has_calls in (False, True):
        users = [next(message for message in row["messages"] if message["role"] == "user")
                 for row in accepted if any(message.get("tool_calls") for message in row["messages"]) == has_calls]
        rng.shuffle(users)
        for index, user in enumerate(users):
            text = user["content"]
            user["content"] = text.lower() if index % 2 == 0 else text[:1].upper() + text[1:]
    report = finalize_stats(stats, semantic_ids, quota_summary=quota.summary())
    report["requested_stt_rate"] = config.stt_noise_rate
    report["achieved_stt_rate"] = round(stats["stt_corrupted"] / max(stats["accepted"], 1), 4)
    report["requested"] = config.count
    report["attempts"] = attempts
    report["config"] = config.to_dict()
    report["grounding"] = {
        "requested_rate": config.grounding_rate,
        "rows": sum(grounding_rows.values()),
        "achieved_rate": round(sum(grounding_rows.values()) / max(len(accepted), 1), 4),
        "by_family": dict(sorted(grounding_rows.items())),
    }
    if config.real_home_path:
        report["real_home"] = {
            "path": str(config.real_home_path),
            "requested_rate": config.real_home_rate,
            "synthetic_only": config.synthetic_only,
            # One row is one row: a multi-target row names several entities but is
            # still a single real-home example.
            "rows": real_home_rows,
            "achieved_rate": round(real_home_rows / max(len(accepted), 1), 4),
            "entity_cap": real_home_entity_cap,
            "entities_used": len(real_home_targets),
            "target_counts": dict(sorted(real_home_targets.items())),
            "most_common": real_home_targets.most_common(5),
        }
    report["registry"] = registry_summary()
    from generators.audit import audit_rows

    report["quality_audit"] = audit_rows(
        accepted,
        expected_count=config.count,
        required_operations={
            key for key, target in quota.targets["positive"].items() if target > 0
        },
        min_positive_per_operation=config.min_positive_per_operation,
        min_positive_per_tool=config.min_positive_per_tool,
        max_absence_rate=config.max_absence_rate,
    )
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
