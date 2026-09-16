#!/usr/bin/env python3
"""The LLM half of the v1/v2 pipeline: verbalise, judge, curate.

Specs arrive already labelled. A generator model only supplies wording, a
second independent model scores it, and curation picks the survivors. None of
these stages may change a label — the guards in ``validate_utterance`` and
``_audit_selected`` are what make that a property rather than a hope.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rendering import (  # noqa: E402
    _BANNED_UTTERANCE,
    _CLEAN_DIRECT_START,
    _FRAMING_PREFIXES,
    _FRAMING_SUFFIXES,
    _UNNATURAL_FRAMING,
    _expand_template,
    _protected_slots,
    request_seed,
    template_seed,
)
from adapters.schema import (  # noqa: E402
    tool_schema_map,
    v2_openai_tools,
    validate_tool_arguments,
)
from v2_scenarios import CATEGORY_WEIGHTS, validate_spec  # noqa: E402

DEFAULT_TRAIN_COUNT = 10_000

def _framed_utterance(spec: dict[str, Any], framing: Any) -> str | None:
    if (
        not isinstance(framing, list)
        or len(framing) < 2
        or not all(isinstance(part, str) for part in framing)
    ):
        return None
    prefix, suffix = framing[0].strip(), framing[-1].strip()
    framing_text = f"{prefix} {suffix}".strip()
    if len(framing_text.split()) > 12 or re.search(r"\d", framing_text):
        return None
    if re.search(r"\b(?:not|don['’]?t|dont|instead|except|without|leave|ignore|cancel|stop)\b", framing_text, re.I):
        return None
    allowed_prefix = {
        "hey", "hi", "okay", "ok", "uh", "um", "could", "can", "would", "you", "please",
        "just", "kindly", "maybe", "when", "get", "a", "chance", "before", "i", "forget",
        "id", "d", "like", "to", "go", "ahead", "also", "so", "well", "alright",
    }
    allowed_suffix = {"for", "me", "please", "thanks", "thank", "you", "right", "now", "if", "can"}

    def safe_words(text: str, allowed: set[str]) -> str:
        end = 0
        for match in re.finditer(r"[A-Za-z]+", text):
            if match.group().casefold() not in allowed:
                break
            end = match.end()
        return text[:end].strip()

    prefix = safe_words(prefix, allowed_prefix)
    suffix_words = safe_words(suffix, allowed_suffix)
    punctuation = suffix[-1] if suffix and suffix[-1] in ".?!" else ""
    suffix = suffix_words + punctuation
    seed = request_seed(spec)
    utterance = f"{prefix} {seed}".strip()
    if suffix:
        utterance += suffix if suffix[0] in ".,?!" else f" {suffix}"
    if _UNNATURAL_FRAMING.search(utterance):
        return None
    return utterance[:1].upper() + utterance[1:]


def _verbalizer_prompt(specs: list[dict[str, Any]]) -> str:
    payload = [
        {
            "candidate_id": spec["candidate_id"],
            "category": spec["category"],
            "subcategory": spec["subcategory"],
            "template_seed": template_seed(spec),
            "contrastive_group": spec["contrastive_group"],
        }
        for spec in specs
    ]
    return (
        "Generate natural conversational framing for each authoritative template_seed without rewriting or "
        "answering it. Each framing is [prefix,suffix]; examples are [\"Hey, could you please\",\"for me?\"], "
        "[\"Uh\",\"please.\"], or [\"\",\"\"]. Do not put actions, device names, negation, exclusions, or "
        "numbers in framing. Use sparse framing for clean_direct, varied speech for conversational, and mild "
        "disfluency for stt_corrupted. Return JSON only as {\"framings\":[[\"prefix\",\"suffix\"],...]}, in "
        "exact input order with exactly one pair per input.\nITEMS:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def framing_response_format(count: int) -> dict[str, Any]:
    """Constrain llama.cpp to one safe framing pair per authoritative row."""
    framing = {
        "type": "array",
        "minItems": 2,
        "maxItems": 2,
        "prefixItems": [
            {"type": "string", "enum": list(_FRAMING_PREFIXES)},
            {"type": "string", "enum": list(_FRAMING_SUFFIXES)},
        ],
    }
    schema = {
        "type": "object",
        "properties": {
            "framings": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": framing,
            }
        },
        "required": ["framings"],
        "additionalProperties": False,
    }
    return {"type": "json_schema", "json_schema": {"name": "framings", "strict": True, "schema": schema}}


def judge_response_format(count: int) -> dict[str, Any]:
    """Constrain llama.cpp to three 1-5 quality scores per row."""
    score = {"type": "string", "pattern": "^[1-5]{3}$"}
    schema = {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": score,
            }
        },
        "required": ["scores"],
        "additionalProperties": False,
    }
    return {"type": "json_schema", "json_schema": {"name": "scores", "strict": True, "schema": schema}}


def verbalize_batch(
    specs: list[dict[str, Any]],
    complete: Any,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Fill utterances without allowing the language model to alter labels."""
    response = complete(_verbalizer_prompt(specs))
    framings = response.get("framings") if isinstance(response, dict) else None
    if len(specs) == 1 and isinstance(framings, list) and framings:
        framings = framings[:1]
    if isinstance(framings, list) and len(framings) == len(specs):
        verbalized: list[dict[str, Any]] = []
        rejected: dict[str, str] = {}
        for original, framing in zip(specs, framings):
            utterance = _framed_utterance(original, framing)
            if not utterance:
                rejected[original["candidate_id"]] = "verbalizer_missing_item"
                continue
            spec = deepcopy(original)
            spec["utterance"] = utterance
            verbalized.append(spec)
        return verbalized, rejected
    utterances = response.get("utterances") if isinstance(response, dict) else None
    if isinstance(utterances, list) and len(utterances) == len(specs):
        verbalized: list[dict[str, Any]] = []
        rejected: dict[str, str] = {}
        for original, utterance in zip(specs, utterances):
            expanded = _expand_template(original, utterance) if isinstance(utterance, str) else None
            if not expanded:
                rejected[original["candidate_id"]] = "verbalizer_missing_item"
                continue
            spec = deepcopy(original)
            spec["utterance"] = expanded
            verbalized.append(spec)
        return verbalized, rejected
    items = response.get("items") if isinstance(response, dict) else None
    by_id: dict[str, str] = {}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            candidate_id = item.get("candidate_id")
            utterance = item.get("utterance")
            if isinstance(candidate_id, str) and isinstance(utterance, str) and utterance.strip():
                if candidate_id in by_id:
                    by_id.pop(candidate_id, None)
                    continue
                by_id[candidate_id] = utterance.strip()
    verbalized: list[dict[str, Any]] = []
    rejected: dict[str, str] = {}
    for original in specs:
        candidate_id = original["candidate_id"]
        if candidate_id not in by_id:
            rejected[candidate_id] = "verbalizer_missing_item"
            continue
        expanded = _expand_template(original, by_id[candidate_id])
        if not expanded:
            rejected[candidate_id] = "verbalizer_missing_item"
            continue
        spec = deepcopy(original)
        spec["utterance"] = expanded
        verbalized.append(spec)
    return verbalized, rejected


def verbalize_resilient(
    specs: list[dict[str, Any]],
    complete: Any,
    *,
    attempts: int = 3,
) -> list[dict[str, Any]]:
    """Retry missing rows, splitting malformed batches down to single requests."""
    completed: dict[str, dict[str, Any]] = {}
    remaining = list(specs)
    retry_count = max(attempts, 10) if len(specs) == 1 else attempts
    for attempt in range(retry_count):
        retrying_complete = lambda prompt, n=attempt: complete(
            prompt.replace("ITEMS:\n", f"RETRY_VARIANT:{n}\nITEMS:\n", 1)
        )
        try:
            verbalized, rejected = verbalize_batch(remaining, retrying_complete)
        except Exception:
            verbalized = []
            rejected = {spec["candidate_id"]: "verbalizer_completion_error" for spec in remaining}
        completed.update({spec["candidate_id"]: spec for spec in verbalized})
        remaining = [spec for spec in remaining if spec["candidate_id"] in rejected]
        if not remaining:
            break
    if remaining:
        if len(remaining) == 1:
            spec = deepcopy(remaining[0])
            spec["utterance"] = _framed_utterance(spec, ["", ""]) or request_seed(spec)
            completed[spec["candidate_id"]] = spec
        else:
            midpoint = len(remaining) // 2
            for spec in verbalize_resilient(remaining[:midpoint], complete, attempts=attempts):
                completed[spec["candidate_id"]] = spec
            for spec in verbalize_resilient(remaining[midpoint:], complete, attempts=attempts):
                completed[spec["candidate_id"]] = spec
    return [completed[spec["candidate_id"]] for spec in specs]


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold().replace("’", "'")))


def validate_utterance(spec: dict[str, Any]) -> str | None:
    """Apply deterministic label/context checks before the independent judge."""
    reason = validate_spec(spec)
    if reason:
        return reason
    utterance = spec.get("utterance")
    if not isinstance(utterance, str) or not (2 <= len(utterance.split()) <= 60):
        return "invalid_utterance_length"
    if _UNNATURAL_FRAMING.search(utterance):
        return "unnatural_framing"
    if _BANNED_UTTERANCE.search(utterance):
        return "banned_content"
    text = _normalized(utterance)
    expected = spec["expected"]
    if spec["category"] == "conversational" and _CLEAN_DIRECT_START.match(utterance.strip()):
        return "conversational_not_voice"
    if expected["kind"] == "no_action":
        hint_terms = set(_normalized(spec["request_hint"]).split()) - {"the", "to", "a", "an"}
        if not hint_terms.intersection(text.split()):
            return "no_action_intent_missing"
        if "thermostat" in text and spec["category"] == "unsupported_no_action":
            return "banned_thermostat"
        return None

    if spec["category"] == "ambiguity" and spec.get("request_hint"):
        hint_terms = set(_normalized(spec["request_hint"]).split()) - {"the", "to", "a", "an"}
        if not hint_terms.intersection(text.split()):
            return "missing_expected_target"
        if expected["kind"] in {"action", "no_action"}:
            return None

    for canonical in spec["target_names"]:
        spoken = spec["spoken_targets"].get(canonical, canonical)
        if _normalized(spoken) not in text:
            return "missing_expected_target"

    if expected["kind"] == "status":
        if not any(cue in text for cue in ("status", "is ", "are ", "what ", "check", "doing")):
            return "status_not_query"
        if any(cue in text for cue in ("turn on", "turn off", "switch on", "switch off", "set ", "open ", "close ")):
            return "status_not_query"
        if any(call["name"] != "GetLiveContext" for call in expected.get("calls", [])):
            return "status_wrong_tool"
        return None

    action_cues = {
        "HassTurnOn": ("turn on", "switch on", "activate", "start", "open", "lock"),
        "HassTurnOff": ("turn off", "switch off", "deactivate", "stop", "close", "shut", "unlock"),
        "HassLightSet": ("brightness", "percent", "color", "dim", "bright"),
        "HassFanSetSpeed": ("speed", "percent", "faster", "slower"),
    }
    for call in expected["calls"]:
        split_verb = {"HassTurnOn": "on", "HassTurnOff": "off"}.get(call["name"])
        separated = split_verb and re.search(r"\b(?:turn|switch)\b.+?\b" + split_verb + r"\b", text)
        color_request = (call["name"] == "HassLightSet" and call["arguments"].get("color")
                         and _normalized(str(call["arguments"]["color"])) in text
                         and re.search(r"\b(?:make|change|set)\b", text))
        percentage_request = (call["name"] in {"HassLightSet", "HassFanSetSpeed"}
                              and "%" in utterance and re.search(r"\b(?:set|change|dim|brighten)\b", text))
        if not separated and not color_request and not percentage_request and not any(cue in text for cue in action_cues.get(call["name"], ())):
            return "action_intent_missing"
    if spec["excluded_names"]:
        if any(_normalized(name) not in text for name in spec["excluded_names"]):
            return "missing_exclusion"
        if not any(cue in text for cue in ("leave", "except", "not", "dont", "alone", "exclude")):
            return "missing_exclusion"
    return None


def _judge_prompt(specs: list[dict[str, Any]]) -> str:
    payload = [
        {
            "candidate_id": spec["candidate_id"],
            "utterance": spec["utterance"],
            "authoritative_seed": request_seed(spec),
            "category": spec["category"],
            "subcategory": spec["subcategory"],
            "expected": spec["expected"],
            "excluded_names": spec["excluded_names"],
        }
        for spec in specs
    ]
    return (
        f"Independently judge all {len(specs)} utterance/authoritative-seed pairs. "
        "Score correctness, then clarity, then naturalness. Each digit is 1-5. "
        "correctness: 5 matches action vs status vs no-action, entities, exclusions, and exact_call_count; "
        "1 contradicts the label. "
        "clarity: 5 a listener knows exactly what to do; 1 garbled or ambiguous. "
        "naturalness: 5 a person would say this to a home voice assistant; 4 a normal terse command; "
        "3 robotic template-speak; 1-2 ungrammatical, including stacked framing like "
        "'Could you please what is the status...'. "
        "Do not default to 555. Use the full 1-5 range. Direct commands may still be 5/5/4. "
        f"Return JSON only with exactly {len(specs)} ordered score strings: "
        "{\"scores\":[\"543\"]}. No explanation.\nITEMS:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def judge_batch(
    specs: list[dict[str, Any]],
    complete: Any,
    *,
    generator_model: str,
    judge_model: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Use a distinct model to score label agreement and language quality."""
    if generator_model.strip().casefold() == judge_model.strip().casefold():
        raise ValueError("judge model must differ from generator model")
    eligible: list[dict[str, Any]] = []
    rejected: dict[str, str] = {}
    for spec in specs:
        reason = validate_utterance(spec)
        if reason:
            rejected[spec["candidate_id"]] = reason
        else:
            eligible.append(spec)
    if not eligible:
        return [], rejected
    response = complete(_judge_prompt(eligible))
    compact_scores = response.get("scores") if isinstance(response, dict) else None
    if len(eligible) == 1 and isinstance(compact_scores, dict):
        compact_scores = [compact_scores]
    if isinstance(compact_scores, list) and len(compact_scores) == len(eligible):
        difficulty = {
            "clean_direct": 2,
            "conversational": 3,
            "multi_action_exclusion": 5,
            "stt_corrupted": 5,
            "status": 4,
            "ambiguity": 5,
            "unsupported_no_action": 5,
            "entity_identity": 4,
        }
        accepted: list[dict[str, Any]] = []
        for spec, score in zip(eligible, compact_scores):
            if isinstance(score, str) and re.fullmatch(r"[1-5]{3}", score):
                correctness, clarity, naturalness = map(int, score)
            elif isinstance(score, dict) and all(
                isinstance(score.get(key), int) and 1 <= score[key] <= 5
                for key in ("correctness", "clarity", "naturalness")
            ):
                correctness, clarity, naturalness = (
                    score["correctness"], score["clarity"], score["naturalness"]
                )
            elif isinstance(score, list) and len(score) == 3 and all(
                isinstance(value, int) and 1 <= value <= 5 for value in score
            ):
                correctness, clarity, naturalness = score
            else:
                rejected[spec["candidate_id"]] = "judge_invalid_item"
                continue
            if min(correctness, clarity, naturalness) < 4:
                rejected[spec["candidate_id"]] = "judge_below_threshold"
                continue
            judged = deepcopy(spec)
            judged["quality"] = {
                "correctness": correctness,
                "clarity": clarity,
                "naturalness": naturalness,
                "difficulty": difficulty[spec["category"]],
                "semantic_key": request_seed(spec),
            }
            accepted.append(judged)
        return accepted, rejected
    items = response.get("items") if isinstance(response, dict) else None
    by_id = {
        item.get("candidate_id"): item
        for item in items or []
        if isinstance(item, dict) and isinstance(item.get("candidate_id"), str)
    }
    accepted: list[dict[str, Any]] = []
    for spec in eligible:
        candidate_id = spec["candidate_id"]
        item = by_id.get(candidate_id)
        if item is None:
            rejected[candidate_id] = "judge_missing_item"
            continue
        scores = {key: item.get(key) for key in ("correctness", "clarity", "naturalness", "difficulty")}
        semantic_key = item.get("semantic_key")
        if (
            not all(isinstance(value, int) and 1 <= value <= 5 for value in scores.values())
            or not isinstance(semantic_key, str)
            or not semantic_key.strip()
            or not isinstance(item.get("accept"), bool)
        ):
            rejected[candidate_id] = "judge_invalid_item"
            continue
        if not item["accept"] or min(scores["correctness"], scores["clarity"], scores["naturalness"]) < 4:
            rejected[candidate_id] = "judge_below_threshold"
            continue
        judged = deepcopy(spec)
        judged["quality"] = {**scores, "semantic_key": semantic_key.strip()}
        accepted.append(judged)
    return accepted, rejected


def judge_resilient(
    specs: list[dict[str, Any]],
    complete: Any,
    *,
    generator_model: str,
    judge_model: str,
    attempts: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Retry malformed judge output and split batches without retrying real quality failures."""
    accepted: dict[str, dict[str, Any]] = {}
    final_rejected: dict[str, str] = {}
    remaining = list(specs)
    retry_count = max(attempts, 3) if len(specs) == 1 else attempts
    for _attempt in range(retry_count):
        try:
            passed, rejected = judge_batch(
                remaining,
                complete,
                generator_model=generator_model,
                judge_model=judge_model,
            )
        except Exception:
            passed = []
            rejected = {spec["candidate_id"]: "judge_invalid_item" for spec in remaining}
        accepted.update({spec["candidate_id"]: spec for spec in passed})
        retry_ids = {
            candidate_id
            for candidate_id, reason in rejected.items()
            if reason in {"judge_missing_item", "judge_invalid_item"}
        }
        final_rejected.update(
            (candidate_id, reason)
            for candidate_id, reason in rejected.items()
            if candidate_id not in retry_ids
        )
        remaining = [spec for spec in remaining if spec["candidate_id"] in retry_ids]
        if not remaining:
            break
    if remaining and len(remaining) > 1:
        midpoint = len(remaining) // 2
        for half in (remaining[:midpoint], remaining[midpoint:]):
            passed, rejected = judge_resilient(
                half,
                complete,
                generator_model=generator_model,
                judge_model=judge_model,
                attempts=attempts,
            )
            accepted.update({spec["candidate_id"]: spec for spec in passed})
            final_rejected.update(rejected)
        remaining = []
    for spec in remaining:
        final_rejected[spec["candidate_id"]] = "judge_invalid_item"
    return [accepted[spec["candidate_id"]] for spec in specs if spec["candidate_id"] in accepted], final_rejected


def _behavior_key(spec: dict[str, Any]) -> str:
    behavior = {
        "expected": spec["expected"],
        "excluded": spec["excluded_names"],
        "home_id": spec["home"]["home_id"],
    }
    return json.dumps(behavior, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _ranked(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    semantic_counts = Counter(_normalized(spec["quality"]["semantic_key"]) for spec in specs)
    category_counts = Counter(spec["category"] for spec in specs)
    ranked: list[dict[str, Any]] = []
    for original in specs:
        spec = deepcopy(original)
        quality = spec["quality"]
        semantic = _normalized(quality["semantic_key"])
        uniqueness = 5 / semantic_counts[semantic]
        coverage = 5 * len(specs) / (len(CATEGORY_WEIGHTS) * category_counts[spec["category"]])
        quality["rank_score"] = round(
            quality["correctness"] * 8
            + quality["clarity"] * 5
            + quality["naturalness"] * 4
            + quality["difficulty"] * 3
            + min(uniqueness, 5) * 2
            + min(coverage, 5),
            4,
        )
        ranked.append(spec)
    return sorted(ranked, key=lambda spec: (-spec["quality"]["rank_score"], spec["candidate_id"]))


def curate(
    specs: list[dict[str, Any]],
    *,
    min_count: int = DEFAULT_TRAIN_COUNT,
    max_count: int = DEFAULT_TRAIN_COUNT,
    excluded_utterances: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Hard-filter duplicates, then retain the highest-quality balanced rows."""
    if min_count <= 0 or max_count < min_count:
        raise ValueError("invalid curation bounds")
    drops: Counter[str] = Counter()
    exact_seen: set[tuple[str, str]] = set()
    semantic_seen: set[tuple[str, str]] = set()
    excluded = {_normalized(text) for text in excluded_utterances or set()}
    unique: list[dict[str, Any]] = []
    for spec in _ranked(specs):
        if _normalized(spec["utterance"]) in excluded:
            drops["heldout_overlap"] += 1
            continue
        behavior = _behavior_key(spec)
        exact = (_normalized(spec["utterance"]), behavior)
        if exact in exact_seen:
            drops["exact_duplicate"] += 1
            continue
        exact_seen.add(exact)
        semantic = (_normalized(spec["quality"]["semantic_key"]), behavior)
        if semantic in semantic_seen:
            drops["semantic_duplicate"] += 1
            continue
        semantic_seen.add(semantic)
        unique.append(spec)
    if len(unique) < min_count:
        raise ValueError(
            f"quality floor left {len(unique)} rows, below required minimum {min_count}; thresholds unchanged"
        )
    if min_count >= 100:
        available = Counter(spec["category"] for spec in unique)
        missing = {
            category: math.ceil(min_count * weight / 200)
            for category, weight in CATEGORY_WEIGHTS.items()
            if available[category] < math.ceil(min_count * weight / 200)
        }
        if missing:
            raise ValueError(f"quality coverage floor not met: {missing}; thresholds unchanged")
    if len(unique) <= max_count:
        return unique, dict(drops)

    quotas = {category: max_count * weight // 100 for category, weight in CATEGORY_WEIGHTS.items()}
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for category, quota in quotas.items():
        rows = [spec for spec in unique if spec["category"] == category][:quota]
        selected.extend(rows)
        selected_ids.update(spec["candidate_id"] for spec in rows)
    direct_cap = max_count * CATEGORY_WEIGHTS["clean_direct"] // 100
    for spec in unique:
        if len(selected) >= max_count:
            break
        if spec["candidate_id"] in selected_ids:
            continue
        if spec["category"] == "clean_direct" and sum(
            row["category"] == "clean_direct" for row in selected
        ) >= direct_cap:
            continue
        selected.append(spec)
        selected_ids.add(spec["candidate_id"])
    if len(selected) < min_count:
        raise ValueError(
            f"coverage constraints left {len(selected)} rows, below required minimum {min_count}; thresholds unchanged"
        )
    drops["ranked_below_cut"] += len(unique) - len(selected)
    return sorted(
        selected, key=lambda spec: (-spec["quality"]["rank_score"], spec["candidate_id"])
    ), dict(drops)


def _json_content(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.I)
    start = stripped.find("{")
    if start < 0:
        raise ValueError("model response did not contain a JSON object")
    parsed, _end = json.JSONDecoder().raw_decode(stripped[start:])
    if not isinstance(parsed, dict):
        raise ValueError("model response JSON must be an object")
    return parsed


def openai_complete(
    prompt: str,
    *,
    base_url: str,
    model: str,
    api_key: str = "",
    temperature: float = 0.2,
    max_tokens: int = 8192,
    timeout: float = 600,
    post: Any = None,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call an OpenAI-compatible chat endpoint and parse its JSON response."""
    if post is None:
        import requests

        post = requests.post
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": response_format or {"type": "json_object"},
    }
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = post(
                f"{base_url.rstrip('/')}/chat/completions",
                json=body,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("model response content must be text")
            return _json_content(content)
        except Exception as error:  # endpoint and model formatting failures share retry policy
            last_error = error
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(f"OpenAI-compatible completion failed after 3 attempts: {last_error}")


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            candidate_id = row.get("candidate_id")
            if not isinstance(candidate_id, str):
                raise ValueError(f"{path}:{line_no} lacks candidate_id")
            rows[candidate_id] = row
    return rows


def load_user_utterances(path: Path) -> set[str]:
    """Read user prompts from canonical SaySo JSONL for leakage exclusion."""
    utterances: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            for message in json.loads(line).get("messages", []):
                if message.get("role") != "user":
                    continue
                content = message.get("content")
                if isinstance(content, str):
                    utterances.add(content)
                elif isinstance(content, list):
                    text = " ".join(
                        part["text"] for part in content
                        if isinstance(part, dict) and isinstance(part.get("text"), str)
                    )
                    if text:
                        utterances.add(text)
    return utterances


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["category"] for row in rows).items()))


def _audit_selected(rows: list[dict[str, Any]]) -> dict[str, int]:
    schemas = tool_schema_map(v2_openai_tools())
    calls = [call for row in rows for call in row["expected"].get("calls", [])]
    exact = [(_normalized(row["utterance"]), _behavior_key(row)) for row in rows]
    semantic = [(_normalized(row["quality"]["semantic_key"]), _behavior_key(row)) for row in rows]
    return {
        "deterministically_valid_rows": sum(validate_utterance(row) is None for row in rows),
        "schema_valid_tool_calls": sum(
            validate_tool_arguments(call["name"], call["arguments"], schemas) is None for call in calls
        ),
        "tool_calls": len(calls),
        "no_action_rows": sum(not row["expected"].get("calls") for row in rows),
        "status_rows": sum(row["expected"]["kind"] == "status" for row in rows),
        "multi_call_rows": sum(len(row["expected"].get("calls", [])) > 1 for row in rows),
        "exclusion_rows": sum(bool(row["excluded_names"]) for row in rows),
        "stt_resolution_rows": sum(row["category"] == "stt_corrupted" for row in rows),
        "apostrophe_argument_rows": sum(
            any("'" in call["arguments"].get("name", "") for call in row["expected"].get("calls", []))
            for row in rows
        ),
        "contrastive_rows": sum(bool(row["contrastive_group"]) for row in rows),
        "contrastive_groups": len({row["contrastive_group"] for row in rows if row["contrastive_group"]}),
        "excluded_entity_call_violations": sum(
            call["arguments"].get("name") in set(row["excluded_names"])
            for row in rows
            for call in row["expected"].get("calls", [])
        ),
        "exact_duplicate_keys": len(exact) - len(set(exact)),
        "semantic_duplicate_keys": len(semantic) - len(set(semantic)),
    }


def _run_batches(rows: list[dict[str, Any]], batch_size: int, workers: int, process: Any):
    batches = [rows[offset : offset + batch_size] for offset in range(0, len(rows), batch_size)]
    if workers == 1:
        for batch in batches:
            yield process(batch)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(process, batches)
