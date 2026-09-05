#!/usr/bin/env python3
"""Generate a held-out synthetic test set with a fixed behavior mix."""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_synthetic_dataset import build_specs, expand_utterance, render_example  # noqa: E402

DEFAULT_COUNT = 2_500
MIX = {
    "normal_household": 50,
    "paraphrase_conversational": 20,
    "multi_action_exclusion": 10,
    "ambiguity_context": 10,
    "stt_corrupted": 5,
    "refusal_unsupported_clarification": 5,
}

_CONVERSATIONAL = (
    "Hey, could you {command}?",
    "When you get a chance, {command}.",
    "I'd like you to {command}, please.",
    "Can you {command} for me?",
    "Please {command}.",
    "I need you to {command}.",
    "Okay, {command}.",
    "Before I forget, {command}.",
    "Could you go ahead and {command}?",
    "Would you please {command}?",
)

_AMBIGUOUS = (
    ("turn on the light", "Which light would you like me to turn on?"),
    ("turn off the fan", "Which fan should I turn off?"),
    ("open the blinds", "Which blinds should I open?"),
    ("lock the door", "Which door should I lock?"),
    ("close the garage door", "Which garage door should I close?"),
)

_UNSUPPORTED = (
    ("set the thermostat to {value} degrees", "I can't control thermostats with the available tools."),
    ("start the robot vacuum in {room}", "I can't control robot vacuums with the available tools."),
    ("play music in {room}", "I can't control media players with the available tools."),
    ("add milk to my {room} list", "I can't update lists with the available tools."),
    ("start a {value} minute timer", "I can't start timers with the available tools."),
)

_ROOMS = (
    "kitchen",
    "bedroom",
    "living room",
    "garage",
    "office",
    "hallway",
    "basement",
    "patio",
    "laundry room",
    "guest room",
)

_STT_REPLACEMENTS = (
    (r"\blights\b", "likes"),
    (r"\blight\b", "lite"),
    (r"\bblinds\b", "blends"),
    (r"\bgarage\b", "garaj"),
    (r"\block\b", "lok"),
    (r"\bfan\b", "van"),
    (r"\bturn\b", "tern"),
    (r"\boff\b", "of"),
    (r"\bon\b", "own"),
)


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return ""


def _first_user_text(example: dict) -> str:
    return next(
        _content_text(message.get("content"))
        for message in example["messages"]
        if message.get("role") == "user"
    )


def _set_first_user_text(example: dict, text: str) -> None:
    message = next(message for message in example["messages"] if message.get("role") == "user")
    message["content"] = text


def _replace_dialogue(example: dict, user: str, assistant: str) -> None:
    system = next(message for message in example["messages"] if message.get("role") == "system")
    example["messages"] = [
        system,
        {"role": "user", "content": user, "train_on_turn": False},
        {"role": "assistant", "content": assistant, "train_on_turn": True},
    ]


def _mark(example: dict, category: str, subcategory: str, seed: int) -> dict:
    example["metadata"] = {
        **example.get("metadata", {}),
        "evaluation_category": category,
        "evaluation_subcategory": subcategory,
        "evaluation_seed": seed,
        "held_out": True,
    }
    return example


def _plain_command(text: str) -> str:
    command = text.strip().rstrip(".?!")
    command = re.sub(r"^(please|can you|could you|would you)\s+", "", command, flags=re.I)
    if not command:
        return command
    return command[:1].lower() + command[1:]


def _conversational(text: str, index: int) -> str:
    return _CONVERSATIONAL[index % len(_CONVERSATIONAL)].format(command=_plain_command(text))


def _stt_corrupt(text: str, index: int) -> str:
    corrupted = text.lower().strip().rstrip(".?!")
    replacements = _STT_REPLACEMENTS[index % len(_STT_REPLACEMENTS) :] + _STT_REPLACEMENTS[: index % len(_STT_REPLACEMENTS)]
    for pattern, replacement in replacements:
        changed = re.sub(pattern, replacement, corrupted, count=1)
        if changed != corrupted:
            corrupted = changed
            break
    if index % 3 == 0:
        corrupted = "uh " + corrupted
    elif index % 3 == 1:
        corrupted = re.sub(r"\bthe\s+", "", corrupted, count=1)
    else:
        corrupted = corrupted.replace(",", "")
    return corrupted


def _tool_names(example: dict) -> set[str]:
    names: set[str] = set()
    for message in example["messages"]:
        for call in message.get("tool_calls") or []:
            arguments = call.get("function", {}).get("arguments")
            if isinstance(arguments, str):
                value = json.loads(arguments).get("name")
                if isinstance(value, str):
                    names.add(value)
    return names


def _render_spec(spec: dict[str, Any], utterance: str | None = None) -> dict[str, Any]:
    rendered = copy.deepcopy(spec)
    rendered["utterance"] = utterance or expand_utterance(rendered)
    return render_example(rendered)


def _spec_pool_size(count: int) -> int:
    return ((max(count * 10, 2_500) + 99) // 100) * 100


def _take(
    pool: list[dict],
    count: int,
    *,
    category: str,
    subcategory: str,
    seed: int,
    rng: random.Random,
    used_prompts: set[str],
    transform: Callable[[dict, int], None] | None = None,
) -> list[dict]:
    candidates = list(pool)
    rng.shuffle(candidates)
    selected: list[dict] = []
    for candidate in candidates:
        example = copy.deepcopy(candidate)
        if transform:
            transform(example, len(selected))
        prompt = _first_user_text(example).casefold().strip()
        if not prompt or prompt in used_prompts:
            continue
        used_prompts.add(prompt)
        selected.append(_mark(example, category, subcategory, seed))
        if len(selected) == count:
            return selected
    raise ValueError(f"not enough unique examples for {category}/{subcategory}: {len(selected)}/{count}")


def _manual_no_tool(
    prototype: dict,
    pairs: list[tuple[str, str]],
    count: int,
    *,
    category: str,
    subcategory: str,
    seed: int,
    used_prompts: set[str],
) -> list[dict]:
    selected: list[dict] = []
    for user, assistant in pairs:
        prompt = user.casefold().strip()
        if prompt in used_prompts:
            continue
        example = copy.deepcopy(prototype)
        _replace_dialogue(example, user, assistant)
        used_prompts.add(prompt)
        selected.append(_mark(example, category, subcategory, seed))
        if len(selected) == count:
            return selected
    raise ValueError(f"not enough unique examples for {category}/{subcategory}")


def _ambiguity_pairs(prefix: str = "") -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    openings = ("", "please ", "could you ", "hey, ", "when you can, ", "would you ")
    endings = ("", " right now", " for me", " before I leave", " in here", " please")
    for opening in openings:
        for command, answer in _AMBIGUOUS:
            for ending in endings:
                pairs.append((f"{prefix}{opening}{command}{ending}".strip(), answer))
    return pairs


def _unsupported_pairs() -> list[tuple[str, str]]:
    return [
        (command.format(room=room, value=10 + index), answer)
        for index, room in enumerate(_ROOMS)
        for command, answer in _UNSUPPORTED
    ]


def build_balanced_test_set(
    *,
    count: int = DEFAULT_COUNT,
    seed: int = 1042,
    excluded_prompts: set[str] | None = None,
) -> list[dict]:
    """Build a deterministic held-out test set with the requested percentages."""
    if count <= 0 or count % 20:
        raise ValueError("count must be a positive multiple of 20")
    counts = {name: count * percent // 100 for name, percent in MIX.items()}
    rng = random.Random(seed)
    specs = build_specs(_spec_pool_size(count), seed=seed)
    by_category: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        by_category.setdefault(spec["category"], []).append(spec)

    singles = [
        _render_spec(spec)
        for spec in by_category.get("clean_direct", []) + by_category.get("entity_identity", [])
    ]
    multi_examples = [_render_spec(spec) for spec in by_category.get("multi_action_exclusion", [])]
    status_examples = [_render_spec(spec) for spec in by_category.get("status", [])]
    refusal_specs = [
        spec
        for spec in by_category.get("unsupported_no_action", [])
        if spec["expected"].get("response") == "refuse"
    ]
    refusal_examples = [_render_spec(spec) for spec in refusal_specs]
    prototype = _render_spec(
        next(spec for spec in by_category.get("ambiguity", []) if spec["expected"].get("response") == "clarify")
    )

    used = {prompt.casefold().strip() for prompt in excluded_prompts or set()}
    result: list[dict] = []

    result += _take(
        singles,
        counts["normal_household"],
        category="normal_household",
        subcategory="direct",
        seed=seed,
        rng=rng,
        used_prompts=used,
    )
    result += _take(
        singles,
        counts["paraphrase_conversational"],
        category="paraphrase_conversational",
        subcategory="conversational",
        seed=seed,
        rng=rng,
        used_prompts=used,
        transform=lambda example, index: _set_first_user_text(
            example, _conversational(_first_user_text(example), index)
        ),
    )

    multi_count = counts["multi_action_exclusion"] // 2
    result += _take(
        multi_examples,
        multi_count,
        category="multi_action_exclusion",
        subcategory="multi_action",
        seed=seed,
        rng=rng,
        used_prompts=used,
    )
    device_names = sorted({name for example in singles for name in _tool_names(example)})

    def add_exclusion(example: dict, index: int) -> None:
        called = _tool_names(example)
        excluded = next(name for name in device_names[index:] + device_names[:index] if name not in called)
        text = _first_user_text(example).strip().rstrip(".?!")
        _set_first_user_text(example, f"{text}, but leave {excluded} alone.")

    result += _take(
        multi_examples,
        counts["multi_action_exclusion"] - multi_count,
        category="multi_action_exclusion",
        subcategory="exclusion",
        seed=seed,
        rng=rng,
        used_prompts=used,
        transform=add_exclusion,
    )

    ambiguity_count = counts["ambiguity_context"] // 2
    result += _manual_no_tool(
        prototype,
        _ambiguity_pairs(),
        ambiguity_count,
        category="ambiguity_context",
        subcategory="ambiguity",
        seed=seed,
        used_prompts=used,
    )
    result += _take(
        status_examples,
        counts["ambiguity_context"] - ambiguity_count,
        category="ambiguity_context",
        subcategory="context_resolution",
        seed=seed,
        rng=rng,
        used_prompts=used,
        transform=lambda example, index: _set_first_user_text(
            example, _conversational(_first_user_text(example), index + 3)
        ),
    )

    result += _take(
        singles,
        counts["stt_corrupted"],
        category="stt_corrupted",
        subcategory="phonetic_or_dropped_word",
        seed=seed,
        rng=rng,
        used_prompts=used,
        transform=lambda example, index: _set_first_user_text(
            example, _stt_corrupt(_first_user_text(example), index)
        ),
    )

    edge_count = counts["refusal_unsupported_clarification"]
    refusal_count = edge_count * 2 // 5
    unsupported_count = edge_count * 2 // 5
    result += _take(
        refusal_examples,
        refusal_count,
        category="refusal_unsupported_clarification",
        subcategory="refusal",
        seed=seed,
        rng=rng,
        used_prompts=used,
        transform=lambda example, index: _set_first_user_text(
            example, _conversational(_first_user_text(example), index + 5)
        ),
    )
    result += _manual_no_tool(
        prototype,
        _unsupported_pairs(),
        unsupported_count,
        category="refusal_unsupported_clarification",
        subcategory="unsupported",
        seed=seed,
        used_prompts=used,
    )
    result += _manual_no_tool(
        prototype,
        _ambiguity_pairs("just to be clear, "),
        edge_count - refusal_count - unsupported_count,
        category="refusal_unsupported_clarification",
        subcategory="clarification",
        seed=seed,
        used_prompts=used,
    )

    rng.shuffle(result)
    return result


def load_prompts(paths: list[Path]) -> set[str]:
    prompts: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                prompts.add(_first_user_text(json.loads(line)))
    return prompts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=1042)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "datasets" / "sayso_test_balanced.jsonl",
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        nargs="*",
        default=[ROOT / "datasets" / "sayso_train.jsonl", ROOT / "datasets" / "sayso_val.jsonl"],
    )
    args = parser.parse_args()
    examples = build_balanced_test_set(
        count=args.count,
        seed=args.seed,
        excluded_prompts=load_prompts(args.exclude),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(example, ensure_ascii=False, separators=(",", ":")) for example in examples) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(examples)} examples to {args.out}")
    print(dict(sorted(Counter(example["metadata"]["evaluation_category"] for example in examples).items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
