"""Entity-grounding regressions: the request is fixed, the home decides the answer.

The anchor case is the one that failed in production -- Living Room,
``media_player.living_room_tv`` named "TV", "turn on the living room TV" ->
``HassTurnOn(name="TV", domain=["media_player"])``. Around it sit held-out
variations that change the name, the entity id, the area, the aliases, the
distractors and the device's supported actions, so passing the anchor by
memorising "living room media player -> TV" fails the rest.

Every expected call here is derived by ``generators.gold`` from the supplied
graph and the pinned v2 contract, not written by hand. Rows render through
``generators.labels.render_example``, so the prompt, context and tool schema are
the production ones. All prompts join ``excluded_train_prompts``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from generators import grounding

EVAL_SEED = 20260911


def eval_variants() -> list[dict[str, Any]]:
    """Held out of training: names, ids and areas differ from grounding.training_variants."""
    variants: list[dict[str, Any]] = []
    # The exact production regression.
    variants += grounding.media_presence_family(
        prefix="grounding_livingroom_tv",
        target_name="TV",
        renamed="Cinema",
        area="Living Room",
        elsewhere="Bedroom",
        entity_id="media_player.living_room_tv",
    )
    # Same four shapes, different name, id, area and aliases.
    variants += grounding.media_presence_family(
        prefix="grounding_office_display",
        target_name="Office Display",
        renamed="Big Screen",
        area="Office",
        elsewhere="Basement",
        entity_id="media_player.office_display_2",
        aliases=["Office Display", "the monitor"],
    )
    variants += grounding.ambiguity_family(
        prefix="grounding_kitchen_light", area="Kitchen", names=("Sink Light", "Island Pendants")
    )
    variants += grounding.domain_family(
        prefix="grounding_bathroom_fan",
        area="Bathroom",
        fan_name="Bathroom Extractor",
        light_name="Mirror Light",
    )
    variants += grounding.supported_action_family(
        prefix="grounding_porch_speaker", area="Porch", name="Porch Speaker"
    )
    variants += grounding.alias_family(
        prefix="grounding_den_lamp", area="Den", name="Corner Lamp", alias="mood light"
    )
    return variants


def grounding_specs() -> list[dict[str, Any]]:
    return [
        grounding.build_spec(variant, seed=EVAL_SEED, index=index)
        for index, variant in enumerate(eval_variants())
    ]


def build_grounding_examples() -> list[dict[str, Any]]:
    rows = []
    for index, variant in enumerate(eval_variants()):
        row = grounding.build_row(variant, seed=EVAL_SEED, index=index)
        row["metadata"]["quality_eval"] = True
        row["metadata"]["grounding_eval"] = True
        rows.append(row)
    return rows


def grounding_user_prompts() -> list[str]:
    return [
        next(message["content"] for message in row["messages"] if message["role"] == "user")
        for row in build_grounding_examples()
    ]


def expected_behavior(spec: dict[str, Any]) -> dict[str, Any]:
    """Compact assertion target: the tool, its arguments, and the outcome."""
    expected = spec["expected"]
    return {
        "family": spec["grounding_family"],
        "prompt": spec["utterance"],
        "kind": expected["kind"],
        "response": expected.get("response"),
        "calls": [
            {"name": call["name"], "arguments": call["arguments"]}
            for call in expected.get("calls") or []
        ],
    }


if __name__ == "__main__":
    import json

    for spec in grounding_specs():
        print(json.dumps(expected_behavior(spec), ensure_ascii=False))
