"""LFM prompt builder tests."""

import json
from pathlib import Path

from sayso_server.conversation import LastIntent, LastTarget, SatelliteConversationState
from sayso_server.home_graph import Area, Entity, HomeGraphSnapshot, Scene, Script
from sayso_server.models import ENTITY_ID_PATTERN
from sayso_server.prompt import GENERATION_INSTRUCTION, PromptOrigin, build_lfm_prompt
from sayso_server.runtime import parse_lfm_prompt_payload

FIXTURES = Path(__file__).resolve().parents[3] / "evals" / "fixtures"


def _load_graph() -> HomeGraphSnapshot:
    data = json.loads((FIXTURES / "home_graph.json").read_text())
    return HomeGraphSnapshot.model_validate(data)


def test_prompt_includes_json_only_generation_instruction() -> None:
    graph = _load_graph()
    lamp = next(entity for entity in graph.entities if entity.name == "Floor Lamp")

    prompt = build_lfm_prompt(
        user_text="Turn off the lamp",
        origin=PromptOrigin(satellite_id="sat-1", area_name="Living Room"),
        conversation=SatelliteConversationState(),
        candidates=[lamp],
        areas=graph.areas,
    )

    assert prompt.startswith(GENERATION_INSTRUCTION)
    assert prompt[len(GENERATION_INSTRUCTION) :].startswith("\n")

    payload = json.loads(prompt[prompt.index("{") :])
    assert "generation_instruction" not in payload
    assert "ControlPlan JSON" in GENERATION_INSTRUCTION
    assert "No prose" in GENERATION_INSTRUCTION


def test_prompt_includes_origin_state_candidates_and_user_text_without_schema() -> None:
    graph = _load_graph()
    candidates = [graph.entities[1], graph.entities[0]]
    origin = PromptOrigin(
        satellite_id="satellite-kitchen",
        area_name="Living Room",
        area_aliases=["lounge"],
    )
    conversation = SatelliteConversationState(
        last_target=LastTarget(entity_ids=["light.floor_lamp"]),
        last_intent=LastIntent(intent="turn on the lamp", outcome="action"),
    )

    prompt = build_lfm_prompt(
        user_text="Turn off the lamp",
        origin=origin,
        conversation=conversation,
        candidates=candidates,
        areas=graph.areas,
    )

    assert "Turn off the lamp" in prompt
    assert "satellite-kitchen" in prompt
    assert "Living Room" in prompt
    assert "lounge" in prompt
    assert "Floor Lamp" in prompt
    assert "Living Room Ceiling" in prompt
    assert "turn on the lamp" in prompt

    payload = json.loads(prompt[prompt.index("{") :])
    assert "control_plan_schema" not in payload


def test_prompt_excludes_raw_entity_ids() -> None:
    graph = _load_graph()
    candidates = graph.entities[:2]

    prompt = build_lfm_prompt(
        user_text="Dim the ceiling lights",
        origin=PromptOrigin(satellite_id="sat-1", area_name="Living Room"),
        conversation=SatelliteConversationState(),
        candidates=candidates,
        areas=graph.areas,
    )

    for match in ENTITY_ID_PATTERN.finditer(prompt):
        raise AssertionError(f"prompt contains raw entity id: {match.group()}")


def test_prompt_excludes_unrelated_entities_from_full_graph() -> None:
    graph = _load_graph()
    lamp = next(entity for entity in graph.entities if entity.name == "Floor Lamp")
    ceiling = next(entity for entity in graph.entities if entity.name == "Living Room Ceiling")

    prompt = build_lfm_prompt(
        user_text="Turn off the lamp",
        origin=PromptOrigin(satellite_id="sat-1", area_name="Living Room"),
        conversation=SatelliteConversationState(),
        candidates=[lamp, ceiling],
        areas=graph.areas,
    )

    unrelated = [
        "Front Door",
        "Downstairs Thermostat",
        "Movie Time",
        "Good Night",
        "Primary Bedroom",
        "Kitchen",
    ]
    for label in unrelated:
        assert label not in prompt


def test_prompt_serializes_candidates_with_semantic_area_names_not_ids() -> None:
    graph = _load_graph()
    lamp = next(entity for entity in graph.entities if entity.name == "Floor Lamp")

    prompt = build_lfm_prompt(
        user_text="Turn off the lamp",
        origin=PromptOrigin(satellite_id="sat-1", area_name="Living Room"),
        conversation=SatelliteConversationState(),
        candidates=[lamp],
        areas=graph.areas,
    )

    assert "area_living_room" not in prompt
    assert "Living Room" in prompt
    assert "lamp" in prompt
    assert "reading lamp" in prompt


def test_prompt_resolves_last_target_to_semantic_names_from_candidates() -> None:
    graph = _load_graph()
    lamp = next(entity for entity in graph.entities if entity.name == "Floor Lamp")

    prompt = build_lfm_prompt(
        user_text="Turn it off",
        origin=PromptOrigin(satellite_id="sat-1", area_name="Living Room"),
        conversation=SatelliteConversationState(
            last_target=LastTarget(entity_ids=["light.floor_lamp"]),
        ),
        candidates=[lamp],
        areas=graph.areas,
    )

    assert "light.floor_lamp" not in prompt
    assert "Floor Lamp" in prompt


def test_prompt_supports_scene_and_script_candidates() -> None:
    graph = _load_graph()
    scene = graph.scenes[0]
    script = graph.scripts[0]
    living_room = next(area for area in graph.areas if area.name == "Living Room")
    bedroom = next(area for area in graph.areas if area.name == "Primary Bedroom")

    prompt = build_lfm_prompt(
        user_text="Run bedtime",
        origin=PromptOrigin(satellite_id="sat-1", area_name="Primary Bedroom"),
        conversation=SatelliteConversationState(),
        candidates=[scene, script],
        areas=[living_room, bedroom],
    )

    assert "scene.movie_time" not in prompt
    assert "script.good_night" not in prompt
    assert "Movie Time" in prompt
    assert "Good Night" in prompt
    assert "bedtime" in prompt


def test_prompt_uses_compact_json_without_timestamps_or_null_capability_fields() -> None:
    graph = _load_graph()
    ceiling = next(entity for entity in graph.entities if entity.name == "Living Room Ceiling")

    prompt = build_lfm_prompt(
        user_text="Turn off the ceiling lights",
        origin=PromptOrigin(satellite_id="sat-1", area_name="Living Room"),
        conversation=SatelliteConversationState(),
        candidates=[ceiling],
        areas=graph.areas,
    )

    json_body = prompt[prompt.index("{") :]
    assert "\n" not in json_body
    assert "last_changed" not in json_body
    assert "last_updated" not in json_body

    payload = parse_lfm_prompt_payload(prompt)
    candidate = payload["candidate_entities"][0]
    assert candidate["capabilities"] == [
        {"kind": "power"},
        {"kind": "brightness", "max_value": 100, "min_value": 1},
    ]
    assert candidate["state"] == {
        "attributes": {"brightness": 180, "color_mode": "brightness"},
        "value": "on",
    }
