from __future__ import annotations

from typing import Any

from evals.runner import production_catalog


def entity(domain: str, name: str, **extra: Any) -> dict[str, Any]:
    data = {"domain": domain, "name": name}
    data.update(extra)
    return data


def names(home: dict[str, Any]) -> set[str]:
    return {tool["function"]["name"] for tool in production_catalog(home)}


ALWAYS_OFFERED = {
    "llm__GetDateTime",
    "homeassistant__GetLiveContext",
    "intent__HassTurnOn",
    "intent__HassTurnOff",
    "intent__HassCancelAllTimers",
    "assist_satellite__HassBroadcast",
}


def test_always_offered_tools_present() -> None:
    assert names({"entities": []}) >= ALWAYS_OFFERED


def test_search_and_play_never_use_old_combined_tool() -> None:
    home = {
        "entities": [
            entity(
                "media_player", "Living TV", capabilities=["search_media", "play_media"]
            )
        ]
    }
    offered = names(home)
    assert not any("SearchAndPlay" in name for name in offered)


def test_search_and_play_require_both_capabilities() -> None:
    both = entity(
        "media_player", "Living TV", capabilities=["search_media", "play_media"]
    )
    search_only = entity("media_player", "Living TV", capabilities=["search_media"])
    play_only = entity("media_player", "Living TV", capabilities=["play_media"])

    assert {
        "media_player__search_media",
        "media_player__play_media",
    } <= names({"entities": [both]})

    for media in (search_only, play_only):
        offered = names({"entities": [media]})
        assert "media_player__search_media" not in offered
        assert "media_player__play_media" not in offered

    assert "media_player__search_media" not in names({"entities": []})


def test_todo_tools_require_todo_entity() -> None:
    todo_names = {
        "todo__get_items",
        "todo__HassListAddItem",
        "todo__HassListCompleteItem",
        "todo__HassListRemoveItem",
    }
    assert not (todo_names & names({"entities": [entity("light", "Lamp")]}))

    todo = entity("todo", "Grocery List", aliases=["shopping"])
    home = {"entities": [entity("light", "Lamp"), todo]}
    offered = names(home)
    assert todo_names <= offered

    get_items = next(
        tool
        for tool in production_catalog(home)
        if tool["function"]["name"] == "todo__get_items"
    )
    enum = get_items["function"]["parameters"]["properties"]["todo_list"]["enum"]
    assert set(enum) >= {"Grocery List", "shopping"}


def test_cover_home_gets_position_tools_but_light_home_does_not() -> None:
    position_names = {"intent__HassSetPosition", "intent__HassStopMoving"}

    cover = {"entities": [entity("cover", "Living Blinds")]}
    assert position_names <= names(cover)

    valve = {"entities": [entity("valve", "Irrigation Valve")]}
    assert position_names <= names(valve)

    light_only = {"entities": [entity("light", "Kitchen Light")]}
    assert not (position_names & names(light_only))


def test_script_tools_use_object_id_name() -> None:
    home = {
        "entities": [
            entity("script", "Movie Night", entity_id="script.movie_night"),
            entity("script", "Night Mode", entity_id="script.night_mode"),
        ]
    }
    assert {"script__movie_night", "script__night_mode"} <= names(home)


def test_removed_tools_still_work() -> None:
    home = {"entities": [entity("light", "Lamp")]}
    catalog = production_catalog(home, removed_tools=["intent__HassTurnOn"])
    assert "intent__HassTurnOn" not in {tool["function"]["name"] for tool in catalog}
