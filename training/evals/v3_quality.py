"""Locked gold and shadow quality eval rows for the 40k v3 synthetic contract."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from adapters.schema import tool_schema_map, validate_tool_arguments, v2_openai_tools  # noqa: E402
from generators.labels import render_example  # noqa: E402
from evals.metrics import parse_tool_arguments  # noqa: E402
from generators.tools import script_tool_name  # noqa: E402
from generators.utterances import _phrase_for_call, expand_utterance, request_seed_from_spec  # noqa: E402
from evals.specs import (  # noqa: E402
    action as _action,
    assert_row_contract,
    entity as _entity,
    expected_tool_calls,  # noqa: F401
    fan_speed as _fan_speed,
    home as _home,
    light_set as _light_set,
    no_action as _no_action,
    normalized as _normalized,
    score_quality_gold,  # noqa: F401
    spec as _spec_base,
    status as _status,
    turn_off as _turn_off,
    turn_on as _turn_on,
)
from evals.recipe_lock import quality_eval_user_prompts as recipe_lock_prompts  # noqa: E402

_GOLD_AREAS = (
    "Great Room", "Family Room", "Home Theater", "Rec Room", "Study",
    "Dining Room", "Sunroom", "Mudroom", "Guest Suite", "Library",
)
_SHADOW_AREAS = (
    "Annex", "Atrium", "Conservatory", "Solarium", "Studio", "Terrace",
    "Balcony", "Cellar", "Attic", "Porch", "Veranda", "Courtyard",
)
SHADOW_MIN = 80
SHADOW_MAX = 120
DEFAULT_SHADOW_COUNT = 100


def _run_script(entity: dict[str, Any]) -> dict[str, Any]:
    """Scripts are their own zero-argument tool, not HassTurnOn."""
    return {"name": script_tool_name(entity), "arguments": {}}


def _spec(
    *,
    row_id: str,
    category: str,
    subcategory: str,
    utterance: str,
    home: dict[str, Any],
    expected: dict[str, Any],
    target_names: list[str] | None = None,
    request_hint: str = "",
) -> dict[str, Any]:
    """One locked v3 gold row."""
    return _spec_base(
        candidate_id=f"v3_quality_gold_{row_id}",
        category=category,
        subcategory=subcategory,
        utterance=utterance,
        home=home,
        expected=expected,
        target_names=target_names or None,
        request_hint=request_hint,
        dataset="v3_quality_gold",
    )


def _shadow_spec_shell(
    *,
    candidate_id: str,
    seed: int,
    category: str,
    subcategory: str,
    home: dict[str, Any],
    expected: dict[str, Any],
    target_names: list[str],
    request_hint: str = "",
    utterance: str | None = None,
) -> dict[str, Any]:
    """One shadow row: the same shape as gold, but not a promotion gate."""
    return _spec_base(
        candidate_id=candidate_id,
        seed=seed,
        category=category,
        subcategory=subcategory,
        utterance=utterance,
        home=home,
        expected=expected,
        target_names=target_names,
        request_hint=request_hint,
        quality_eval=False,
        dataset="v3_quality_shadow",
    )


def gold_specs() -> list[dict[str, Any]]:
    """Return authoritative locked gold rows for the 40k v3 quality gate."""
    great_room_thermostat = _entity(name="Great Room Thermostat", kind="climate", area="Great Room", state="heat")
    family_room_tv = _entity(name="Family Room TV", kind="media_player", area="Family Room", aliases=["tv"])
    home_theater_tv = _entity(name="Home Theater TV", kind="media_player", area="Home Theater", aliases=["tv"])
    rec_room_tv = _entity(name="Rec Room TV", kind="media_player", area="Rec Room")
    bedroom_tv = _entity(name="Guest Suite TV", kind="media_player", area="Guest Suite")
    upstairs_vacuum = _entity(name="Upstairs Robot Vacuum", kind="vacuum", area="Sunroom", state="docked")
    mudroom_vacuum = _entity(name="Mudroom Robot Vacuum", kind="vacuum", area="Mudroom", state="cleaning")
    movie_scene = _entity(name="Movie Night Scene", kind="scene", area="Home Theater", state="off")
    bedtime_scene = _entity(name="Bedtime Scene", kind="scene", area="Guest Suite", state="off")
    morning_script = _entity(name="Good Morning Script", kind="script", area="Kitchen", state="off")
    away_script = _entity(name="Leave Home Script", kind="script", area="Foyer", state="off")
    dining_light = _entity(name="Dining Room Pendant Light", kind="light", area="Dining Room", aliases=["light"])
    study_fan = _entity(name="Study Desk Fan", kind="fan", area="Study", aliases=["fan"])
    library_light_a = _entity(name="Library Reading Lamp", kind="light", area="Library", aliases=["light"])
    library_light_b = _entity(name="Library Desk Lamp", kind="light", area="Library")
    sunroom_light = _entity(name="Sunroom Accent Light", kind="light", area="Sunroom", aliases=["light"])

    rows: list[dict[str, Any]] = [
        _spec(
            row_id="climate_a",
            category="climate_setpoint",
            subcategory="named_device",
            utterance="Set Great Room Thermostat to 72 degrees",
            home=_home(great_room_thermostat, satellite_area="Great Room", home_id="v3_gold_climate_a"),
            expected=_action(
                {
                    "name": "HassClimateSetTemperature",
                    "arguments": {"name": "Great Room Thermostat", "temperature": 72},
                }
            ),
        ),
        _spec(
            row_id="climate_b",
            category="climate_setpoint",
            subcategory="conversational",
            utterance="Could you set the family room thermostat to 68 degrees for me?",
            home=_home(
                _entity(name="Family Room Thermostat", kind="climate", area="Family Room", state="cool"),
                satellite_area="Family Room",
                home_id="v3_gold_climate_b",
            ),
            expected=_action(
                {
                    "name": "HassClimateSetTemperature",
                    "arguments": {"name": "Family Room Thermostat", "temperature": 68},
                }
            ),
        ),
        _spec(
            row_id="media_play",
            category="media_play",
            subcategory="named_device",
            utterance="Play Home Theater TV",
            home=_home(home_theater_tv, satellite_area="Home Theater", home_id="v3_gold_media_play"),
            expected=_action({"name": "HassMediaUnpause", "arguments": {"name": "Home Theater TV", "domain": ["media_player"], "device_class": ["tv"]}}),
        ),
        _spec(
            row_id="media_pause",
            category="media_pause",
            subcategory="named_device",
            utterance="Pause Rec Room TV",
            home=_home(rec_room_tv, satellite_area="Rec Room", home_id="v3_gold_media_pause"),
            expected=_action({"name": "HassMediaPause", "arguments": {"name": "Rec Room TV", "domain": ["media_player"], "device_class": ["tv"]}}),
        ),
        _spec(
            row_id="media_volume",
            category="media_volume",
            subcategory="absolute",
            utterance="Set Family Room TV volume to 45 percent",
            home=_home(family_room_tv, satellite_area="Family Room", home_id="v3_gold_media_volume"),
            expected=_action(
                {
                    "name": "HassSetVolume",
                    "arguments": {"name": "Family Room TV", "domain": ["media_player"], "device_class": ["tv"], "volume_level": 45},
                }
            ),
        ),
        _spec(
            row_id="media_volume_rel",
            category="media_volume",
            subcategory="relative",
            utterance="Turn up Guest Suite TV volume",
            home=_home(bedroom_tv, satellite_area="Guest Suite", home_id="v3_gold_media_volume_rel"),
            expected=_action(
                {
                    "name": "HassSetVolumeRelative",
                    "arguments": {"name": "Guest Suite TV", "volume_step": "up"},
                }
            ),
        ),
        _spec(
            row_id="media_mute",
            category="media_mute",
            subcategory="named_device",
            utterance="Mute Home Theater TV",
            home=_home(home_theater_tv, satellite_area="Home Theater", home_id="v3_gold_media_mute"),
            expected=_action({"name": "HassMediaPlayerMute", "arguments": {"name": "Home Theater TV", "domain": ["media_player"], "device_class": ["tv"]}}),
        ),
        _spec(
            row_id="timer_start",
            category="timer_start",
            subcategory="minutes",
            utterance="Start a 15 minute timer",
            home=_home(sunroom_light, satellite_area="Sunroom", home_id="v3_gold_timer_start"),
            expected=_action({"name": "HassStartTimer", "arguments": {"minutes": 15}}),
            target_names=[],
        ),
        _spec(
            row_id="timer_start_named",
            category="timer_start",
            subcategory="hours",
            utterance="Start a 1 hour timer",
            home=_home(sunroom_light, satellite_area="Sunroom", home_id="v3_gold_timer_start_named"),
            expected=_action({"name": "HassStartTimer", "arguments": {"hours": 1}}),
            target_names=[],
        ),
        _spec(
            row_id="timer_pause",
            category="timer_pause",
            subcategory="generic",
            utterance="Pause the timer",
            home=_home(sunroom_light, satellite_area="Sunroom", home_id="v3_gold_timer_pause"),
            expected=_action({"name": "HassPauseTimer", "arguments": {}}),
            target_names=[],
        ),
        _spec(
            row_id="timer_status",
            category="timer_status",
            subcategory="generic",
            utterance="What is the timer status?",
            home=_home(sunroom_light, satellite_area="Sunroom", home_id="v3_gold_timer_status"),
            expected=_action({"name": "HassTimerStatus", "arguments": {}}),
            target_names=[],
        ),
        _spec(
            row_id="timer_cancel_all",
            category="timer_cancel",
            subcategory="all",
            utterance="Cancel all timers",
            home=_home(sunroom_light, satellite_area="Sunroom", home_id="v3_gold_timer_cancel_all"),
            expected=_action({"name": "HassCancelAllTimers", "arguments": {}}),
            target_names=[],
        ),
        _spec(
            row_id="timer_cancel_named",
            category="timer_cancel",
            subcategory="area",
            utterance="Cancel all timers in the Sunroom",
            home=_home(sunroom_light, satellite_area="Sunroom", home_id="v3_gold_timer_cancel_named"),
            expected=_action({"name": "HassCancelAllTimers", "arguments": {"area": "Sunroom"}}),
            target_names=[],
        ),
        _spec(
            row_id="vacuum_start",
            category="vacuum_start",
            subcategory="named_device",
            utterance="Start Upstairs Robot Vacuum",
            home=_home(upstairs_vacuum, satellite_area="Sunroom", home_id="v3_gold_vacuum_start"),
            expected=_action({"name": "HassVacuumStart", "arguments": {"name": "Upstairs Robot Vacuum", "domain": ["vacuum"]}}),
        ),
        _spec(
            row_id="vacuum_return",
            category="vacuum_return",
            subcategory="named_device",
            utterance="Send Mudroom Robot Vacuum home",
            home=_home(mudroom_vacuum, satellite_area="Mudroom", home_id="v3_gold_vacuum_return"),
            expected=_action({"name": "HassVacuumReturnToBase", "arguments": {"name": "Mudroom Robot Vacuum", "domain": ["vacuum"]}}),
        ),
        _spec(
            row_id="vacuum_clean_area",
            category="vacuum_clean_area",
            subcategory="area",
            utterance="Vacuum the Sunroom",
            home=_home(upstairs_vacuum, satellite_area="Sunroom", home_id="v3_gold_vacuum_clean_area"),
            expected=_action(
                {
                    "name": "HassVacuumCleanArea",
                    "arguments": {"name": "Upstairs Robot Vacuum", "area": "Sunroom"},
                }
            ),
        ),
        _spec(
            row_id="scene_a",
            category="scene_activate",
            subcategory="named_scene",
            utterance="Activate Movie Night Scene",
            home=_home(movie_scene, satellite_area="Home Theater", home_id="v3_gold_scene_a"),
            expected=_action(_turn_on(movie_scene)),
        ),
        _spec(
            row_id="scene_b",
            category="scene_activate",
            subcategory="conversational",
            utterance="Run the bedtime scene in the guest suite",
            home=_home(bedtime_scene, satellite_area="Guest Suite", home_id="v3_gold_scene_b"),
            expected=_action(_turn_on(bedtime_scene)),
        ),
        _spec(
            row_id="script_a",
            category="script_run",
            subcategory="named_script",
            utterance="Run Good Morning Script",
            home=_home(morning_script, satellite_area="Kitchen", home_id="v3_gold_script_a"),
            expected=_action(_run_script(morning_script)),
        ),
        _spec(
            row_id="script_b",
            category="script_run",
            subcategory="conversational",
            utterance="Please start the leave home script",
            home=_home(away_script, satellite_area="Foyer", home_id="v3_gold_script_b"),
            expected=_action(_run_script(away_script)),
        ),
        _spec(
            row_id="ordinary_on",
            category="ordinary_on",
            subcategory="light",
            utterance="Turn on Dining Room Pendant Light",
            home=_home(dining_light, satellite_area="Dining Room", home_id="v3_gold_ordinary_on"),
            expected=_action(_turn_on(dining_light)),
        ),
        _spec(
            row_id="ordinary_off",
            category="ordinary_off",
            subcategory="fan",
            utterance="Turn off Study Desk Fan",
            home=_home(study_fan, satellite_area="Study", home_id="v3_gold_ordinary_off"),
            expected=_action(_turn_off(study_fan)),
        ),
        # HassLightSet and HassFanSetSpeed are 12.3% of training calls. Until these
        # rows existed no v3 suite covered either, so the light/fan argument
        # confusion that Run 008 failed on was invisible to gold and shadow.
        _spec(
            row_id="light_brightness",
            category="light_brightness",
            subcategory="named_device",
            utterance="Set Dining Room Pendant Light brightness to 40 percent",
            home=_home(dining_light, satellite_area="Dining Room", home_id="v3_gold_light_brightness"),
            expected=_action(_light_set(dining_light, 40)),
        ),
        _spec(
            row_id="light_brightness_conversational",
            category="light_brightness",
            subcategory="conversational",
            utterance="Could you dim the Sunroom Accent Light to 25 percent?",
            home=_home(sunroom_light, satellite_area="Sunroom", home_id="v3_gold_light_brightness_b"),
            expected=_action(_light_set(sunroom_light, 25)),
        ),
        _spec(
            row_id="fan_speed",
            category="fan_speed",
            subcategory="named_device",
            utterance="Set Study Desk Fan speed to 40 percent",
            home=_home(study_fan, satellite_area="Study", home_id="v3_gold_fan_speed"),
            expected=_action(_fan_speed(study_fan, 40)),
        ),
        # Same number, same phrasing shape, both devices present: the row fails if
        # the model reaches for brightness on a fan or speed on a light.
        _spec(
            row_id="fan_speed_contrastive",
            category="fan_speed",
            subcategory="light_fan_contrast",
            utterance="Set Study Desk Fan speed to 60 percent",
            home=_home(
                study_fan,
                dining_light,
                satellite_area="Study",
                home_id="v3_gold_fan_speed_contrast",
            ),
            expected=_action(_fan_speed(study_fan, 60)),
        ),
        _spec(
            row_id="light_brightness_contrastive",
            category="light_brightness",
            subcategory="light_fan_contrast",
            utterance="Set Dining Room Pendant Light brightness to 60 percent",
            home=_home(
                dining_light,
                study_fan,
                satellite_area="Dining Room",
                home_id="v3_gold_light_brightness_contrast",
            ),
            expected=_action(_light_set(dining_light, 60)),
        ),
        _spec(
            row_id="status_media",
            category="status",
            subcategory="media_player",
            utterance="What is the status of Home Theater TV?",
            home=_home(home_theater_tv, satellite_area="Home Theater", home_id="v3_gold_status_media"),
            expected=_status(home_theater_tv),
            target_names=["Home Theater TV"],
        ),
        _spec(
            row_id="status_climate",
            category="status",
            subcategory="climate",
            utterance="Is the Great Room Thermostat heating?",
            home=_home(great_room_thermostat, satellite_area="Great Room", home_id="v3_gold_status_climate"),
            expected=_status(great_room_thermostat),
            target_names=["Great Room Thermostat"],
        ),
        _spec(
            row_id="ambiguity_resolve",
            category="ambiguity",
            subcategory="one_default_light",
            utterance="Turn on the reading lamp",
            home=_home(library_light_a, library_light_b, satellite_area="Library", home_id="v3_gold_ambiguity_resolve"),
            expected=_action(_turn_on(library_light_a)),
        ),
        _spec(
            row_id="ambiguity_clarify",
            category="ambiguity",
            subcategory="two_lights",
            utterance="Switch on the accent light",
            home=_home(
                _entity(name="Sunroom Table Lamp", kind="light", area="Sunroom", aliases=["accent light"]),
                _entity(name="Sunroom Floor Lamp", kind="light", area="Sunroom", aliases=["accent light"]),
                satellite_area="Sunroom",
                home_id="v3_gold_ambiguity_clarify",
            ),
            expected=_no_action("clarify"),
            target_names=[],
            request_hint="switch on the accent light",
        ),
        _spec(
            row_id="ambiguity_area_unavailable",
            category="ambiguity",
            subcategory="zero_lights",
            utterance="Turn on the desk light",
            home=_home(study_fan, satellite_area="Study", home_id="v3_gold_ambiguity_area"),
            expected=_no_action(
                "area_unavailable",
                unavailable={"area": "study", "type": "lights"},
            ),
            target_names=[],
            request_hint="turn on the desk light",
        ),
        _spec(
            row_id="unsupported_lawn",
            category="unsupported_no_action",
            subcategory="lawn_mower",
            utterance="Start the lawn mower in the courtyard",
            home=_home(sunroom_light, satellite_area="Sunroom", home_id="v3_gold_unsupported_lawn"),
            expected=_no_action("unsupported"),
            target_names=[],
            request_hint="start the lawn mower in the courtyard",
        ),
        _spec(
            row_id="unsupported_todo",
            category="unsupported_no_action",
            subcategory="todo",
            utterance="Add milk to the shopping list",
            home=_home(sunroom_light, satellite_area="Sunroom", home_id="v3_gold_unsupported_todo"),
            expected=_no_action("unsupported"),
            target_names=[],
            request_hint="add milk to the shopping list",
        ),
        _spec(
            row_id="no_call_clarify",
            category="unsupported_no_action",
            subcategory="incomplete",
            utterance="Set the thermostat to",
            home=_home(great_room_thermostat, satellite_area="Great Room", home_id="v3_gold_no_call_clarify"),
            expected=_no_action("clarify"),
            target_names=[],
            request_hint="set the thermostat to",
        ),
    ]
    return rows


def _utterance_for_spec(spec: dict[str, Any]) -> str:
    utterance = expand_utterance(spec)
    if isinstance(utterance, str) and utterance.strip():
        return utterance.strip()
    expected = spec.get("expected") or {}
    if expected.get("kind") == "no_action":
        return str(spec.get("request_hint") or "unsupported request")
    calls = expected.get("calls") or []
    if not calls:
        return str(spec.get("request_hint") or "")
    target_names = spec.get("target_names") or []
    target = target_names[0] if target_names else ""
    phrase = _phrase_for_call(target, calls[0])
    if phrase:
        return phrase[0].upper() + phrase[1:]
    return request_seed_from_spec(spec)


@dataclass(frozen=True, slots=True)
class _ShadowFamily:
    """One shadow concept and how many rows of it a run of ``count`` gets.

    ``floor``/``divisor`` are the quota: at least ``floor`` rows, otherwise
    ``count // divisor``. ``offset`` staggers which area each family starts on
    so two families never keep producing the same room. ``build`` returns the
    entities the row's home contains plus everything about the expectation.
    """

    category: str
    slug: str
    floor: int
    divisor: int
    offset: int
    build: Callable[[int, str], dict[str, Any]]


def _rows(**fields: Any) -> dict[str, Any]:
    """A shadow row's varying half; the loop supplies the rest."""
    return fields


# Timers belong to the home, not to a device, so every timer row reuses one
# light rather than inventing a device the request never mentions.
_TIMER_LIGHT = _entity(name="Porch Accent Light", kind="light", area="Porch")

_TV_TARGET = {"domain": ["media_player"], "device_class": ["tv"]}


def _shadow_climate(index: int, area: str) -> dict[str, Any]:
    thermostat = _entity(
        name=f"{area} Comfort Thermostat", kind="climate", area=area, state="heat"
    )
    return _rows(
        entities=[thermostat],
        subcategory="named_device",
        expected=_action(
            {
                "name": "HassClimateSetTemperature",
                "arguments": {
                    "name": thermostat["name"],
                    "temperature": 66 + (index * 3) % 8,
                },
            }
        ),
        target_names=[thermostat["name"]],
    )


def _shadow_media(name_suffix: str, tool: str) -> Callable[[int, str], dict[str, Any]]:
    """Play, pause and mute differ only by the tool and the TV's name."""

    def build(index: int, area: str) -> dict[str, Any]:
        aliases = ["tv"] if name_suffix == "Wall TV" else None
        tv = _entity(
            name=f"{area} {name_suffix}", kind="media_player", area=area, aliases=aliases
        )
        return _rows(
            entities=[tv],
            subcategory="named_device",
            expected=_action(
                {"name": tool, "arguments": {"name": tv["name"], **_TV_TARGET}}
            ),
            target_names=[tv["name"]],
        )

    return build


def _shadow_media_volume(index: int, area: str) -> dict[str, Any]:
    """Alternate absolute and relative volume; the model confuses the two."""
    tv = _entity(name=f"{area} Soundbar TV", kind="media_player", area=area)
    if index % 2 == 0:
        expected = _action(
            {
                "name": "HassSetVolume",
                "arguments": {
                    "name": tv["name"],
                    **_TV_TARGET,
                    "volume_level": 25 + index * 7,
                },
            }
        )
        subcategory = "absolute"
    else:
        expected = _action(
            {
                "name": "HassSetVolumeRelative",
                "arguments": {"name": tv["name"], "volume_step": "down"},
            }
        )
        subcategory = "relative"
    return _rows(
        entities=[tv],
        subcategory=subcategory,
        expected=expected,
        target_names=[tv["name"]],
    )


def _shadow_timer_start(index: int, area: str) -> dict[str, Any]:
    minutes = 7 + index * 9
    return _rows(
        entities=[_TIMER_LIGHT],
        subcategory="minutes",
        expected=_action({"name": "HassStartTimer", "arguments": {"minutes": minutes}}),
        target_names=[],
        utterance=f"Set a {minutes} minute timer in the {area}",
    )


def _shadow_timer(
    tool: str, subcategory: str, phrasing: str
) -> Callable[[int, str], dict[str, Any]]:
    """Pause and status are one argument-free tool call and one sentence."""

    def build(_index: int, area: str) -> dict[str, Any]:
        return _rows(
            entities=[_TIMER_LIGHT],
            subcategory=subcategory,
            expected=_action({"name": tool, "arguments": {}}),
            target_names=[],
            utterance=phrasing.format(area=area),
        )

    return build


def _shadow_timer_cancel(index: int, area: str) -> dict[str, Any]:
    """Cancelling everywhere and cancelling in one area are the same tool."""
    scoped = index % 2
    return _rows(
        entities=[_TIMER_LIGHT],
        subcategory="area" if scoped else "all",
        expected=_action(
            {
                "name": "HassCancelAllTimers",
                "arguments": {"area": area} if scoped else {},
            }
        ),
        target_names=[],
        utterance=(
            f"Clear all timers in the {area}"
            if scoped
            else f"Stop every running timer in the {area}"
        ),
    )


def _shadow_vacuum(
    name_suffix: str, tool: str, subcategory: str, *, state: str = "off", with_area: bool = False
) -> Callable[[int, str], dict[str, Any]]:
    def build(_index: int, area: str) -> dict[str, Any]:
        vacuum = _entity(
            name=f"{area} {name_suffix}", kind="vacuum", area=area, state=state
        )
        arguments = {"name": vacuum["name"]}
        arguments |= {"area": area} if with_area else {"domain": ["vacuum"]}
        return _rows(
            entities=[vacuum],
            subcategory=subcategory,
            expected=_action({"name": tool, "arguments": arguments}),
            target_names=[vacuum["name"]],
        )

    return build


def _shadow_scene(_index: int, area: str) -> dict[str, Any]:
    scene = _entity(name=f"{area} Relax Scene", kind="scene", area=area)
    return _rows(
        entities=[scene],
        subcategory="named_scene",
        expected=_action(_turn_on(scene)),
        target_names=[scene["name"]],
    )


def _shadow_script(_index: int, area: str) -> dict[str, Any]:
    script = _entity(name=f"{area} Away Script", kind="script", area=area)
    return _rows(
        entities=[script],
        subcategory="named_script",
        expected=_action(_run_script(script)),
        target_names=[script["name"]],
        utterance=f"turn on {script['name']}",
    )


def _shadow_ordinary(
    name_suffix: str, kind: str, alias: str, on: bool
) -> Callable[[int, str], dict[str, Any]]:
    def build(_index: int, area: str) -> dict[str, Any]:
        item = _entity(
            name=f"{area} {name_suffix}", kind=kind, area=area, aliases=[alias]
        )
        return _rows(
            entities=[item],
            subcategory=kind,
            expected=_action(_turn_on(item) if on else _turn_off(item)),
            target_names=[item["name"]],
        )

    return build


def _shadow_setting(
    kind: str,
    name_suffix: str,
    other_suffix: str,
    other_kind: str,
    base: int,
    span: int,
    word: str,
) -> Callable[[int, str], dict[str, Any]]:
    """Brightness and speed, with the other device present half the time.

    The contrast is the point: a row fails if the model reaches for brightness
    on a fan or speed on a light.
    """

    def build(index: int, area: str) -> dict[str, Any]:
        item = _entity(
            name=f"{area} {name_suffix}", kind=kind, area=area, aliases=[kind]
        )
        value = base + (index * 15) % span
        entities = [item]
        if index % 2:
            entities.append(
                _entity(name=f"{area} {other_suffix}", kind=other_kind, area=area)
            )
        call = _light_set if kind == "light" else _fan_speed
        return _rows(
            entities=entities,
            subcategory="light_fan_contrast" if index % 2 else "named_device",
            expected=_action(call(item, value)),
            target_names=[item["name"]],
            utterance=f"set {item['name']} {word} to {value} percent",
        )

    return build


def _shadow_status(index: int, area: str) -> dict[str, Any]:
    if index % 2 == 0:
        target = _entity(name=f"{area} Lounge TV", kind="media_player", area=area)
        subcategory = "media_player"
    else:
        target = _entity(name=f"{area} Thermostat", kind="climate", area=area)
        subcategory = "climate"
    return _rows(
        entities=[target],
        subcategory=subcategory,
        expected=_status(target),
        target_names=[target["name"]],
    )


def _shadow_ambiguity(index: int, area: str) -> dict[str, Any]:
    """One clear target, two equally good ones, and none at all."""
    if index % 3 == 0:
        light = _entity(
            name=f"{area} Reading Lamp", kind="light", area=area, aliases=["reading lamp"]
        )
        entities, expected = [light], _action(_turn_on(light))
        hint = "turn on the reading lamp"
        utterance = f"Please switch on the {area} reading lamp"
    elif index % 3 == 1:
        entities = [
            _entity(name=f"{area} Lamp A", kind="light", area=area, aliases=["accent light"]),
            _entity(name=f"{area} Lamp B", kind="light", area=area, aliases=["accent light"]),
        ]
        expected = _no_action("clarify")
        hint = "enable the mood light"
        utterance = f"Enable the mood light in the {area}"
    else:
        entities = [_entity(name=f"{area} Desk Fan", kind="fan", area=area)]
        expected = _no_action(
            "area_unavailable", unavailable={"area": area.casefold(), "type": "lights"}
        )
        hint = "turn on the desk light"
        utterance = f"Turn on the {area} desk light"
    return _rows(
        entities=entities,
        subcategory="generic",
        expected=expected,
        target_names=[],
        request_hint=hint,
        utterance=utterance,
    )


def _shadow_unsupported(index: int, area: str) -> dict[str, Any]:
    hints = (
        f"mow the front lawn with the robot mower in the {area}",
        f"press the panic button in the {area}",
        f"add batteries to the {area} shopping list",
    )
    hint = hints[index % len(hints)]
    return _rows(
        entities=[_entity(name=f"{area} Hall Light", kind="light", area=area)],
        subcategory="unsupported",
        expected=_no_action("unsupported"),
        target_names=[],
        request_hint=hint,
        utterance=hint,
    )


# Every shadow concept, in the order rows are minted. Order is load-bearing:
# it fixes the row numbering and decides which family absorbs a rounding
# remainder.
_SHADOW_FAMILIES: tuple[_ShadowFamily, ...] = (
    _ShadowFamily("climate_setpoint", "climate", 2, 12, 0, _shadow_climate),
    _ShadowFamily("media_play", "media_play", 1, 20, 2, _shadow_media("Wall TV", "HassMediaUnpause")),
    _ShadowFamily("media_pause", "media_pause", 1, 20, 4, _shadow_media("Corner TV", "HassMediaPause")),
    _ShadowFamily("media_volume", "media_volume", 2, 15, 1, _shadow_media_volume),
    _ShadowFamily("media_mute", "media_mute", 1, 20, 6, _shadow_media("Bedroom TV", "HassMediaPlayerMute")),
    _ShadowFamily("timer_start", "timer_start", 2, 15, 3, _shadow_timer_start),
    _ShadowFamily("timer_pause", "timer_pause", 1, 25, 5, _shadow_timer("HassPauseTimer", "generic", "Hold the {area} countdown timer")),
    _ShadowFamily("timer_status", "timer_status", 1, 25, 7, _shadow_timer("HassTimerStatus", "generic", "How much time is left on the {area} timer?")),
    _ShadowFamily("timer_cancel", "timer_cancel", 2, 15, 8, _shadow_timer_cancel),
    _ShadowFamily("vacuum_start", "vacuum_start", 1, 20, 9, _shadow_vacuum("Floor Vacuum", "HassVacuumStart", "named_device")),
    _ShadowFamily("vacuum_return", "vacuum_return", 1, 20, 10, _shadow_vacuum("Robot Vacuum", "HassVacuumReturnToBase", "named_device", state="cleaning")),
    _ShadowFamily("vacuum_clean_area", "vacuum_area", 1, 20, 11, _shadow_vacuum("Robot Vacuum", "HassVacuumCleanArea", "area", with_area=True)),
    _ShadowFamily("scene_activate", "scene", 2, 15, 0, _shadow_scene),
    _ShadowFamily("script_run", "script", 2, 15, 2, _shadow_script),
    _ShadowFamily("ordinary_on", "on", 2, 15, 4, _shadow_ordinary("Task Light", "light", "light", True)),
    _ShadowFamily("ordinary_off", "off", 1, 20, 6, _shadow_ordinary("Ceiling Fan", "fan", "fan", False)),
    _ShadowFamily("light_brightness", "bright", 2, 15, 8, _shadow_setting("light", "Reading Lamp", "Column Fan", "fan", 20, 80, "brightness")),
    _ShadowFamily("fan_speed", "speed", 2, 15, 10, _shadow_setting("fan", "Column Fan", "Reading Lamp", "light", 25, 75, "speed")),
    _ShadowFamily("status", "status", 2, 15, 1, _shadow_status),
    _ShadowFamily("ambiguity", "ambiguity", 3, 10, 3, _shadow_ambiguity),
    _ShadowFamily("unsupported_no_action", "unsupported", 2, 15, 5, _shadow_unsupported),
)


def _shadow_quotas(count: int) -> dict[str, int]:
    """Split ``count`` rows across the families, to the row.

    Rounding is taken off whichever family is currently largest and any
    shortfall goes to ambiguity, which is the concept that most rewards extra
    examples.
    """
    quotas = {
        family.category: max(family.floor, count // family.divisor)
        for family in _SHADOW_FAMILIES
    }
    while sum(quotas.values()) > count:
        quotas[max(quotas, key=lambda name: quotas[name])] -= 1
    while sum(quotas.values()) < count:
        quotas["ambiguity"] += 1
    return quotas


def build_shadow_specs(
    seed: int = 20260906, count: int = DEFAULT_SHADOW_COUNT
) -> list[dict[str, Any]]:
    """Build shadow rows that mirror gold concepts with fresh homes and phrasing."""
    if not SHADOW_MIN <= count <= SHADOW_MAX:
        raise ValueError(f"shadow count must be {SHADOW_MIN}-{SHADOW_MAX}, got {count}")
    quotas = _shadow_quotas(count)

    specs: list[dict[str, Any]] = []
    for family in _SHADOW_FAMILIES:
        for index in range(quotas[family.category]):
            area = _SHADOW_AREAS[(index + family.offset) % len(_SHADOW_AREAS)]
            row = family.build(index, area)
            # One row, one id: the home is named after the row it belongs to.
            candidate_id = f"v3_shadow_{family.slug}_{len(specs):04d}"
            specs.append(
                _shadow_spec_shell(
                    candidate_id=candidate_id,
                    seed=seed,
                    category=family.category,
                    subcategory=row["subcategory"],
                    home=_home(
                        *row["entities"],
                        satellite_area=area,
                        home_id=candidate_id,
                    ),
                    expected=row["expected"],
                    target_names=row["target_names"],
                    request_hint=row.get("request_hint", ""),
                    utterance=row.get("utterance"),
                )
            )

    for spec in specs:
        if not spec.get("utterance"):
            spec["utterance"] = _utterance_for_spec(spec)
    return specs[:count]


def gold_user_prompts() -> set[str]:
    return {spec["utterance"] for spec in gold_specs()}


def shadow_user_prompts() -> set[str]:
    return {spec["utterance"] for spec in build_shadow_specs()}


def excluded_train_prompts() -> set[str]:
    """Normalized user prompts that must not appear in train JSONL overlap checks."""
    excluded = {_normalized(text) for text in gold_user_prompts()}
    excluded.update(_normalized(text) for text in shadow_user_prompts())
    excluded.update(_normalized(text) for text in recipe_lock_prompts())
    excluded.update(_normalized(text) for text in _grounding_prompts())
    return excluded


def _grounding_prompts() -> list[str]:
    """Entity-grounding regressions. Imported lazily: evals.grounding_eval imports
    the generators, which import this module for the exclusion set.
    """
    try:
        from evals.grounding_eval import grounding_user_prompts
    except ImportError:  # pragma: no cover - eval package incomplete
        return []
    return grounding_user_prompts()


def build_gold_examples() -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for spec in gold_specs():
        example = render_example(spec)
        example["metadata"]["quality_eval"] = True
        example["metadata"]["v3_quality_gold"] = True
        examples.append(example)
    return examples


def build_shadow_examples(seed: int = 20260906, count: int = DEFAULT_SHADOW_COUNT) -> list[dict[str, Any]]:
    specs = build_shadow_specs(seed=seed, count=count)
    rows = [render_example(spec) for spec in specs]
    for row in rows:
        row["metadata"]["shadow_eval"] = True
        row["metadata"]["v3_quality_shadow"] = True
    return rows


def assert_quality_eval_contract(example: dict[str, Any]) -> None:
    """Validate one rendered v3 row against the pinned tool contract."""
    calls = assert_row_contract(example, "v3 quality eval")
    schemas = tool_schema_map(v2_openai_tools())
    # Per-script tools are named after the script, so they are absent from the
    # pinned catalog by design; they must still be offered to the row and take
    # no arguments.
    offered = {tool["function"]["name"] for tool in example.get("tools") or []}
    prompt = "".join(
        str(message.get("content") or "")
        for message in example.get("messages") or []
        if message.get("role") == "system"
    )
    for call in calls:
        parsed = parse_tool_arguments(call["function"]["arguments"]) or {}
        name = call["function"]["name"]
        # A target the prompt never names cannot be answered: the row is
        # unscoreable, not hard. Scripts are excluded from the overview, so
        # they ground on the tool instead.
        target = parsed.get("name")
        if isinstance(target, str) and target not in prompt:
            raise ValueError(
                f"v3 quality eval targets a name absent from the prompt: {target}"
            )
        if name not in schemas:
            if name not in offered:
                raise ValueError(
                    f"v3 quality eval calls a tool the row never offered: {name}"
                )
            if parsed:
                raise ValueError("v3 quality eval script tools take no arguments")
            continue
        if reason := validate_tool_arguments(name, parsed, schemas):
            raise ValueError(
                f"v3 quality eval arguments failed schema validation: {reason}"
            )


def v3_quality_summary() -> dict[str, Any]:
    gold = gold_specs()
    shadow = build_shadow_specs()
    return {
        "gold_count": len(gold),
        "shadow_count": len(shadow),
        "gold_categories": sorted({spec["category"] for spec in gold}),
        "shadow_categories": sorted({spec["category"] for spec in shadow}),
        "gold_user_prompts": sorted(gold_user_prompts()),
    }
