"""Fetch a real Home Assistant home as a `generators.homes.generate_home` dict.

Two REST calls: `/api/states` for entities and their attributes, `/api/template`
for the area/floor map (the states API does not carry either). Output is the
same shape `generate_home` returns, so it feeds the v3 pipeline and the eval
builders without a second format.

    HA_URL=http://192.168.1.35:8123 HA_TOKEN=... \
      python training/scripts/fetch_ha_home.py --out training/fixtures/real_home.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Any

try:  # importable both as `scripts.fetch_ha_home` and as a standalone script
    from scripts.ha_websocket import fetch_assist_registry
except ImportError:  # pragma: no cover - depends on how the module was reached
    from ha_websocket import fetch_assist_registry

# Reverse of homes._KIND_MAP. Domains absent here are not part of the trained
# capability contract, which drops sensors, updates, automations and the rest of
# the 900-entity tail a real home carries.
CAPABILITY_BY_DOMAIN: dict[str, str] = {
    "light": "lights",
    "fan": "fans",
    "switch": "switches",
    "cover": "covers",
    "lock": "locks",
    "media_player": "media_players",
    "climate": "climate",
    "vacuum": "vacuums",
    "scene": "scenes",
    "script": "scripts",
    "lawn_mower": "lawn_mowers",
    "todo": "todo_lists",
    "button": "buttons",
}

# Home Assistant does not auto-expose `button` to Assist, and a real home's
# button domain is almost entirely per-device diagnostics ("<device> Restart").
# Pull them in with --domains if a home actually voice-controls buttons.
DEFAULT_DOMAINS: tuple[str, ...] = tuple(d for d in CAPABILITY_BY_DOMAIN if d != "button")

# ponytail: attribute-derived features for the domains where the model has to
# distinguish them; everything else takes the capability default.
_LIGHT_COLOR_MODES = {
    "brightness": ("brightness",),
    "color_temp": ("brightness", "color_temp"),
    "hs": ("brightness", "color"),
    "rgb": ("brightness", "color"),
    "rgbw": ("brightness", "color"),
    "rgbww": ("brightness", "color", "color_temp"),
    "xy": ("brightness", "color"),
}

# MediaPlayerEntityFeature bits -> the feature names generators.capability_registry
# gates operations on. An Echo Dot reports no TURN_ON/TURN_OFF and a dumb TV
# reports no SEARCH_MEDIA, so assuming power, pause, volume and mute everywhere
# produces labels Home Assistant refuses at runtime.
_MEDIA_FEATURE_BITS: tuple[tuple[int, str], ...] = (
    (1, "pause"),
    (4, "volume"),
    (8, "mute"),
    (16, "previous"),
    (32, "next"),
    (128, "on"),
    (256, "off"),
    (1024, "volume_step"),
    (16384, "play"),
    (4194304, "search"),
)

# FanEntityFeature.SET_SPEED
_FAN_SET_SPEED = 1

# Voice-assistant bridges (Alexa, Google) name entities after the linked account,
# so a real home carries the owner's email in entity names and ids. Names and
# possessives stay -- they are the shape the model has to learn -- but an email
# address is not something to put in weights.
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def _strip_email(name: str, entity_id: str) -> tuple[str, str]:
    """Remove any email address from a name and its slug from the entity id."""
    found = _EMAIL.search(name)
    if found is None:
        return name, entity_id
    email = found.group(0)
    cleaned = _EMAIL.sub("", name).strip(" -_")
    slug = "".join(char if char.isalnum() else "_" for char in email.casefold())
    domain, _, object_id = entity_id.partition(".")
    object_id = object_id.replace(slug, "").strip("_")
    # A name that was only an email leaves the object id to carry it.
    return (cleaned or object_id.replace("_", " ").title() or domain), f"{domain}.{object_id}"


_AREA_FLOOR_TEMPLATE = (
    "{% set ns = namespace(rows=[]) %}"
    "{% for s in states %}"
    "{% set a = area_id(s.entity_id) %}"
    "{% set ns.rows = ns.rows + [[s.entity_id, area_name(s.entity_id),"
    " floor_name(a) if a else none]] %}"
    "{% endfor %}"
    "{{ ns.rows | tojson }}"
)


def _request(url: str, token: str, payload: dict[str, Any] | None = None) -> Any:
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - operator-supplied URL
        return json.loads(response.read())


def fetch_raw(base_url: str, token: str) -> tuple[list[dict[str, Any]], list[list[Any]]]:
    """Return `/api/states` and the `[entity_id, area, floor]` rows."""
    base = base_url.rstrip("/")
    states = _request(f"{base}/api/states", token)
    areas = _request(f"{base}/api/template", token, {"template": _AREA_FLOOR_TEMPLATE})
    if isinstance(areas, str):  # /api/template returns text for some HA versions
        areas = json.loads(areas)
    return states, areas


def _features(domain: str, attributes: dict[str, Any]) -> tuple[str, ...] | None:
    supported = attributes.get("supported_features") or 0
    if not isinstance(supported, int):
        supported = 0
    if domain == "light":
        modes = attributes.get("supported_color_modes") or []
        features = {"on", "off"}
        for mode in modes:
            features.update(_LIGHT_COLOR_MODES.get(mode, ()))
        return tuple(sorted(features))
    if domain == "fan":
        features = {"on", "off"}
        if supported & _FAN_SET_SPEED or "percentage" in attributes:
            features.add("percentage")
        return tuple(sorted(features))
    if domain == "media_player":
        features = {name for bit, name in _MEDIA_FEATURE_BITS if supported & bit}
        # A player that reports pause without play still resumes; HA's unpause
        # intent uses media_play_pause for exactly this case.
        if "pause" in features:
            features.add("play")
        return tuple(sorted(features))
    return None


def build_home(
    states: list[dict[str, Any]],
    area_rows: list[list[Any]],
    *,
    home_id: str = "real_home",
    sayso_entity_area: str | None = None,
    domains: tuple[str, ...] = DEFAULT_DOMAINS,
    keep_entities: frozenset[str] = frozenset(),
    exposed_entities: frozenset[str] | None = None,
    aliases: dict[str, list[str]] | None = None,
    exposure_source: str = "domain_filter",
) -> dict[str, Any]:
    """Assemble the home dict from raw Home Assistant payloads.

    ``exposed_entities`` is the Assist exposure list from the websocket API. When
    supplied it is authoritative: an entity Home Assistant does not expose to the
    conversation agent never reaches the corpus, whatever its domain.
    """
    from generators.homes import make_entity  # local: keeps the module importable standalone

    placement = {row[0]: (row[1], row[2]) for row in area_rows}
    aliases = aliases or {}
    entities: list[dict[str, Any]] = []
    for state in sorted(states, key=lambda s: s["entity_id"]):
        entity_id = state["entity_id"]
        domain = entity_id.split(".", 1)[0]
        capability = CAPABILITY_BY_DOMAIN.get(domain)
        if capability is None:
            continue
        if exposed_entities is not None and entity_id not in exposed_entities:
            continue
        if domain not in domains and entity_id not in keep_entities:
            continue
        attributes = state.get("attributes") or {}
        if attributes.get("hidden_by") or attributes.get("restored"):
            continue
        area, floor = placement.get(entity_id, (None, None))
        name = attributes.get("friendly_name") or entity_id.split(".", 1)[1]
        entity_aliases = [alias.strip() for alias in aliases.get(entity_id) or [] if isinstance(alias, str) and alias.strip()]
        name, entity_id = _strip_email(name, entity_id)
        entity = make_entity(
            name=name,
            capability=capability,
            area=area or "Unassigned",
            floor=floor or "Main Floor",
            rng=None,  # unused: state is supplied
            state=state.get("state") or "unknown",
            features=_features(domain, attributes),
            aliases=list(dict.fromkeys([name, *entity_aliases])),
            device_class=attributes.get("device_class"),
        )
        # Home Assistant, not the slug rule, owns the entity id.
        entity["entity_id"] = entity_id
        entities.append(entity)

    timers = [
        {
            "name": _strip_email(
                (state.get("attributes") or {}).get("friendly_name", state["entity_id"]),
                state["entity_id"],
            )[0],
            "remaining_seconds": _remaining_seconds(state),
            "area": placement.get(state["entity_id"], (None, None))[0] or "Unassigned",
        }
        for state in states
        if state["entity_id"].startswith("timer.") and state.get("state") == "active"
    ]

    default_area = next((e["area"] for e in entities if e["area"] != "Unassigned"), "Unassigned")
    # Same keys generate_home returns: build_scenario injects a missing capability
    # into whatever home it is handed and needs the area/floor map to place it.
    area_floors: dict[str, str] = {}
    for entity in entities:
        area_floors.setdefault(entity["area"], entity["floor"])
    return {
        "home_id": home_id,
        "size": len(entities),
        "sayso_entity_area": sayso_entity_area or default_area,
        "entities": entities,
        "active_timers": timers,
        "areas": list(area_floors),
        "area_floors": area_floors,
        # A real home's names come from Home Assistant, so there is no synthetic
        # owner pool to draw from.
        "owners": (),
        "exposure_source": exposure_source,
    }


def _remaining_seconds(state: dict[str, Any]) -> int:
    remaining = (state.get("attributes") or {}).get("remaining", "0:00:00")
    try:
        hours, minutes, seconds = (float(part) for part in str(remaining).split(":"))
    except ValueError:
        return 0
    return int(hours * 3600 + minutes * 60 + seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Where to write the home JSON")
    parser.add_argument("--home-id", default="real_home")
    parser.add_argument("--sayso-entity-area", default=None)
    parser.add_argument(
        "--domains",
        default=",".join(DEFAULT_DOMAINS),
        help="Comma-separated Home Assistant domains to keep",
    )
    parser.add_argument(
        "--keep-entity",
        action="append",
        default=[],
        metavar="ENTITY_ID",
        help="Keep this entity even when its domain is filtered out (repeatable)",
    )
    parser.add_argument(
        "--allow-unexposed",
        action="store_true",
        help="Export without the Assist exposure list when the websocket API is "
             "unreachable. The snapshot records exposure_source=domain_filter and "
             "is not a valid input for the home-specific training recipe.",
    )
    parser.add_argument(
        "--require-entity",
        action="append",
        default=[],
        metavar="ENTITY_ID",
        help="Fail unless this entity is present in the export (repeatable)",
    )
    args = parser.parse_args(argv)

    base_url, token = os.environ.get("HA_URL", ""), os.environ.get("HA_TOKEN", "")
    if not base_url or not token:
        print("HA_URL and HA_TOKEN are required", file=sys.stderr)
        return 2

    try:
        states, area_rows = fetch_raw(base_url, token)
    except (urllib.error.URLError, urllib.error.HTTPError) as err:
        print(f"Home Assistant request failed: {err}", file=sys.stderr)
        return 1

    exposed: frozenset[str] | None = None
    alias_map: dict[str, list[str]] = {}
    exposure_source = "domain_filter"
    try:
        registry = fetch_assist_registry(base_url, token)
        exposed = frozenset(registry["exposed"])
        alias_map = registry["aliases"]
        exposure_source = "assist_exposure"
    except Exception as err:  # noqa: BLE001 - any websocket failure is the same decision
        if not args.allow_unexposed:
            print(
                f"could not read the Assist exposure list: {err}\n"
                "Home Assistant is authoritative for exposure; rerun with "
                "--allow-unexposed only to produce a clearly-marked partial snapshot.",
                file=sys.stderr,
            )
            return 1
        print(f"warning: exporting without the Assist exposure list ({err})", file=sys.stderr)

    home = build_home(
        states,
        area_rows,
        home_id=args.home_id,
        sayso_entity_area=args.sayso_entity_area,
        domains=tuple(d.strip() for d in args.domains.split(",") if d.strip()),
        keep_entities=frozenset(args.keep_entity),
        exposed_entities=exposed,
        aliases=alias_map,
        exposure_source=exposure_source,
    )

    present = {entity["entity_id"] for entity in home["entities"]}
    missing = [entity_id for entity_id in args.require_entity if entity_id not in present]
    if missing:
        print(f"required entities missing from the export: {missing}", file=sys.stderr)
        return 1
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(home, handle, indent=2, ensure_ascii=False, sort_keys=False)
        handle.write("\n")

    counts: dict[str, int] = {}
    for entity in home["entities"]:
        counts[entity["capability"]] = counts.get(entity["capability"], 0) + 1
    print(f"{home['size']} entities -> {args.out}")
    for capability, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {capability:<15}{count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
