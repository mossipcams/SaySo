"""Exposure, aliases and supported_features in the Home Assistant export.

Payloads here are clearly synthetic: this repository has no credentials for the
live instance, so the exporter is exercised against fixtures shaped like
``/api/states`` and the websocket registry. Refreshing the real snapshot stays a
prerequisite for the home-specific recipe (see generators.real_home).
"""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from scripts.fetch_ha_home import build_home
from scripts.ha_websocket import decode_frame, encode_frame

# MediaPlayerEntityFeature: PAUSE|VOLUME_SET|VOLUME_MUTE|PREVIOUS|NEXT|TURN_ON|
# TURN_OFF|VOLUME_STEP|PLAY|SEARCH_MEDIA
TV_FEATURES = 1 | 4 | 8 | 16 | 32 | 128 | 256 | 1024 | 16384 | 4194304
# An Echo Dot: playback and volume, no power control, no search.
SPEAKER_FEATURES = 1 | 4 | 8 | 16 | 32 | 1024 | 16384

STATES = [
    {
        "entity_id": "media_player.living_room_tv",
        "state": "playing",
        "attributes": {
            "friendly_name": "TV",
            "device_class": "tv",
            "supported_features": TV_FEATURES,
        },
    },
    {
        "entity_id": "media_player.kitchen_dot",
        "state": "idle",
        "attributes": {
            "friendly_name": "Kitchen Dot",
            "device_class": "speaker",
            "supported_features": SPEAKER_FEATURES,
        },
    },
    {
        "entity_id": "light.hidden_closet",
        "state": "off",
        "attributes": {"friendly_name": "Closet Light", "supported_color_modes": ["onoff"]},
    },
    {
        "entity_id": "fan.plain_extractor",
        "state": "off",
        "attributes": {"friendly_name": "Extractor", "supported_features": 0},
    },
    {
        "entity_id": "fan.bedroom_tower",
        "state": "on",
        "attributes": {"friendly_name": "Tower Fan", "supported_features": 1, "percentage": 40},
    },
]
AREA_ROWS = [
    ["media_player.living_room_tv", "Living Room", "Main Floor"],
    ["media_player.kitchen_dot", "Kitchen", "Main Floor"],
    ["light.hidden_closet", "Hallway", "Main Floor"],
    ["fan.plain_extractor", "Bathroom", "Main Floor"],
    ["fan.bedroom_tower", "Bedroom", "Upstairs"],
]
EXPOSED = frozenset(
    {
        "media_player.living_room_tv",
        "media_player.kitchen_dot",
        "fan.plain_extractor",
        "fan.bedroom_tower",
    }
)
ALIASES = {"media_player.living_room_tv": ["the telly", "big screen"]}


@pytest.fixture
def home():
    return build_home(
        STATES,
        AREA_ROWS,
        exposed_entities=EXPOSED,
        aliases=ALIASES,
        exposure_source="assist_exposure",
    )


def test_home_assistant_exposure_is_authoritative(home):
    """A controllable domain is not enough: Assist has to expose the entity."""
    assert "light.hidden_closet" not in {e["entity_id"] for e in home["entities"]}
    assert home["exposure_source"] == "assist_exposure"


def test_without_the_exposure_list_the_export_says_so():
    home = build_home(STATES, AREA_ROWS)
    assert home["exposure_source"] == "domain_filter"
    assert "light.hidden_closet" in {e["entity_id"] for e in home["entities"]}


def test_aliases_come_from_the_entity_registry(home):
    tv = next(e for e in home["entities"] if e["entity_id"] == "media_player.living_room_tv")
    assert tv["name"] == "TV"
    assert tv["aliases"] == ["TV", "the telly", "big screen"]


def test_null_and_blank_registry_aliases_are_discarded():
    home = build_home(
        STATES,
        AREA_ROWS,
        exposed_entities=EXPOSED,
        aliases={"media_player.living_room_tv": [None, "", "  ", "the telly"]},
        exposure_source="assist_exposure",
    )
    tv = next(e for e in home["entities"] if e["entity_id"] == "media_player.living_room_tv")
    assert tv["aliases"] == ["TV", "the telly"]


def test_media_player_features_come_from_supported_features(home):
    from generators.capability_registry import entity_supports

    tv = next(e for e in home["entities"] if e["entity_id"] == "media_player.living_room_tv")
    dot = next(e for e in home["entities"] if e["entity_id"] == "media_player.kitchen_dot")
    assert entity_supports(tv, "media_players", "turn_on")
    assert entity_supports(tv, "media_players", "search_and_play")
    # Not every media player has power, and none of them all have search.
    assert not entity_supports(dot, "media_players", "turn_on")
    assert not entity_supports(dot, "media_players", "search_and_play")
    assert entity_supports(dot, "media_players", "volume_set")
    assert entity_supports(dot, "media_players", "mute")


def test_fan_speed_is_not_assumed(home):
    from generators.capability_registry import entity_supports

    plain = next(e for e in home["entities"] if e["entity_id"] == "fan.plain_extractor")
    tower = next(e for e in home["entities"] if e["entity_id"] == "fan.bedroom_tower")
    assert not entity_supports(plain, "fans", "set_speed")
    assert entity_supports(tower, "fans", "set_speed")


def test_the_export_carries_the_area_map_build_scenario_needs(home):
    assert home["areas"] and home["area_floors"]
    assert home["area_floors"]["Bedroom"] == "Upstairs"

    # A capability the home lacks can be injected without a KeyError.
    from generators.scenarios import build_scenario

    scenario = build_scenario(
        index=1, seed=2, capability="locks", operation="lock", home_size=16,
        home=json.loads(json.dumps(home)),
    )
    assert scenario["target_entity"]["capability"] == "locks"


@pytest.mark.parametrize("payload", [b"", b"hi", b"x" * 200, b"y" * 70000])
def test_websocket_frames_round_trip(payload):
    """The codec is the only hand-rolled protocol here, so it gets a direct check."""
    fin, opcode, decoded = decode_frame(_unmask(encode_frame(payload)))
    assert fin and opcode == 0x1
    assert decoded == payload


def _unmask(frame: bytes):
    """Server frames are unmasked; re-encode the client frame as a server one."""
    import io
    import struct

    masked_bit = frame[1] & 0x80
    assert masked_bit, "client frames must be masked"
    length = frame[1] & 0x7F
    offset = 2
    if length == 126:
        length = struct.unpack("!H", frame[2:4])[0]
        offset = 4
    elif length == 127:
        length = struct.unpack("!Q", frame[2:10])[0]
        offset = 10
    mask = frame[offset:offset + 4]
    payload = bytes(b ^ mask[i % 4] for i, b in enumerate(frame[offset + 4:]))
    header = bytes([frame[0]])
    if length < 126:
        header += bytes([length])
    elif length < 1 << 16:
        header += bytes([126]) + struct.pack("!H", length)
    else:
        header += bytes([127]) + struct.pack("!Q", length)
    return io.BytesIO(header + payload)


def test_websocket_client_authenticates_and_returns_results():
    """Drive the client against a loopback server speaking Home Assistant's protocol."""
    import threading

    from scripts.ha_websocket import HomeAssistantWebSocket

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    seen: list[dict] = []

    def serve():
        conn, _ = listener.accept()
        stream = conn.makefile("rb")
        while stream.readline() not in (b"\r\n", b"\n", b""):
            pass
        conn.sendall(
            b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
            b"Connection: Upgrade\r\n\r\n"
        )
        conn.sendall(_server_frame({"type": "auth_required", "ha_version": "2026.8.3"}))
        seen.append(_read(stream))
        conn.sendall(_server_frame({"type": "auth_ok"}))
        request = _read(stream)
        seen.append(request)
        conn.sendall(
            _server_frame(
                {
                    "id": request["id"],
                    "type": "result",
                    "success": True,
                    "result": {"exposed_entities": {"light.a": {"conversation": True}}},
                }
            )
        )
        conn.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    with HomeAssistantWebSocket(f"http://127.0.0.1:{port}", "TOKEN", timeout=5) as client:
        result = client.command({"type": "homeassistant/expose_entity/list"})
    thread.join(timeout=5)
    listener.close()

    assert seen[0] == {"type": "auth", "access_token": "TOKEN"}
    assert seen[1]["type"] == "homeassistant/expose_entity/list"
    assert result == {"exposed_entities": {"light.a": {"conversation": True}}}


def _server_frame(message: dict) -> bytes:
    """Unmasked text frame, as a server sends."""
    import struct

    payload = json.dumps(message).encode()
    header = bytes([0x81])
    if len(payload) < 126:
        header += bytes([len(payload)])
    else:
        header += bytes([126]) + struct.pack("!H", len(payload))
    return header + payload


def _read(stream) -> dict:
    fin, _opcode, payload = decode_frame(stream)
    assert fin
    return json.loads(payload.decode())
