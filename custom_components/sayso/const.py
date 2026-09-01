"""Constants for the SaySo Home Assistant integration."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry

DOMAIN = "sayso"

DEVICE_NAME = "SaySo Voice Assistant"

CONF_URL = "url"
CONF_TOKEN = "token"

WS_PATH = "/api/v1/ws"

RECONNECT_INITIAL_DELAY = 1.0
RECONNECT_MAX_DELAY = 30.0
RECONNECT_BACKOFF_FACTOR = 2.0
HEARTBEAT_INTERVAL = 30.0

API_VERSION = 1

MSG_GRAPH_SNAPSHOT = "graph_snapshot"
MSG_STATE_DELTA = "state_delta"
MSG_REGISTRY_DELTA = "registry_delta"
MSG_ACTION_REQUEST = "action_request"
MSG_ACTION_RESULT = "action_result"

REJECT_ENTITY_NOT_EXPOSED = "entity_not_exposed"
REJECT_DOMAIN_MISMATCH = "domain_mismatch"
REJECT_DOMAIN_NOT_ALLOWED = "domain_not_allowed"
REJECT_ACTION_NOT_ALLOWED = "action_not_allowed"
REJECT_CAPABILITY_NOT_SUPPORTED = "capability_not_supported"

REASON_STATE_CHANGED = "state_changed"
REASON_STATE_UNCHANGED = "state_unchanged"
REASON_STATE_VERIFICATION_TIMEOUT = "state_verification_timeout"

STATE_VERIFICATION_TIMEOUT = 5.0

BRIGHTNESS_MIN = 1
BRIGHTNESS_MAX = 100

ACTION_PAYLOAD_BRIGHTNESS = "brightness"
ACTION_PAYLOAD_TEMPERATURE = "temperature"

POWER_DOMAINS = frozenset(
    {
        "light",
        "switch",
        "fan",
        "cover",
        "lock",
        "vacuum",
        "media_player",
    },
)
QUERY_ONLY_DOMAINS = frozenset({"sensor", "binary_sensor"})

CLIMATE_QUERY_ATTRIBUTES = (
    "current_temperature",
    "hvac_mode",
    "temperature",
)
SENSOR_QUERY_ATTRIBUTES = ("unit_of_measurement", "device_class", "state_class")

CONF_DOMAIN_ALLOWLIST = "domain_allowlist"
CONF_ACTION_ALLOWLIST = "action_allowlist"
CONF_EXPOSURE_MODE = "exposure_mode"
CONF_AREA_IDS = "area_ids"
CONF_ENTITY_IDS = "entity_ids"

EXPOSURE_MODE_ALL = "all"
EXPOSURE_MODE_AREA = "area"
EXPOSURE_MODE_ENTITY = "entity"

DOMAIN_OPTIONS = (
    "light",
    "switch",
    "climate",
    "cover",
    "fan",
    "media_player",
    "scene",
    "script",
    "lock",
    "vacuum",
    "binary_sensor",
    "sensor",
)

ACTION_OPTIONS = (
    "on",
    "off",
    "toggle",
    "set_brightness",
    "set_temperature",
    "query",
    "scene",
    "script",
)

DEFAULT_OPTIONS: dict[str, list[str] | str] = {
    CONF_DOMAIN_ALLOWLIST: [],
    CONF_ACTION_ALLOWLIST: [],
    CONF_EXPOSURE_MODE: EXPOSURE_MODE_ALL,
    CONF_AREA_IDS: [],
    CONF_ENTITY_IDS: [],
}


def normalize_options(raw: dict) -> dict[str, list[str] | str]:
    """Normalize submitted or stored options to the canonical shape."""

    mode = raw.get(CONF_EXPOSURE_MODE, EXPOSURE_MODE_ALL)
    if mode not in {EXPOSURE_MODE_ALL, EXPOSURE_MODE_AREA, EXPOSURE_MODE_ENTITY}:
        mode = EXPOSURE_MODE_ALL

    return {
        CONF_DOMAIN_ALLOWLIST: list(raw.get(CONF_DOMAIN_ALLOWLIST, [])),
        CONF_ACTION_ALLOWLIST: list(raw.get(CONF_ACTION_ALLOWLIST, [])),
        CONF_EXPOSURE_MODE: mode,
        CONF_AREA_IDS: (
            list(raw.get(CONF_AREA_IDS, []))
            if mode == EXPOSURE_MODE_AREA
            else []
        ),
        CONF_ENTITY_IDS: (
            list(raw.get(CONF_ENTITY_IDS, []))
            if mode == EXPOSURE_MODE_ENTITY
            else []
        ),
    }


def get_entry_options(entry: ConfigEntry) -> dict[str, list[str] | str]:
    """Return resolved options for a config entry, applying MVP defaults."""

    merged = {**DEFAULT_OPTIONS, **entry.options}
    return normalize_options(merged)
