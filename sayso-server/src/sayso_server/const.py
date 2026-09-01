"""SaySo server wire constants."""

TEXT_PATH = "/api/v1/text"
AUDIO_PATH = "/api/v1/audio"
WS_PATH = "/api/v1/ws"
READINESS_PATH = "/api/v1/ready"

TOKEN_ENV_VAR = "SAYSO_TOKEN"
HOST_ENV_VAR = "SAYSO_HOST"
PORT_ENV_VAR = "SAYSO_PORT"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

DEFAULT_SATELLITE_ID = "macbook"
DEFAULT_SATELLITE_AREA_ID = "area_living_room"
SATELLITE_AREA_ID_ENV_VAR = "SAYSO_SATELLITE_AREA_ID"

# aiohttp WebSocketResponse defaults to 4 MiB; HA graph snapshots can exceed that.
HA_WS_MAX_MSG_SIZE = 64 * 1024 * 1024
