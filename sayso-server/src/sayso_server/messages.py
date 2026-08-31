"""SaySo API v1 message types."""

from enum import StrEnum

from sayso_server.api import API_VERSION


class MessageType(StrEnum):
    HELLO = "hello"
    HELLO_ACK = "hello_ack"
    PING = "ping"
    PONG = "pong"
    ERROR = "error"
    GRAPH_SNAPSHOT = "graph_snapshot"
    STATE_DELTA = "state_delta"
    REGISTRY_DELTA = "registry_delta"


MESSAGE_TYPES_V1: frozenset[str] = frozenset(member.value for member in MessageType)


def supported_message_types(version: int) -> frozenset[str]:
    if version != API_VERSION:
        msg = f"unsupported SaySo API version: {version}"
        raise ValueError(msg)
    return MESSAGE_TYPES_V1
