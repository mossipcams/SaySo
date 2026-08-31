"""Versioned SaySo API v1 message envelope."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from sayso_server.api import API_VERSION
from sayso_server.messages import MessageType


class SaySoEnvelope(BaseModel):
    version: Literal[API_VERSION]
    type: MessageType
    correlation_id: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def json_schema(cls) -> dict[str, object]:
        schema = cls.model_json_schema()
        schema["title"] = "SaySoEnvelope"
        return schema
