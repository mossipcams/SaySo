"""Served-model evaluation against an OpenAI-compatible chat endpoint."""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from evals.outcomes import PRODUCTION_MAX_OUTPUT_TOKENS, PRODUCTION_TEMPERATURE


class EndpointAdapter:
    """One HTTP chat completion. The request shape is what production sends."""

    name = "endpoint"

    def __init__(
        self,
        server: str,
        *,
        model: str = "sayso",
        timeout: float = 120.0,
        api_key: str = "",
        max_tokens: int = PRODUCTION_MAX_OUTPUT_TOKENS,
    ) -> None:
        if not server:
            raise ValueError("endpoint adapter requires an explicit --server")
        self.server = server
        self.model = model
        self.timeout = timeout
        self.api_key = api_key
        self.max_tokens = max_tokens

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": PRODUCTION_TEMPERATURE,
            "max_tokens": self.max_tokens,
            "tools": tools,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.server.rstrip('/')}/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
            headers=headers,
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())
