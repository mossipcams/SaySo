"""Process entrypoint for the Mac text satellite client."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

from sayso_satellite.client import send_text
from sayso_satellite.response import render_text_response_payload


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m sayso_satellite <text>", file=sys.stderr)
        raise SystemExit(2)

    text = " ".join(sys.argv[1:])
    try:
        status, body = send_text(text)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    if body is not None:
        print_response_body(body)
    if status >= 400:
        raise SystemExit(1)


def print_response_body(
    body: dict[str, Any],
    *,
    sink: Callable[[str], None] | None = None,
) -> None:
    """Print a text_response via response policy, otherwise dump JSON."""

    if body.get("type") == "text_response" and isinstance(body.get("payload"), dict):
        render_text_response_payload(body["payload"], sink=sink)
        return
    writer = sink or print
    writer(json_dumps(body))


def json_dumps(value: object) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
