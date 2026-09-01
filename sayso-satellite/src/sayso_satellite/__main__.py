"""Process entrypoint for the Mac recorded-audio Assist satellite client."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from sayso_satellite.assist import (
    DEFAULT_WEBSOCKET_URL,
    AssistError,
    run_assist,
)
from sayso_satellite.capture import read_pcm16_file
from sayso_satellite.response import render_assist_response

HA_TOKEN_ENV_VAR = "SAYSO_HA_TOKEN"
HA_WEBSOCKET_URL_ENV_VAR = "SAYSO_HA_WEBSOCKET_URL"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sayso_satellite")
    parser.add_argument(
        "--audio-file",
        dest="audio_file",
        metavar="PATH",
        help="Send raw 16 kHz mono PCM16 from PATH",
    )
    parser.add_argument(
        "--timeout",
        dest="timeout",
        type=float,
        metavar="SECONDS",
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--ha-websocket-url",
        default=None,
        metavar="URL",
        help=f"Home Assistant WebSocket URL (default: {DEFAULT_WEBSOCKET_URL})",
    )
    parser.add_argument(
        "--ha-token",
        default=None,
        metavar="TOKEN",
        help=f"Home Assistant access token (or {HA_TOKEN_ENV_VAR})",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv[1:] if argv is not None else None)

    if not args.audio_file:
        parser.print_usage(file=sys.stderr)
        raise SystemExit(2)

    try:
        pcm = read_pcm16_file(args.audio_file)
        token = (args.ha_token or os.environ.get(HA_TOKEN_ENV_VAR, "")).strip()
        if not token:
            raise ValueError(
                f"Home Assistant access token is required via --ha-token or {HA_TOKEN_ENV_VAR}"
            )
        result = run_assist(
            pcm,
            token=token,
            websocket_url=(
                args.ha_websocket_url
                or os.environ.get(HA_WEBSOCKET_URL_ENV_VAR, DEFAULT_WEBSOCKET_URL)
            ),
            timeout=args.timeout,
        )
        print_assist_result(result)
    except (AssistError, OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


def print_assist_result(result: dict[str, Any]) -> None:
    """Render Home Assistant's completed Assist response."""

    content: str | None = None
    intent = result.get("intent")
    if isinstance(intent, dict):
        response = intent.get("response")
        if isinstance(response, str):
            content = response
        elif isinstance(response, dict):
            speech = response.get("speech")
            plain = speech.get("plain") if isinstance(speech, dict) else None
            if isinstance(plain, dict) and isinstance(plain.get("speech"), str):
                content = plain["speech"]
    if content is None and isinstance(result.get("text"), str):
        content = result["text"]
    render_assist_response(content)


if __name__ == "__main__":
    main()
