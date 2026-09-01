"""Process entrypoint for the Mac text and recorded-audio satellite client.

Recorded audio uses the same HTTP timeout defaults as text (180s). Override
with ``--timeout SECONDS`` or the ``SAYSO_TIMEOUT_SECONDS`` environment
variable (see ``sayso_satellite.client``).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import Any

from sayso_satellite.capture import read_pcm16_file
from sayso_satellite.client import send_audio, send_text
from sayso_satellite.response import render_text_response_payload


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
        help="HTTP request timeout (default: 180; override via SAYSO_TIMEOUT_SECONDS)",
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Text to send when not using --audio-file",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv[1:] if argv is not None else None)

    if args.audio_file:
        try:
            pcm = read_pcm16_file(args.audio_file)
            status, body = send_audio(pcm, timeout=args.timeout)
        except (OSError, ValueError, RuntimeError) as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from exc
    else:
        if not args.text:
            parser.print_usage(file=sys.stderr)
            raise SystemExit(2)
        text = " ".join(args.text)
        try:
            status, body = send_text(text, timeout=args.timeout)
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
