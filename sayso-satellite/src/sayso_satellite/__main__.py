"""Process entrypoint for the Mac text satellite client."""

from __future__ import annotations

import sys

from sayso_satellite.client import send_text


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
        print(json_dumps(body))
    if status >= 400:
        raise SystemExit(1)


def json_dumps(value: object) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
