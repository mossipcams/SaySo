"""Process entrypoint for the SaySo aiohttp server."""

from __future__ import annotations

import os
import sys

from aiohttp import web

from sayso_server.app import MissingServerTokenError, create_aiohttp_app, load_server_token
from sayso_server.const import DEFAULT_HOST, DEFAULT_PORT, HOST_ENV_VAR, PORT_ENV_VAR


def main() -> None:
    try:
        token = load_server_token()
    except MissingServerTokenError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    host = os.environ.get(HOST_ENV_VAR, DEFAULT_HOST)
    port = int(os.environ.get(PORT_ENV_VAR, str(DEFAULT_PORT)))
    app = create_aiohttp_app(token)
    web.run_app(app, host=host, port=port)


if __name__ == "__main__":
    main()
