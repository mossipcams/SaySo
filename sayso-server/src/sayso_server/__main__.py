"""Process entrypoint for the SaySo aiohttp server."""

from __future__ import annotations

import os
import sys

from aiohttp import web

from sayso_server.app import MissingServerTokenError, create_aiohttp_app, load_server_token
from sayso_server.const import DEFAULT_HOST, DEFAULT_PORT, HOST_ENV_VAR, PORT_ENV_VAR
from sayso_server.mlx_runtime import build_mlx_runtime_for_server


def _preload_stt_runtime(app: web.Application) -> None:
    """Warm Whisper after LFM startup; failures must not affect ``model_ready``."""

    stt_runtime = app.get("stt_runtime")
    if stt_runtime is None:
        return
    readiness = app.get("readiness")
    try:
        stt_runtime.load()
    except Exception as exc:  # noqa: BLE001 — preload is best-effort
        print(f"Warning: STT preload failed: {exc}", file=sys.stderr)
        return
    if readiness is not None:
        readiness.set_stt_ready(True)


def main() -> None:
    try:
        token = load_server_token()
    except MissingServerTokenError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    host = os.environ.get(HOST_ENV_VAR, DEFAULT_HOST)
    port = int(os.environ.get(PORT_ENV_VAR, str(DEFAULT_PORT)))
    try:
        runtime = build_mlx_runtime_for_server()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    app = create_aiohttp_app(token, model_runtime=runtime)
    _preload_stt_runtime(app)
    web.run_app(app, host=host, port=port)


if __name__ == "__main__":
    main()
