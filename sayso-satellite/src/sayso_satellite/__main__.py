"""Process entrypoint for the Mac recorded-audio Assist satellite client."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from typing import Any

from sayso_satellite.assist import (
    DEFAULT_WEBSOCKET_URL,
    AssistError,
    TtsOutput,
    run_assist,
)
from sayso_satellite.capture import read_pcm16_file
from sayso_satellite.microphone import (
    DEFAULT_CHUNK_MS,
    MicInputError,
    capture_live_pcm,
    capture_wake_pcm,
    default_chunk_bytes,
    open_mac_microphone,
)
from sayso_satellite.loop import run_continuous_loop
from sayso_satellite.wake import (
    DEFAULT_WAKE_HITS,
    DEFAULT_WAKE_THRESHOLD,
    EnergyThresholdWakeEngine,
)
from sayso_satellite.playback import (
    AudioPlayer,
    PlaybackError,
    default_audio_player,
    ha_base_url_from_websocket,
    play_earcon,
    play_tts_response,
)
from sayso_satellite.response import (
    ResponsePlaybackMode,
    extract_assist_response_speech,
    extract_assist_speech,
    render_assist_response,
    resolve_playback_mode,
)

HA_TOKEN_ENV_VAR = "SAYSO_HA_TOKEN"
HA_WEBSOCKET_URL_ENV_VAR = "SAYSO_HA_WEBSOCKET_URL"
HA_DEVICE_ID_ENV_VAR = "SAYSO_HA_DEVICE_ID"
DEFAULT_CAPTURE_MS = 3000
DEFAULT_LISTEN_MS = 30_000
WAKE_THRESHOLD_ENV_VAR = "SAYSO_WAKE_THRESHOLD"
WAKE_HITS_ENV_VAR = "SAYSO_WAKE_HITS"
LISTEN_MS_ENV_VAR = "SAYSO_LISTEN_MS"


def resolve_device_id(
    cli_value: str | None,
    environ: Mapping[str, str],
) -> str | None:
    """Return the configured device ID, preferring CLI over environment."""

    if cli_value is not None:
        stripped = cli_value.strip()
        return stripped or None
    env_value = environ.get(HA_DEVICE_ID_ENV_VAR)
    if env_value is None:
        return None
    stripped = env_value.strip()
    return stripped or None


def resolve_wake_threshold(
    cli_value: float | None,
    environ: Mapping[str, str],
) -> float:
    """Return the configured wake RMS threshold."""

    if cli_value is not None:
        return cli_value
    env_value = environ.get(WAKE_THRESHOLD_ENV_VAR)
    if env_value is None:
        return DEFAULT_WAKE_THRESHOLD
    return float(env_value.strip())


def resolve_wake_hits(
    cli_value: int | None,
    environ: Mapping[str, str],
) -> int:
    """Return consecutive loud chunks required before wake detection."""

    if cli_value is not None:
        return cli_value
    env_value = environ.get(WAKE_HITS_ENV_VAR)
    if env_value is None:
        return DEFAULT_WAKE_HITS
    return int(env_value.strip())


def resolve_listen_ms(
    cli_value: int | None,
    environ: Mapping[str, str],
) -> int:
    """Return how long to wait for wake before giving up."""

    if cli_value is not None:
        return cli_value
    env_value = environ.get(LISTEN_MS_ENV_VAR)
    if env_value is None:
        return DEFAULT_LISTEN_MS
    return int(env_value.strip())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sayso_satellite")
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--audio-file",
        dest="audio_file",
        metavar="PATH",
        help="Send raw 16 kHz mono PCM16 from PATH (deterministic test/debug mode)",
    )
    input_group.add_argument(
        "--live",
        action="store_true",
        help="Capture one bounded utterance from the Mac microphone",
    )
    parser.add_argument(
        "--wake",
        action="store_true",
        help="Wait for wake detection before capturing (requires --live)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Continuously listen, wake, capture, and respond (requires --live --wake)",
    )
    parser.add_argument(
        "--wake-threshold",
        dest="wake_threshold",
        type=float,
        default=None,
        metavar="RMS",
        help=(
            "Wake RMS threshold for the energy wake engine "
            f"(default: {DEFAULT_WAKE_THRESHOLD}, or {WAKE_THRESHOLD_ENV_VAR})"
        ),
    )
    parser.add_argument(
        "--wake-hits",
        dest="wake_hits",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Consecutive loud chunks required before wake "
            f"(default: {DEFAULT_WAKE_HITS}, or {WAKE_HITS_ENV_VAR})"
        ),
    )
    parser.add_argument(
        "--listen-ms",
        dest="listen_ms",
        type=int,
        default=None,
        metavar="MS",
        help=(
            "How long to wait for wake before giving up "
            f"(default: {DEFAULT_LISTEN_MS}, or {LISTEN_MS_ENV_VAR})"
        ),
    )
    parser.add_argument(
        "--capture-ms",
        dest="capture_ms",
        type=int,
        default=DEFAULT_CAPTURE_MS,
        metavar="MS",
        help=f"Live capture duration in milliseconds (default: {DEFAULT_CAPTURE_MS})",
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
    parser.add_argument(
        "--device-id",
        dest="device_id",
        default=None,
        metavar="DEVICE_ID",
        help=f"Home Assistant device ID for this satellite (or {HA_DEVICE_ID_ENV_VAR})",
    )
    return parser


def _resolve_ha_token(args: argparse.Namespace) -> str:
    token = (args.ha_token or os.environ.get(HA_TOKEN_ENV_VAR, "")).strip()
    if not token:
        raise ValueError(
            f"Home Assistant access token is required via --ha-token or {HA_TOKEN_ENV_VAR}"
        )
    return token


def _resolve_websocket_url(args: argparse.Namespace) -> str:
    return (
        args.ha_websocket_url
        or os.environ.get(HA_WEBSOCKET_URL_ENV_VAR, DEFAULT_WEBSOCKET_URL)
    )


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv[1:] if argv is not None else None)

    if args.wake and not args.live:
        print("--wake requires --live", file=sys.stderr)
        raise SystemExit(1)
    if args.loop and (not args.live or not args.wake):
        print("--loop requires --live --wake", file=sys.stderr)
        raise SystemExit(1)
    if not args.audio_file and not args.live:
        parser.print_usage(file=sys.stderr)
        raise SystemExit(2)

    try:
        token = _resolve_ha_token(args)
        websocket_url = _resolve_websocket_url(args)
        device_id = resolve_device_id(args.device_id, os.environ)
        if args.live:
            if args.capture_ms <= 0:
                raise ValueError("--capture-ms must be positive")
            mic = open_mac_microphone(chunk_ms=DEFAULT_CHUNK_MS)
            chunk_bytes = default_chunk_bytes(chunk_ms=DEFAULT_CHUNK_MS)
            if args.loop:
                engine = EnergyThresholdWakeEngine(
                    threshold=resolve_wake_threshold(args.wake_threshold, os.environ),
                    required_hits=resolve_wake_hits(args.wake_hits, os.environ),
                )
                run_continuous_loop(
                    mic,
                    engine,
                    capture_ms=args.capture_ms,
                    listen_timeout_ms=resolve_listen_ms(args.listen_ms, os.environ),
                    chunk_bytes=chunk_bytes,
                    on_turn=lambda pcm: _run_live_turn(
                        pcm,
                        token=token,
                        websocket_url=websocket_url,
                        device_id=device_id,
                        timeout=args.timeout,
                    ),
                )
                return
            if args.wake:
                engine = EnergyThresholdWakeEngine(
                    threshold=resolve_wake_threshold(args.wake_threshold, os.environ),
                    required_hits=resolve_wake_hits(args.wake_hits, os.environ),
                )
                pcm = capture_wake_pcm(
                    mic,
                    engine,
                    capture_ms=args.capture_ms,
                    listen_timeout_ms=resolve_listen_ms(args.listen_ms, os.environ),
                    chunk_bytes=chunk_bytes,
                )
                if pcm is None:
                    return
            else:
                pcm = capture_live_pcm(
                    mic,
                    duration_ms=args.capture_ms,
                    chunk_bytes=chunk_bytes,
                )
        else:
            pcm = read_pcm16_file(args.audio_file)
        _run_live_turn(
            pcm,
            token=token,
            websocket_url=websocket_url,
            device_id=device_id,
            timeout=args.timeout,
        )
    except KeyboardInterrupt:
        raise SystemExit(0) from None
    except (AssistError, MicInputError, OSError, PlaybackError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


def _run_live_turn(
    pcm: bytes,
    *,
    token: str,
    websocket_url: str,
    device_id: str | None,
    timeout: float | None,
    player: AudioPlayer | None = None,
) -> None:
    """Run Assist and play the response for one captured utterance."""

    result = run_assist(
        pcm,
        token=token,
        websocket_url=websocket_url,
        device_id=device_id,
        timeout=timeout,
    )
    handle_assist_result(
        result,
        ha_token=token,
        websocket_url=websocket_url,
        player=player,
    )


def handle_assist_result(
    result: dict[str, Any],
    *,
    ha_token: str,
    websocket_url: str,
    player: AudioPlayer | None = None,
) -> None:
    """Render Assist text and play HA TTS or a local earcon."""

    speech = extract_assist_speech(result)
    print_assist_result(result, speech=speech)
    response_speech = extract_assist_response_speech(result)
    if (
        response_speech is not None
        and resolve_playback_mode(response_speech) is ResponsePlaybackMode.EARCON
    ):
        play_earcon(player or default_audio_player())
        return
    tts = result.get("tts")
    if not isinstance(tts, dict):
        return
    play_tts_response(
        _coerce_tts_output(tts),
        token=ha_token,
        base_url=ha_base_url_from_websocket(websocket_url),
        player=player or default_audio_player(),
    )


def _coerce_tts_output(tts: dict[str, Any]) -> TtsOutput:
    url = tts.get("url")
    token = tts.get("token")
    mime_type = tts.get("mime_type")
    media_id = tts.get("media_id")
    if not all(isinstance(value, str) and value.strip() for value in (url, token, mime_type, media_id)):
        raise PlaybackError("malformed TTS output")
    return {
        "url": url,
        "token": token,
        "mime_type": mime_type,
        "media_id": media_id,
    }


def print_assist_result(
    result: dict[str, Any],
    *,
    speech: str | None = None,
) -> None:
    """Render Home Assistant's completed Assist response."""

    render_assist_response(speech if speech is not None else extract_assist_speech(result))


if __name__ == "__main__":
    main()
