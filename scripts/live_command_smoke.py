
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

DEFAULT_BASE_URL = "http://192.168.1.35:8123"
DEFAULT_AGENT = "conversation.home_assistant"
_TRANSIENT_STATES = {"unavailable", "unknown", "none", ""}




@dataclass(frozen=True, slots=True)
class Command:

    tool: str
    utterance: str
    target_domains: tuple[str, ...] = ()
    requires: str | None = None
    needs_area: bool = False


MATRIX: tuple[Command, ...] = (
    Command("GetDateTime", "What time is it?"),
    Command(
        "GetLiveContext",
        "What is the state of {target}?",
        target_domains=("light", "switch", "fan", "media_player"),
        requires="light",
    ),
    Command(
        "HassTurnOn",
        "Turn on {target}",
        target_domains=("light", "switch", "fan", "media_player", "cover", "lock"),
        requires="light",
    ),
    Command(
        "HassTurnOff",
        "Turn off {target}",
        target_domains=("light", "switch", "fan", "media_player"),
        requires="light",
    ),
    Command(
        "HassLightSet",
        "Set {target} to 50 percent",
        target_domains=("light",),
        requires="light",
    ),
    Command(
        "HassFanSetSpeed",
        "Set {target} to 50 percent",
        target_domains=("fan",),
        requires="fan",
    ),
    Command(
        "HassClimateSetTemperature",
        "Set {target} to 70 degrees",
        target_domains=("climate",),
        requires="climate",
    ),
    Command(
        "HassSetVolume",
        "Set the volume to 30 percent",
        target_domains=("media_player",),
        requires="media_player",
    ),
    Command(
        "HassSetVolumeRelative",
        "Turn the volume up",
        target_domains=("media_player",),
        requires="media_player",
    ),
    Command(
        "HassMediaNext",
        "Next track",
        target_domains=("media_player",),
        requires="media_player",
    ),
    Command(
        "HassMediaPrevious",
        "Previous track",
        target_domains=("media_player",),
        requires="media_player",
    ),
    Command(
        "HassMediaPause",
        "Pause the music",
        target_domains=("media_player",),
        requires="media_player",
    ),
    Command(
        "HassMediaUnpause",
        "Resume the music",
        target_domains=("media_player",),
        requires="media_player",
    ),
    Command(
        "HassMediaPlayerMute",
        "Mute the TV",
        target_domains=("media_player",),
        requires="media_player",
    ),
    Command(
        "HassMediaPlayerUnmute",
        "Unmute the TV",
        target_domains=("media_player",),
        requires="media_player",
    ),
    Command(
        "HassMediaSearchAndPlay",
        "Play some jazz",
        target_domains=("media_player",),
        requires="media_player",
    ),
    Command(
        "HassVacuumStart",
        "Start the vacuum",
        target_domains=("vacuum",),
        requires="vacuum",
    ),
    Command(
        "HassVacuumReturnToBase",
        "Send the vacuum back to its base",
        target_domains=("vacuum",),
        requires="vacuum",
    ),
    Command(
        "HassVacuumCleanArea",
        "Vacuum the {area}",
        requires="vacuum",
        needs_area=True,
    ),
    Command("HassStartTimer", "Set a 5 minute timer"),
    Command("HassTimerStatus", "How much time is left on my timer?"),
    Command("HassPauseTimer", "Pause my timer"),
    Command("HassUnpauseTimer", "Resume my timer"),
    Command("HassIncreaseTimer", "Add 1 minute to my timer"),
    Command("HassDecreaseTimer", "Remove 1 minute from my timer"),
    Command("HassCancelTimer", "Cancel my timer"),
    Command("HassCancelAllTimers", "Cancel all my timers"),
)

ACTION_TOOLS = frozenset(
    name for name in (c.tool for c in MATRIX) if name.startswith("Hass")
)

REQUIRE_TOOL_TOOLS = ACTION_TOOLS | {"GetDateTime", "GetLiveContext"}

NO_TARGET_TOOLS = frozenset(
    {
        "HassStartTimer",
        "HassTimerStatus",
        "HassPauseTimer",
        "HassUnpauseTimer",
        "HassIncreaseTimer",
        "HassDecreaseTimer",
        "HassCancelTimer",
        "HassCancelAllTimers",
    }
)

_DIAGNOSTIC_TOKENS = (
    "apollo",
    "msr",
    "levds",
    "ld2410",
    "radar",
    "engineering",
    "calibration",
    "firmware",
    "child lock",
    "do not disturb",
    "announcement",
    "communication",
    "startup",
    "reporting",
    "bluetooth",
    "proxy",
    "pre-release",
    "battery",
    "motion detection",
    "person detection",
    "tamper",
    "cry detection",
    "permit join",
    "power-on",
    "display",
)


def _name_rank(name: str) -> tuple[int, int, str]:
    folded = name.casefold()
    suspect = int(any(token in folded for token in _DIAGNOSTIC_TOKENS))
    return (suspect, len(name), name.casefold())

_AREAS_TEMPLATE = (
    "{% set ns = namespace(rows=[]) %}"
    "{% for a in areas() %}{% set ns.rows = ns.rows + [area_name(a)] %}{% endfor %}"
    "{{ ns.rows | tojson }}"
)




class HomeAssistantClient:

    def __init__(self, base_url: str, token: str, *, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = token
        self.timeout = timeout

    def _call(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        body = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout
            ) as response:
                raw = response.read()
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", "replace")
            raise RuntimeError(f"{method} {path} -> HTTP {err.code}: {detail}") from err
        except urllib.error.URLError as err:
            raise RuntimeError(f"{method} {path} -> {err.reason}") from err
        if not raw:
            return None
        return json.loads(raw)

    def states(self) -> list[dict[str, Any]]:
        return self._call("GET", "/api/states")

    def areas(self) -> list[str]:
        result = self._call("POST", "/api/template", {"template": _AREAS_TEMPLATE})
        if isinstance(result, str):
            result = json.loads(result)
        return [str(name) for name in result] if isinstance(result, list) else []

    def converse(self, agent_id: str, text: str, *, language: str = "en") -> dict[str, Any]:
        return self._call(
            "POST",
            "/api/conversation/process",
            {"text": text, "language": language, "agent_id": agent_id},
        )

    def list_traces(self, *, limit: int = 1) -> list[dict[str, Any]]:
        result = self._call(
            "POST",
            "/api/services/sayso/list_traces?return_response",
            {"limit": limit},
        )
        response = unwrap_service_response(result)
        return list(response.get("traces", [])) if isinstance(response, dict) else []

    def get_trace(self, trace_id: str) -> dict[str, Any]:
        result = self._call(
            "POST",
            "/api/services/sayso/get_trace?return_response",
            {"trace_id": trace_id},
        )
        response = unwrap_service_response(result)
        return response if isinstance(response, dict) else {"summary": None, "events": []}




def unwrap_service_response(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    inner = result.get("service_response")
    return inner if isinstance(inner, dict) else result


def discover_agent(states: list[dict[str, Any]]) -> str:
    agents = [
        state["entity_id"]
        for state in states
        if str(state.get("entity_id", "")).startswith("conversation.")
        and state["entity_id"] != DEFAULT_AGENT
    ]
    if len(agents) == 1:
        return agents[0]
    if not agents:
        raise RuntimeError(
            "No non-default conversation agent found. Set SAYSO_AGENT explicitly."
        )
    raise RuntimeError(
        "Multiple conversation agents found; set SAYSO_AGENT explicitly: "
        + ", ".join(sorted(agents))
    )


def targets_by_domain(states: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for state in states:
        entity_id = str(state.get("entity_id", ""))
        if "." not in entity_id or entity_id.startswith("conversation."):
            continue
        if str(state.get("state", "")).casefold() in _TRANSIENT_STATES:
            continue
        attributes = state.get("attributes") or {}
        name = attributes.get("friendly_name") or entity_id.split(".", 1)[1]
        grouped.setdefault(entity_id.split(".", 1)[0], []).append(str(name))
    for names in grouped.values():
        names.sort(key=_name_rank)
    return grouped


def build_utterance(
    command: Command,
    targets: dict[str, list[str]],
    areas: list[str],
) -> str | None:
    if command.requires is not None and command.requires not in targets:
        return None
    replacements: dict[str, str] = {}
    if command.target_domains:
        name = next(
            (
                targets[domain][0]
                for domain in command.target_domains
                if targets.get(domain)
            ),
            None,
        )
        if name is None:
            return None
        replacements["target"] = name
    if command.needs_area:
        if not areas:
            return None
        replacements["area"] = areas[0]
    return command.utterance.format(**replacements)


def speech_from_response(payload: dict[str, Any] | None) -> str:
    if not isinstance(payload, dict):
        return ""
    response = payload.get("response")
    if not isinstance(response, dict):
        return ""
    speech = response.get("speech")
    if not isinstance(speech, dict):
        return ""
    plain = speech.get("plain")
    if isinstance(plain, dict):
        return str(plain.get("speech") or "")
    return ""


def base_tool_name(tool: str | None) -> str:
    if not tool:
        return ""
    return tool.rsplit("__", 1)[-1]


def classify(
    command: Command,
    *,
    response: dict[str, Any] | None,
    trace: dict[str, Any] | None,
    lenient: bool = False,
) -> tuple[str, str]:
    spoken = speech_from_response(response)
    response_type = ""
    if isinstance(response, dict) and isinstance(response.get("response"), dict):
        response_type = str(response["response"].get("response_type") or "")

    summary = (trace or {}).get("summary") or {}
    actual_raw = summary.get("tool")
    actual_tool = base_tool_name(actual_raw) or None
    target = summary.get("target")
    expected = command.tool

    detail = f"tool={actual_raw or 'none'}"
    if target:
        detail += f" target={target}"
    if actual_tool and actual_tool != expected:
        detail += f" (expected {expected})"

    if response_type == "error":
        return "FAIL", f"spoken error {spoken!r} | {detail}"
    if not spoken:
        return "FAIL", f"empty speech (response_type={response_type or 'missing'}) | {detail}"
    if summary.get("success") is False:
        stage = summary.get("error_stage") or "?"
        kind = summary.get("error_type") or "?"
        message = summary.get("error_message")
        suffix = f" ({message})" if message else ""
        return "FAIL", f"trace failure {stage}/{kind}{suffix} | {detail}"
    if expected in REQUIRE_TOOL_TOOLS and not actual_raw:
        return "FAIL", "no tool call recorded in trace"
    if not lenient and actual_tool != expected:
        return "FAIL", detail
    if (
        expected in ACTION_TOOLS
        and expected not in NO_TARGET_TOOLS
        and not target
    ):
        return "FAIL", f"tool={actual_raw} called but Home Assistant resolved no entity"
    return "PASS", detail




@dataclass
class RunResult:
    tool: str
    utterance: str
    status: str
    detail: str
    spoken: str = ""
    trace_id: str | None = None
    trace: dict[str, Any] = field(default_factory=dict)


def wait_for_trace(
    client: HomeAssistantClient,
    before_id: str | None,
    *,
    timeout: float = 8.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        traces = client.list_traces(limit=3)
        fresh = [t for t in traces if t.get("trace_id") and t.get("trace_id") != before_id]
        if fresh:
            trace_id = str(fresh[0]["trace_id"])
            return client.get_trace(trace_id)
        time.sleep(0.2)
    return {"summary": None, "events": []}


def run_command(
    client: HomeAssistantClient,
    agent: str,
    command: Command,
    utterance: str,
    *,
    strict: bool,
) -> RunResult:
    before = client.list_traces(limit=1)
    before_id = str(before[0]["trace_id"]) if before and before[0].get("trace_id") else None
    try:
        response = client.converse(agent, utterance)
    except RuntimeError as err:
        return RunResult(command.tool, utterance, "FAIL", str(err))
    trace = wait_for_trace(client, before_id)
    status, detail = classify(command, response=response, trace=trace, lenient=not strict)
    summary = trace.get("summary") or {}
    return RunResult(
        tool=command.tool,
        utterance=utterance,
        status=status,
        detail=detail,
        spoken=speech_from_response(response),
        trace_id=summary.get("trace_id"),
        trace=trace,
    )


def run_matrix(
    client: HomeAssistantClient,
    agent: str,
    commands: tuple[Command, ...] = MATRIX,
    *,
    strict: bool,
) -> list[RunResult]:
    states = client.states()
    targets = targets_by_domain(states)
    areas = client.areas()
    results: list[RunResult] = []
    for command in commands:
        utterance = build_utterance(command, targets, areas)
        if utterance is None:
            reason = (
                f"no {command.requires or 'target'} in home"
                if command.requires or command.target_domains or command.needs_area
                else "missing required entity/area"
            )
            results.append(
                RunResult(command.tool, command.utterance, "SKIP", f"skipped: {reason}")
            )
            continue
        result = run_command(client, agent, command, utterance, strict=strict)
        results.append(result)
        print(
            f"  {result.status:<5} {command.tool:<28} {utterance!r} -> {result.detail}",
            flush=True,
        )
    return results


def print_summary(results: list[RunResult]) -> int:
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    print("\n=== summary ===")
    print(f"  PASS {counts['PASS']}  FAIL {counts['FAIL']}  SKIP {counts['SKIP']}")
    for result in results:
        if result.status == "FAIL":
            print(f"  FAIL {result.tool}: {result.detail} | said: {result.spoken!r}")
    return 1 if counts["FAIL"] else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command matrix and exit without touching the network.",
    )
    parser.add_argument("--agent", default=os.environ.get("SAYSO_AGENT", ""))
    parser.add_argument("--url", default=os.environ.get("HA_URL", DEFAULT_BASE_URL))
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="Accept any successful tool instead of requiring the expected one.",
    )
    parser.add_argument("--json-out", help="Write full results (responses + traces) here.")
    parser.add_argument(
        "--token-file",
        default=os.environ.get("HA_TOKEN_FILE", ""),
        help="Read the HA token from this file instead of HA_TOKEN (never printed).",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated tool names (or prefixes) to run instead of the full matrix.",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        print(f"{len(MATRIX)} command families:")
        for command in MATRIX:
            gate = command.requires or "-"
            print(f"  {command.tool:<28} requires={gate:<13} {command.utterance!r}")
        return 0

    token = os.environ.get("HA_TOKEN", "")
    if not token and args.token_file:
        try:
            with open(args.token_file, encoding="utf-8") as handle:
                token = handle.read().strip()
        except OSError as err:
            print(f"could not read --token-file: {err}", file=sys.stderr)
            return 2
    if not token:
        print(
            "HA_TOKEN is required (export it or pass --token-file; never printed)",
            file=sys.stderr,
        )
        return 2

    client = HomeAssistantClient(args.url, token)
    commands = MATRIX
    if args.only:
        wanted = tuple(part.strip().casefold() for part in args.only.split(",") if part.strip())
        commands = tuple(
            command
            for command in MATRIX
            if any(command.tool.casefold().startswith(prefix) for prefix in wanted)
        )
        if not commands:
            print(f"--only matched no tools: {args.only}", file=sys.stderr)
            return 2
    try:
        agent = args.agent or discover_agent(client.states())
        print(f"Home Assistant: {args.url}")
        print(f"SaySo agent:    {agent}\n")
        results = run_matrix(client, agent, commands, strict=not args.lenient)
    except RuntimeError as err:
        print(f"setup failed: {err}", file=sys.stderr)
        return 2

    if args.json_out:
        payload = [
            {
                "tool": result.tool,
                "utterance": result.utterance,
                "status": result.status,
                "detail": result.detail,
                "spoken": result.spoken,
                "trace_id": result.trace_id,
                "trace": result.trace,
            }
            for result in results
        ]
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=False)
            handle.write("\n")
        print(f"\nwrote {args.json_out}")

    return print_summary(results)


if __name__ == "__main__":
    raise SystemExit(main())
