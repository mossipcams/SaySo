"""Generate the locked SaySo tool-schema reference artifact.

Accepts a JSON payload from the controlled reference setup and writes one
canonical artifact. Stdlib only; fingerprint logic mirrors schema.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_VERSION = "sayso-tool-schema/v1"
LOCKED_ARTIFACT_NAME = "sayso-tool-schema-v1.json"
LOCKED_ARTIFACT_DIR = "schemas"
REQUIRED_PAYLOAD_KEYS = (
    "home_assistant_version",
    "sayso_source_commit",
    "home_assistant_llm_api_identifier",
    "tools",
)


def canonicalize_schema(node: Any) -> Any:
    """Recursively sort mapping keys and required arrays for stable serialization."""
    if isinstance(node, list):
        return [canonicalize_schema(item) for item in node]

    if not isinstance(node, dict):
        return node

    canonical: dict[str, Any] = {}
    for key in sorted(node):
        value = node[key]
        if key == "required" and isinstance(value, list):
            canonical[key] = sorted(value)
        else:
            canonical[key] = canonicalize_schema(value)
    return canonical


def canonicalize_compiled_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Canonicalize compiled tools and sort them by function name."""
    canonical_tools = [canonicalize_schema(tool) for tool in tools]
    return sorted(canonical_tools, key=lambda tool: tool["function"]["name"])


def emit_canonical_json(tools: list[dict[str, Any]]) -> bytes:
    """Emit byte-identical canonical JSON for compiled tools."""
    canonical_tools = canonicalize_compiled_tools(tools)
    return json.dumps(
        canonical_tools,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def schema_fingerprint(tools: list[dict[str, Any]]) -> str:
    """Return the SHA-256 fingerprint of the canonical compiled-tool JSON."""
    digest = hashlib.sha256(emit_canonical_json(tools)).hexdigest()
    return f"sha256:{digest}"


def is_locked_artifact_path(path: Path) -> bool:
    """Return True when ``path`` is the immutable v1 locked artifact location."""
    return path.name == LOCKED_ARTIFACT_NAME and path.parent.name == LOCKED_ARTIFACT_DIR


def build_artifact(payload: dict[str, Any]) -> bytes:
    """Build canonical artifact bytes from a controlled reference payload."""
    missing = [key for key in REQUIRED_PAYLOAD_KEYS if key not in payload]
    if missing:
        raise ValueError(f"Payload missing required keys: {', '.join(missing)}")

    tools = payload["tools"]
    if not isinstance(tools, list):
        raise ValueError("Payload tools must be a JSON array")

    canonical_tools = canonicalize_compiled_tools(tools)
    artifact = {
        "contract_version": CONTRACT_VERSION,
        "home_assistant_version": payload["home_assistant_version"],
        "sayso_source_commit": payload["sayso_source_commit"],
        "home_assistant_llm_api_identifier": payload["home_assistant_llm_api_identifier"],
        "schema_fingerprint": schema_fingerprint(tools),
        "tools": canonical_tools,
    }
    return json.dumps(
        artifact,
        separators=(",", ":"),
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")


def write_artifact(output: Path, artifact_bytes: bytes) -> None:
    """Write artifact bytes, refusing to overwrite a locked artifact."""
    if output.exists() and is_locked_artifact_path(output):
        raise SystemExit(
            f"Refusing to overwrite locked artifact at {output}. "
            "Create a new contract version and file instead."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(artifact_bytes)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSON payload from the controlled reference setup",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Artifact path to write",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        print(f"Invalid payload JSON: {err}", file=sys.stderr)
        return 1

    if not isinstance(payload, dict):
        print("Payload must be a JSON object", file=sys.stderr)
        return 1

    try:
        artifact_bytes = build_artifact(payload)
    except ValueError as err:
        print(str(err), file=sys.stderr)
        return 1

    try:
        write_artifact(args.output, artifact_bytes)
    except SystemExit as err:
        print(str(err), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
