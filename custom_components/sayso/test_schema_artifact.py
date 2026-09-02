"""Focused checks for the locked sayso-tool-schema reference artifact generator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from custom_components.sayso.schema import schema_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "scripts" / "generate_sayso_tool_schema.py"
LOCKED_ARTIFACT = REPO_ROOT / "schemas" / "sayso-tool-schema-v1.json"
CONTRACT_VERSION = "sayso-tool-schema/v1"


def _sample_tool(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Control {name}.",
            "parameters": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                },
            },
        },
    }


def _sample_payload(*, commit: str = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef") -> dict[str, Any]:
    tools = [_sample_tool("AlphaTool"), _sample_tool("BetaTool")]
    return {
        "home_assistant_version": "2026.8.3",
        "sayso_source_commit": commit,
        "home_assistant_llm_api_identifier": "assist",
        "tools": tools,
    }


def _run_generator(*, payload_path: Path, output_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--input",
            str(payload_path),
            "--output",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_generator_produces_byte_identical_output_from_same_payload(
    tmp_path: Path,
) -> None:
    """Generating twice from the same payload yields byte-identical artifacts."""
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(_sample_payload()), encoding="utf-8")

    first_output = tmp_path / "artifact-a.json"
    second_output = tmp_path / "artifact-b.json"

    first = _run_generator(payload_path=payload_path, output_path=first_output)
    second = _run_generator(payload_path=payload_path, output_path=second_output)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert first_output.read_bytes() == second_output.read_bytes()


def test_embedded_fingerprint_matches_production_schema_fingerprint(
    tmp_path: Path,
) -> None:
    """The artifact fingerprint matches production schema_fingerprint(tools)."""
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(_sample_payload()), encoding="utf-8")
    output_path = tmp_path / "artifact.json"

    completed = _run_generator(payload_path=payload_path, output_path=output_path)
    assert completed.returncode == 0, completed.stderr

    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["contract_version"] == CONTRACT_VERSION
    assert artifact["schema_fingerprint"] == schema_fingerprint(artifact["tools"])


def test_generator_refuses_to_overwrite_locked_artifact(tmp_path: Path) -> None:
    """The locked v1 artifact path cannot be overwritten once it exists."""
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(_sample_payload()), encoding="utf-8")

    locked_path = tmp_path / "schemas" / "sayso-tool-schema-v1.json"
    locked_path.parent.mkdir(parents=True)
    locked_path.write_text('{"contract_version":"existing"}\n', encoding="utf-8")

    completed = _run_generator(payload_path=payload_path, output_path=locked_path)

    assert completed.returncode != 0
    assert "refusing" in completed.stderr.lower() or "refusing" in completed.stdout.lower()
    assert locked_path.read_text(encoding="utf-8") == '{"contract_version":"existing"}\n'


def test_locked_artifact_fingerprint_matches_production() -> None:
    """The checked-in lock artifact fingerprint matches production schema_fingerprint."""
    artifact = json.loads(LOCKED_ARTIFACT.read_text(encoding="utf-8"))
    assert artifact["schema_fingerprint"] == schema_fingerprint(artifact["tools"])


def test_locked_artifact_regenerates_byte_identically(tmp_path: Path) -> None:
    """Regenerating from the locked artifact fields yields byte-identical output."""
    artifact = json.loads(LOCKED_ARTIFACT.read_text(encoding="utf-8"))
    payload = {
        "home_assistant_version": artifact["home_assistant_version"],
        "sayso_source_commit": artifact["sayso_source_commit"],
        "home_assistant_llm_api_identifier": artifact["home_assistant_llm_api_identifier"],
        "tools": artifact["tools"],
    }
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    output_path = tmp_path / "regenerated.json"

    completed = _run_generator(payload_path=payload_path, output_path=output_path)

    assert completed.returncode == 0, completed.stderr
    assert output_path.read_bytes() == LOCKED_ARTIFACT.read_bytes()
