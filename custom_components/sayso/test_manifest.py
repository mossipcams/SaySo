"""Validate the SaySo Home Assistant integration manifest without homeassistant."""

import json
from pathlib import Path

from custom_components.sayso.const import DOMAIN

MANIFEST_PATH = Path(__file__).with_name("manifest.json")
HACS_PATH = Path(__file__).resolve().parents[2] / "hacs.json"
REQUIRED_KEYS = {
    "domain",
    "name",
    "version",
    "documentation",
    "codeowners",
    "iot_class",
    "requirements",
    "config_flow",
    "integration_type",
}


def test_hacs_json_is_valid_with_name() -> None:
    hacs = json.loads(HACS_PATH.read_text(encoding="utf-8"))

    assert isinstance(hacs["name"], str)
    assert hacs["name"]
    assert "hacs" not in hacs
    assert "homeassistant" not in hacs


def test_manifest_is_valid_service_integration() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert REQUIRED_KEYS.issubset(manifest)
    assert manifest["domain"] == DOMAIN == "sayso"
    assert manifest["config_flow"] is True
    assert manifest["integration_type"] == "service"
    assert manifest["requirements"] == []
    assert manifest["iot_class"] in {
        "local_push",
        "local_polling",
        "cloud_push",
        "cloud_polling",
        "calculated",
    }
