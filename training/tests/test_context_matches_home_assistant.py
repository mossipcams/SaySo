"""Guard the training context against drifting from what Home Assistant sends.

The model is trained to read this prompt, so a silent divergence between these
constants and the integration is a training bug that no eval would catch: the
eval sets render through the same generator.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from generators.context import (
    SAYSO_SYSTEM_PROMPT,
    exposed_entities,
    serialize_context,
)

REPO = Path(__file__).resolve().parents[2]

HOME = {
    "sayso_entity_area": "Kitchen",
    "entities": [
        {
            "name": "Kitchen Light",
            "aliases": ["Counter Light"],
            "domain": "light",
            "device_class": None,
            "area": "Kitchen",
            "floor": "Ground",
            "state": "off",
            "capabilities": ["brightness"],
        },
        {
            "name": "Front Door Lock",
            "aliases": [],
            "domain": "lock",
            "device_class": "door",
            "area": "Hallway",
            "floor": "Ground",
            "state": "locked",
            "capabilities": [],
        },
    ],
}


def test_sayso_prompt_matches_the_integration_default() -> None:
    """SAYSO_SYSTEM_PROMPT must stay identical to const.py DEFAULT_SYSTEM_PROMPT."""
    const = (REPO / "custom_components" / "sayso" / "const.py").read_text(encoding="utf-8")
    match = re.search(r'DEFAULT_SYSTEM_PROMPT = """(.*?)"""', const, re.DOTALL)
    assert match, "DEFAULT_SYSTEM_PROMPT not found in const.py"
    assert SAYSO_SYSTEM_PROMPT == match.group(1)


def test_entity_names_are_deduplicated() -> None:
    """Generators default aliases to [name]; repeating it teaches names come in pairs."""
    home = {
        "sayso_entity_area": "Kitchen",
        "entities": [
            {
                "name": "Great Room Thermostat",
                "aliases": ["Great Room Thermostat"],
                "domain": "climate",
                "area": "Great Room",
                "floor": "Ground",
                "state": "heat",
            }
        ],
    }
    assert exposed_entities(home)[0]["names"] == "Great Room Thermostat"


def test_overview_carries_no_state_and_uses_home_assistant_fields() -> None:
    """HA builds the overview with include_state=False and three keys per entity."""
    rows = exposed_entities(HOME)
    assert [row["names"] for row in rows] == ["Front Door Lock", "Kitchen Light, Counter Light"]
    for row in rows:
        assert set(row) <= {"names", "domain", "areas"}
        assert "state" not in row
        assert "capabilities" not in row


def test_serialized_context_is_yaml_and_omits_live_state() -> None:
    prompt = serialize_context(HOME)
    assert "Static Context: An overview of the areas and the devices in this smart home:" in prompt
    assert "You are in area Kitchen (floor Ground)" in prompt
    assert "GetLiveContext" in prompt

    block = prompt.split("smart home:\n", 1)[1].split("\nWhen controlling Home Assistant", 1)[0]
    # the model must not be able to read state straight out of the entity overview
    assert "locked" not in block and "off" not in block
    parsed = yaml.safe_load(block)
    assert parsed == [
        {"names": "Front Door Lock", "domain": "lock", "areas": "Hallway"},
        {"names": "Kitchen Light, Counter Light", "domain": "light", "areas": "Kitchen"},
    ]
