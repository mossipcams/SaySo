"""Reproduce the pinned English grammar subset from an OHF intents checkout.

Maintenance only: python training/generators/ohf_extract.py /path/to/intents
Requires PyYAML; runtime rendering consumes the resulting JSON directly.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml

REVISION = "c805585d4604c2819ce618f034758cda3bf92fbb"
INTENTS = (
    "HassTurnOn", "HassTurnOff", "HassLightSet", "HassFanSetSpeed",
    "HassClimateSetTemperature", "HassMediaPause", "HassMediaUnpause",
    "HassMediaNext", "HassMediaPrevious", "HassMediaPlayerMute",
    "HassMediaPlayerUnmute", "HassSetVolume", "HassSetVolumeRelative",
    "HassStartTimer", "HassPauseTimer", "HassUnpauseTimer", "HassCancelTimer",
    "HassCancelAllTimers", "HassIncreaseTimer", "HassDecreaseTimer", "HassTimerStatus",
)


def extract(root: Path) -> dict:
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"Expected OHF revision {REVISION}, got {revision}")
    result = {"repository": "https://github.com/OHF-Voice/intents", "revision": revision,
              "license": "CC-BY-4.0", "rules": {}, "blocks": [], "source_sha256": {}}
    paths = sorted((root / "rules/en").glob("*.yaml"))
    for intent in INTENTS:
        paths.extend(sorted((root / "sentences/en" / intent).glob("*.yaml")))
    for path in paths:
        source = str(path.relative_to(root))
        result["source_sha256"][source] = hashlib.sha256(path.read_bytes()).hexdigest()
        doc = yaml.safe_load(path.read_text())
        result["rules"].update(doc.get("expansion_rules", {}))
        for index, block in enumerate(doc.get("data", [])):
            result["blocks"].append({"intent": path.parent.name, "source": source, "block": index, **block})
    return result


if __name__ == "__main__":
    destination = Path(__file__).with_name("ohf_en.json")
    destination.write_text(json.dumps(extract(Path(sys.argv[1])), ensure_ascii=False, indent=2) + "\n")
