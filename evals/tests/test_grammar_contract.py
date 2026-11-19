from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "schemas" / "sayso-tool-schema-v2.json"
PROBE = """
import json
import sys
from pathlib import Path


class _BlockHomeAssistant:
    def find_spec(self, name, path=None, target=None):
        if name == "homeassistant" or name.startswith("homeassistant."):
            raise ImportError("homeassistant is blocked")
        return None


sys.meta_path.insert(0, _BlockHomeAssistant())

import evals.runner as runner
from sayso_contract import grammar

tools = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["tools"]
text = grammar.build_grammar(tools, markers=False)
print(json.dumps({"runner": runner.__name__, "has_root": "root ::=" in text}))
"""


def test_grammar_builds_through_contract_without_homeassistant() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", PROBE, str(SCHEMA)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"runner": "evals.runner", "has_root": True}
