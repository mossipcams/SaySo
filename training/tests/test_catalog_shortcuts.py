"""Regressions for the v5 corpus shortcuts (docs/PLAN_V5_CORPUS_SHORTCUT_FIXES.md).

v5 let catalog shape predict the `unavailable` refusal: only those rows withheld a
tool, and homes never had exactly 2 scripts. Status requests were always
"status of X", and STT noise appended "." after "?".
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generators.homes import _capability_slots, generate_home
from generators.rendering import _decoy_removals
from generators.stt_noise import apply_log_stt_noise
from generators.utterances import request_seed_from_spec
from generators.validation import _STATUS_QUERY


def test_script_count_is_not_size_scaled():
    for size in (16, 64):
        counts = {_capability_slots(size, random.Random(seed)).count("scripts") for seed in range(200)}
        assert {1, 2, 3, 4} <= counts, (size, counts)


def test_decoys_withhold_only_unneeded_tools_at_the_set_rate():
    home = generate_home(0, 64, random.Random(1))
    messages = [{"role": "assistant", "tool_calls": [{"function": {"name": "intent__HassTurnOn"}}]}]
    withheld = 0
    for i in range(1000):
        spec = {
            "candidate_id": f"c{i}",
            "home": home,
            "capability": "lights",
            "operation": "set_brightness",
            "expected": {"kind": "action", "calls": []},
        }
        removed = _decoy_removals(spec, messages)
        assert "intent__HassTurnOn" not in removed
        assert "light__HassLightSet" not in removed  # the row's own operation tool
        withheld += bool(removed)
    assert 0.25 < withheld / 1000 < 0.35


def test_status_seeds_include_yes_no_questions_that_validate():
    seeds = [
        request_seed_from_spec({
            "candidate_id": f"c{i}",
            "target_names": ["Front Door Lock"],
            "expected": {"kind": "status", "calls": [
                {"name": "GetLiveContext", "arguments": {"domain": "lock", "name": "Front Door Lock"}}
            ]},
        })
        for i in range(200)
    ]
    assert any(s.startswith("is ") for s in seeds)
    assert any("status of" in s for s in seeds)
    assert all(_STATUS_QUERY.search(s) for s in seeds)


def test_log_stt_period_never_follows_a_question_mark():
    for seed in range(300):
        text, _ = apply_log_stt_noise("Is the kitchen light on?", random.Random(seed))
        assert not text.rstrip().endswith("?."), text
