from __future__ import annotations

from pathlib import Path

from evals.cases import load_home
from evals.runner import _dump_yaml, exposed_entities

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ORIGINAL_EXCLUDED_DOMAINS = frozenset({"calendar", "script"})


def test_live_home_1_renders_live_static_context() -> None:
    rendered = _dump_yaml(exposed_entities(load_home("live_home_1")))
    expected = (FIXTURES / "live_home_1_static_context.yaml").read_text(
        encoding="utf-8"
    )
    assert rendered == expected


def test_home_without_context_entities_matches_original_rule() -> None:
    home = load_home("realistic_eval_household_1")
    assert "context_entities" not in home
    original = []
    for entity in sorted(home.get("entities", []), key=lambda item: item["name"]):
        if entity["domain"] in ORIGINAL_EXCLUDED_DOMAINS:
            continue
        names = dict.fromkeys([entity["name"], *entity.get("aliases", [])])
        info = {"names": ", ".join(names), "domain": entity["domain"]}
        if entity.get("area"):
            info["areas"] = entity["area"]
        original.append(info)
    assert exposed_entities(home) == original


def test_live_home_1_tv_is_followed_by_tv_active_app() -> None:
    names = [info["names"] for info in exposed_entities(load_home("live_home_1"))]
    assert names.index("TV") + 1 == names.index("TV Active app")
