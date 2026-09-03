"""Load Home-LLM English pile fixtures for SaySo generation."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILES_DIR = ROOT / "fixtures" / "piles" / "english"

VAR_PATTERN = re.compile(r"<(.*?)>")


def _contains_vars(text: str) -> str:
    vars_found = [var for var in VAR_PATTERN.findall(text) if var != "device_name"]
    return ",".join(sorted(vars_found))


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@dataclass
class DatasetPiles:
    """In-memory Home-LLM English piles."""

    language: str
    and_words: list[str]
    pile_of_durations: dict[str, str]
    pile_of_media_names: list[str]
    pile_of_todo_items: list[str]
    stacks_of_device_names: dict[str, list[dict[str, str]]]
    pile_of_templated_actions: list[dict[str, str]]
    pile_of_specific_actions: list[dict[str, str]]
    pile_of_responses: list[dict[str, str]]
    pile_of_status_requests: list[dict[str, str]]
    pile_of_system_prompts: dict[str, str]
    pile_of_failed_tool_calls: list[dict[str, str]]
    pile_of_refusals: list[dict[str, str]]

    @classmethod
    def load(cls, language: str = "english", piles_dir: Path | None = None) -> DatasetPiles:
        base = piles_dir or (ROOT / "fixtures" / "piles" / language)
        if not base.is_dir():
            raise FileNotFoundError(f"Missing pile directory: {base}")

        and_words = _read_lines(base / "pile_of_and_words.csv")
        durations = {row["duration"]: row["name"] for row in _read_csv(base / "pile_of_durations.csv")}
        media_names = _read_lines(base / "pile_of_media_names.txt")
        todo_items = _read_lines(base / "pile_of_todo_items.txt")

        stacks: dict[str, list[dict[str, str]]] = {}
        for row in _read_csv(base / "pile_of_device_names.csv"):
            device_type = row["device_name"].split(".", 1)[0]
            stacks.setdefault(device_type, []).append(row)

        templated: list[dict[str, str]] = []
        for row in _read_csv(base / "pile_of_templated_actions.csv"):
            multiplier = int(row["multiplier"])
            templated.extend([row] * multiplier)

        responses = _read_csv(base / "pile_of_responses.csv")
        for row in responses:
            row["contains_vars"] = _contains_vars(row["response_starting"])
            row["short"] = row.get("short", "0")

        return cls(
            language=language,
            and_words=and_words,
            pile_of_durations=durations,
            pile_of_media_names=media_names,
            pile_of_todo_items=todo_items,
            stacks_of_device_names=stacks,
            pile_of_templated_actions=templated,
            pile_of_specific_actions=_read_csv(base / "pile_of_specific_actions.csv"),
            pile_of_responses=responses,
            pile_of_status_requests=_read_csv(base / "pile_of_status_requests.csv"),
            pile_of_system_prompts={
                row["persona"]: row["prompt"] for row in _read_csv(base / "pile_of_system_prompts.csv")
            },
            pile_of_failed_tool_calls=_read_csv(base / "pile_of_failed_tool_calls.csv"),
            pile_of_refusals=_read_csv(base / "pile_of_refusals.csv"),
        )


@dataclass
class GenerationStats:
    """Emitted and dropped example counts."""

    emitted: int = 0
    dropped: dict[str, int] = field(default_factory=dict)

    def record_drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    def record_emit(self) -> None:
        self.emitted += 1


def get_random_response(
    piles: DatasetPiles,
    *,
    service: str,
    persona: str,
    question_template: str,
    short: bool,
    rng: Any,
) -> tuple[str, str] | None:
    required_vars = sorted(
        {var for var in VAR_PATTERN.findall(question_template) if not var.startswith("device_name")}
    )
    required_key = ",".join(required_vars)
    short_flag = "1" if short else "0"
    matches = [
        row
        for row in piles.pile_of_responses
        if row["service"] == service
        and row["persona"] == persona
        and row.get("short", "0") == short_flag
        and row.get("contains_vars", "") == required_key
    ]
    if not matches:
        return None
    row = rng.choice(matches)
    return row["response_starting"], row["response_confirmed"]
