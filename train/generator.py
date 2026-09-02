"""Convert Home-LLM JSONL rows into SaySo SFT prompt + ControlPlan JSONL."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sayso_server.conversation import SatelliteConversationState
from sayso_server.home_graph import Area, Capability, CapabilityKind, Entity, State
from sayso_server.prompt import PromptOrigin, build_lfm_prompt

from train.home_llm import HomeLlmRow, ParsedDevice, parse_home_llm_row
from train.mapper import control_plan_json, map_row_to_control_plan

DEFAULT_SATELLITE_ID = "sft-satellite"
DEFAULT_AREA_ID = "area_sft"
DEFAULT_AREA_NAME = "Living Room"


@dataclass(frozen=True)
class GenerationStats:
    input_rows: int = 0
    kept_rows: int = 0
    dropped_rows: int = 0
    invalid_json_rows: int = 0


@dataclass(frozen=True)
class SftExample:
    prompt: str
    completion: str

    def to_json(self) -> dict[str, str]:
        return {"prompt": self.prompt, "completion": self.completion}


def convert_home_llm_jsonl(
    input_path: Path | str,
    output_path: Path | str,
) -> GenerationStats:
    """Read Home-LLM JSONL and write train-only SaySo SFT JSONL."""
    input_file = Path(input_path)
    output_file = Path(output_path)
    if "evals/datasets" in output_file.as_posix():
        msg = "train output must not be written under evals/datasets/"
        raise ValueError(msg)

    examples = list(generate_sft_examples(input_file.read_text().splitlines()))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_json(), ensure_ascii=False) + "\n")

    return _stats_from_counts(
        input_rows=_count_nonempty_lines(input_file),
        kept_rows=len(examples),
    )


def generate_sft_examples(lines: Iterable[str]) -> Iterable[SftExample]:
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        raw = json.loads(stripped)
        if not isinstance(raw, dict):
            continue
        row = parse_home_llm_row(raw)
        if row is None:
            continue
        example = row_to_sft_example(row)
        if example is not None:
            yield example


def row_to_sft_example(row: HomeLlmRow) -> SftExample | None:
    plan = map_row_to_control_plan(row)
    if plan is None:
        return None

    areas, candidates = _graph_context(row.devices)
    prompt = build_lfm_prompt(
        user_text=row.user_text,
        origin=PromptOrigin(
            satellite_id=DEFAULT_SATELLITE_ID,
            area_name=DEFAULT_AREA_NAME,
        ),
        conversation=SatelliteConversationState(),
        candidates=candidates,
        areas=areas,
    )
    return SftExample(prompt=prompt, completion=control_plan_json(plan))


def _graph_context(
    devices: tuple[ParsedDevice, ...],
) -> tuple[list[Area], list[Entity]]:
    area = Area(id=DEFAULT_AREA_ID, name=DEFAULT_AREA_NAME)
    entities = [_device_to_entity(device, area_id=area.id) for device in devices]
    return [area], entities


def _device_to_entity(device: ParsedDevice, *, area_id: str) -> Entity:
    return Entity(
        entity_id=device.entity_id,
        domain=device.domain,
        name=device.friendly_name,
        aliases=[],
        area_id=area_id,
        capabilities=_capabilities_for_domain(device.domain),
        state=State(value=device.state_value, attributes=dict(device.attributes)),
    )


def _capabilities_for_domain(domain: str) -> list[Capability]:
    if domain == "light":
        return [
            Capability(kind=CapabilityKind.POWER),
            Capability(kind=CapabilityKind.BRIGHTNESS, min_value=1, max_value=100),
        ]
    if domain == "climate":
        return [Capability(kind=CapabilityKind.TEMPERATURE, min_value=60, max_value=85)]
    return [Capability(kind=CapabilityKind.POWER)]


def _count_nonempty_lines(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def _stats_from_counts(*, input_rows: int, kept_rows: int) -> GenerationStats:
    dropped = max(0, input_rows - kept_rows)
    return GenerationStats(
        input_rows=input_rows,
        kept_rows=kept_rows,
        dropped_rows=dropped,
    )
