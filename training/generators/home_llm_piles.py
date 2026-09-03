"""Generate SaySo training examples from Home-LLM English piles (v1-only)."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Iterator

from adapters.schema import v1_openai_tools

from .piles import DatasetPiles, GenerationStats, get_random_response
from .v1_map import (
    COLOR_NAMES,
    V1ToolCall,
    build_cancel_all_timers_call,
    build_get_datetime_call,
    build_live_context_call,
    build_tool_call,
    drop_reason_for_service,
    is_mappable_service,
    is_mappable_status_device,
    slug_to_friendly,
)

@dataclass(frozen=True, slots=True)
class GenerationFactors:
    static_factor: float = 1.0
    template_factor: int = 10
    status_request_factor: int = 8
    refusal_factor: int = 3
    failure_factor: int = 1


SMALL_FACTORS = GenerationFactors(1, 10, 8, 3, 1)
SAMPLE_FACTORS = GenerationFactors(1, 1, 1, 1, 1)

_DATETIME_QUESTIONS: tuple[str, ...] = (
    "what time is it",
    "what's the current time",
    "can you tell me the time",
    "do you know what time it is",
    "what is the date and time",
)

_DATETIME_CONTEXT_QUESTIONS: tuple[str, ...] = (
    "i'm by the {device} — what time is it",
    "before i check the {device}, what time is it",
    "in the {area}, what time is it right now",
)

_CANCEL_ALL_QUESTIONS: tuple[str, ...] = (
    "cancel all timers",
    "stop all timers",
    "clear every timer",
    "turn off all timers",
    "shut off every timer",
)

_CANCEL_AREA_QUESTIONS: tuple[str, ...] = (
    "cancel all timers in the {area}",
    "stop every timer in the {area}",
    "clear all timers in the {area}",
)


def _sys(persona_prompt: str) -> dict[str, Any]:
    return {
        "role": "system",
        "content": [{"type": "text", "text": persona_prompt}],
        "train_on_turn": False,
    }


def _user(text: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [{"type": "text", "text": text}],
        "train_on_turn": False,
    }


def _assistant_tool(calls: list[V1ToolCall], *, text: str = "") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "tool_calls": [
            {
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                }
            }
            for call in calls
        ],
        "train_on_turn": True,
    }


def _assistant_text(text: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "train_on_turn": True,
    }


def _tool_result(name: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "content": [{"name": name, "response": result}],
        "train_on_turn": False,
    }


def _tool_results(pairs: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    return {
        "role": "tool",
        "content": [{"name": name, "response": result} for name, result in pairs],
        "train_on_turn": False,
    }


def _entry(
    *,
    persona_prompt: str,
    messages: list[dict[str, Any]],
    template_family: str,
    phrasing_family: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "messages": [_sys(persona_prompt), *messages],
        "tools": v1_openai_tools(),
        "metadata": {
            "template_family": template_family,
            "phrasing_family": phrasing_family,
            "seed": seed,
        },
    }


def _replace_device(text: str, friendly_name: str) -> str:
    return text.replace("<device_name>", friendly_name)


def _has_var(template: str, name: str) -> bool:
    return f"<{name}>" in template


def _fill_light_params(
    template: str,
    rng: random.Random,
) -> tuple[str, int | None, str | None]:
    text = template
    brightness: int | None = None
    color: str | None = None
    if _has_var(text, "brightness"):
        brightness = rng.randint(10, 100)
        text = text.replace("<brightness>", str(brightness))
    if _has_var(text, "color"):
        color = rng.choice(COLOR_NAMES)
        text = text.replace("<color>", color)
    return text, brightness, color


def _pick_device(piles: DatasetPiles, device_type: str, rng: random.Random) -> dict[str, str] | None:
    stack = piles.stacks_of_device_names.get(device_type) or []
    if not stack:
        return None
    return rng.choice(stack)


def _pick_area_label(piles: DatasetPiles, rng: random.Random) -> str:
    device = _pick_device(piles, "light", rng) or _pick_device(piles, "fan", rng)
    if device is None:
        return "Kitchen"
    slug = device["device_name"].split(".", 1)[-1]
    area_slug = slug.split("_")[0]
    return slug_to_friendly(area_slug)


def _timer_cancel_persona_reply(
    piles: DatasetPiles,
    persona: str,
    rng: random.Random,
    *,
    all_timers: bool,
) -> tuple[str, str]:
    matches = [
        row
        for row in piles.pile_of_responses
        if row["service"] == "timer.cancel" and row["persona"] == persona
    ]
    if matches:
        row = rng.choice(matches)
        starting = row["response_starting"]
        confirmed = row["response_confirmed"]
        if all_timers:
            starting = starting.replace("the timer", "all timers").replace("Timer", "all timers")
            confirmed = confirmed.replace("the timer", "all timers").replace("Timer", "all timers")
        return starting.lower(), confirmed.lower()
    if all_timers:
        return ("canceling all timers now.", "all timers have been canceled.")
    return ("canceling the timer now.", "the timer has been canceled.")


def _datetime_persona_reply(persona: str, rng: random.Random) -> tuple[str, str]:
    hour = rng.randint(1, 12)
    minute = rng.choice([0, 15, 30, 45])
    meridiem = rng.choice(["am", "pm"])
    spoken = f"{hour}:{minute:02d} {meridiem}"
    starting = "let me check the time."
    if persona == "pirate":
        confirmed = f"arr, it be {spoken}."
    elif persona == "robot":
        confirmed = f"beep-boop. the current time is {spoken}."
    else:
        confirmed = f"it's {spoken}."
    return starting, confirmed


def _services_from_template(row: dict[str, str]) -> list[str]:
    device_types = row["device_type"].split("|")
    services = row["service"].split("|")
    return [f"{dtype}.{service}" for dtype, service in zip(device_types, services)]


def _generic_responses(service: str, friendly_name: str) -> tuple[str, str]:
    return (
        f"Working on {friendly_name}.",
        f"Done with {friendly_name}.",
    )


def _build_from_tool_flow(
    *,
    persona_prompt: str,
    question: str,
    calls: list[V1ToolCall],
    response_starting: str,
    response_confirmed: str,
    template_family: str,
    phrasing_family: str,
    seed: int,
    tool_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if tool_results is None:
        tool_results = [{"result": "Success"} for _ in calls]
    messages = [
        _user(question),
        _assistant_tool(calls, text=response_starting.strip()),
        _tool_results(list(zip([c.name for c in calls], tool_results))),
        _assistant_text(response_confirmed.strip()),
    ]
    return _entry(
        persona_prompt=persona_prompt,
        messages=messages,
        template_family=template_family,
        phrasing_family=phrasing_family,
        seed=seed,
    )


def _generate_specific(
    piles: DatasetPiles,
    action: dict[str, str],
    persona: str,
    persona_prompt: str,
    rng: random.Random,
    seed: int,
) -> dict[str, Any] | None:
    service_name = action["service_name"]
    reason = drop_reason_for_service(service_name)
    if reason:
        return None

    device_type = service_name.split(".", 1)[0]
    device_slug = action["device_name"]
    entity_id = f"{device_type}.{device_slug}"
    device = _pick_device(piles, device_type, rng)
    friendly_name = device["description"] if device else slug_to_friendly(device_slug)

    question = _replace_device(action["phrase"], friendly_name).lower()
    has_brightness = _has_var(action["phrase"], "brightness")
    has_color = _has_var(action["phrase"], "color")
    question, brightness, color = _fill_light_params(question, rng)

    call = build_tool_call(
        service_name,
        friendly_name,
        has_brightness=has_brightness or brightness is not None,
        has_color=has_color or color is not None,
        brightness=brightness,
        color=color,
    )
    if call is None:
        return None

    responses = get_random_response(
        piles,
        service=service_name,
        persona=persona,
        question_template=action["phrase"],
        short=False,
        rng=rng,
    ) or _generic_responses(service_name, friendly_name)
    starting, confirmed = responses
    starting = _replace_device(starting, friendly_name).lower()
    confirmed = _replace_device(confirmed, friendly_name).lower()

    return _build_from_tool_flow(
        persona_prompt=persona_prompt,
        question=question,
        calls=[call],
        response_starting=starting,
        response_confirmed=confirmed,
        template_family="pile_specific",
        phrasing_family=service_name,
        seed=seed,
    )


def _generate_templated(
    piles: DatasetPiles,
    template: dict[str, str],
    persona: str,
    persona_prompt: str,
    rng: random.Random,
    seed: int,
) -> dict[str, Any] | None:
    services = _services_from_template(template)
    for service in services:
        if drop_reason_for_service(service):
            return None

    device_types = template["device_type"].split("|")
    chosen: list[dict[str, str]] = []
    for device_type in device_types:
        device = _pick_device(piles, device_type, rng)
        if device is None:
            return None
        chosen.append(device)

    question_template = template["phrase"]
    has_brightness = _has_var(question_template, "brightness")
    has_color = _has_var(question_template, "color")

    if len(chosen) == 1:
        friendly = chosen[0]["description"]
        question = _replace_device(question_template, friendly)
        question, brightness, color = _fill_light_params(question, rng)
        call = build_tool_call(
            services[0],
            friendly,
            has_brightness=has_brightness or brightness is not None,
            has_color=has_color or color is not None,
            brightness=brightness,
            color=color,
        )
        if call is None:
            return None
        responses = get_random_response(
            piles,
            service=services[0],
            persona=persona,
            question_template=question_template,
            short=False,
            rng=rng,
        ) or _generic_responses(services[0], friendly)
        starting, confirmed = responses
        starting = _replace_device(starting, friendly).lower()
        confirmed = _replace_device(confirmed, friendly).lower()
        return _build_from_tool_flow(
            persona_prompt=persona_prompt,
            question=question.lower(),
            calls=[call],
            response_starting=starting,
            response_confirmed=confirmed,
            template_family="pile_templated",
            phrasing_family=question_template,
            seed=seed,
        )

    # Multi-action templates (all services already verified mappable).
    question = question_template
    calls: list[V1ToolCall] = []
    confirmed_parts: list[str] = []
    starting_parts: list[str] = []
    for index, (device, service) in enumerate(zip(chosen, services)):
        placeholder = f"<device_name{index + 1}>"
        friendly = device["description"]
        question = question.replace(placeholder, friendly)
        call = build_tool_call(service, friendly)
        if call is None:
            return None
        calls.append(call)
        responses = get_random_response(
            piles,
            service=service,
            persona=persona,
            question_template=question_template,
            short=True,
            rng=rng,
        ) or _generic_responses(service, friendly)
        start, confirm = responses
        starting_parts.append(_replace_device(start, friendly).strip())
        confirmed_parts.append(_replace_device(confirm, friendly).strip())

    joiner = f" {rng.choice(piles.and_words)} "
    return _build_from_tool_flow(
        persona_prompt=persona_prompt,
        question=question.lower(),
        calls=calls,
        response_starting=joiner.join(starting_parts).lower(),
        response_confirmed=joiner.join(confirmed_parts).lower(),
        template_family="pile_templated_multi",
        phrasing_family=question_template,
        seed=seed,
    )


def _generate_status(
    piles: DatasetPiles,
    template: dict[str, str],
    persona_prompt: str,
    rng: random.Random,
    seed: int,
) -> dict[str, Any] | None:
    device_type = template["device_type"]
    if not is_mappable_status_device(device_type):
        return None
    device = _pick_device(piles, device_type, rng)
    if device is None:
        return None
    friendly = device["description"]
    question = _replace_device(template["phrase"], friendly).lower()
    answer = _replace_device(template["assistant_response"], friendly).lower()
    call = build_live_context_call(friendly, device_type)
    return _build_from_tool_flow(
        persona_prompt=persona_prompt,
        question=question,
        calls=[call],
        response_starting="Let me check.",
        response_confirmed=answer,
        template_family="pile_status",
        phrasing_family=template["phrase"],
        seed=seed,
        tool_results=[{"areas": {friendly: {"state": template["state"]}}}],
    )


def _generate_refusal(
    piles: DatasetPiles,
    refusal: dict[str, str],
    persona_prompt: str,
    rng: random.Random,
    seed: int,
) -> dict[str, Any]:
    service_name = refusal["service_name"]
    friendly = refusal.get("friendly_name") or slug_to_friendly(refusal["device_name"])
    question = _replace_device(refusal["phrase"], friendly).lower()
    answer = _replace_device(refusal["response"], friendly).lower()
    return _entry(
        persona_prompt=persona_prompt,
        messages=[_user(question), _assistant_text(answer)],
        template_family="pile_refusal",
        phrasing_family=refusal.get("reason_type", service_name),
        seed=seed,
    )


def _generate_failure(
    piles: DatasetPiles,
    failure: dict[str, str],
    persona: str,
    persona_prompt: str,
    rng: random.Random,
    seed: int,
) -> dict[str, Any] | None:
    service_name = failure["service_name"]
    if drop_reason_for_service(service_name):
        return None

    friendly = failure.get("correct_friendly_name") or slug_to_friendly(
        failure["correct_device_name"].split(".", 1)[-1]
    )
    bad_name = slug_to_friendly(failure["bad_device_name"].split(".", 1)[-1])
    question = _replace_device(failure["phrase"], friendly).lower()

    good_call = build_tool_call(service_name, friendly)
    bad_call = build_tool_call(service_name, bad_name)
    if good_call is None or bad_call is None:
        return None

    responses = get_random_response(
        piles,
        service=service_name,
        persona=persona,
        question_template=failure["phrase"],
        short=False,
        rng=rng,
    ) or _generic_responses(service_name, friendly)
    starting, confirmed = responses
    starting = _replace_device(starting, friendly).lower()
    confirmed = _replace_device(confirmed, friendly).lower()
    retry = failure.get("retry_prompt", f"Trying again with {friendly}.").replace(
        "<device_name>", friendly
    )

    messages = [
        _user(question),
        _assistant_tool([bad_call], text=starting),
        _tool_result(bad_call.name, {"result": "Failed", "error": "Entity not found"}),
        _assistant_tool([good_call], text=retry),
        _tool_result(good_call.name, {"result": "Success"}),
        _assistant_text(confirmed),
    ]
    return _entry(
        persona_prompt=persona_prompt,
        messages=messages,
        template_family="pile_failure",
        phrasing_family=service_name,
        seed=seed,
    )


def _generate_datetime(
    piles: DatasetPiles,
    persona: str,
    persona_prompt: str,
    rng: random.Random,
    seed: int,
) -> dict[str, Any]:
    if rng.random() < 0.5:
        question = rng.choice(_DATETIME_QUESTIONS)
    else:
        device = _pick_device(piles, "light", rng) or _pick_device(piles, "fan", rng)
        friendly = device["description"] if device else "living room light"
        area = _pick_area_label(piles, rng)
        template = rng.choice(_DATETIME_CONTEXT_QUESTIONS)
        question = template.format(device=friendly.lower(), area=area.lower())
    call = build_get_datetime_call()
    starting, confirmed = _datetime_persona_reply(persona, rng)
    return _build_from_tool_flow(
        persona_prompt=persona_prompt,
        question=question,
        calls=[call],
        response_starting=starting,
        response_confirmed=confirmed,
        template_family="pile_synth_datetime",
        phrasing_family=question,
        seed=seed,
        tool_results=[{"datetime": "2026-03-20T14:30:00"}],
    )


def _generate_cancel_all_timers(
    piles: DatasetPiles,
    persona: str,
    persona_prompt: str,
    rng: random.Random,
    seed: int,
) -> dict[str, Any]:
    area: str | None = None
    if rng.random() < 0.4:
        area = _pick_area_label(piles, rng)
        question = rng.choice(_CANCEL_AREA_QUESTIONS).format(area=area.lower())
        call = build_cancel_all_timers_call(area=area)
    else:
        question = rng.choice(_CANCEL_ALL_QUESTIONS)
        call = build_cancel_all_timers_call()
    starting, confirmed = _timer_cancel_persona_reply(
        piles,
        persona,
        rng,
        all_timers=True,
    )
    return _build_from_tool_flow(
        persona_prompt=persona_prompt,
        question=question,
        calls=[call],
        response_starting=starting,
        response_confirmed=confirmed,
        template_family="pile_synth_cancel_all_timers",
        phrasing_family=question,
        seed=seed,
    )


def _run_factor(
    rng: random.Random,
    factor: float | int,
    callback: Any,
) -> Iterator[dict[str, Any]]:
    if factor >= 1:
        for _ in range(int(factor)):
            example = callback()
            if example is not None:
                yield example
    elif factor > 0 and rng.random() < factor:
        example = callback()
        if example is not None:
            yield example


def generate_pile_examples(
    *,
    seed: int = 42,
    factors: GenerationFactors | None = None,
    piles: DatasetPiles | None = None,
    stats: GenerationStats | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield Home-LLM V2 shaped examples from English piles (v1 tools only)."""
    rng = random.Random(seed)
    piles = piles or DatasetPiles.load()
    factors = factors or SMALL_FACTORS
    stats = stats or GenerationStats()
    personas = list(piles.pile_of_system_prompts.items())
    example_seed = seed

    for persona, persona_prompt in personas:
        for action in piles.pile_of_specific_actions:
            reason = drop_reason_for_service(action["service_name"])

            def _static(
                a: dict[str, str] = action,
                drop_reason: str | None = reason,
            ) -> dict[str, Any] | None:
                nonlocal example_seed
                example_seed += 1
                if drop_reason:
                    stats.record_drop(drop_reason)
                    return None
                built = _generate_specific(piles, a, persona, persona_prompt, rng, example_seed)
                if built is None:
                    stats.record_drop("build_failed_specific")
                return built

            for example in _run_factor(rng, factors.static_factor, _static):
                stats.record_emit()
                yield example

        for template in piles.pile_of_templated_actions:
            services = _services_from_template(template)
            reason = next((drop_reason_for_service(s) for s in services if drop_reason_for_service(s)), None)

            def _templated(
                t: dict[str, str] = template,
                drop_reason: str | None = reason,
            ) -> dict[str, Any] | None:
                nonlocal example_seed
                example_seed += 1
                if drop_reason:
                    stats.record_drop(drop_reason)
                    return None
                built = _generate_templated(piles, t, persona, persona_prompt, rng, example_seed)
                if built is None:
                    stats.record_drop("build_failed_templated")
                return built

            for example in _run_factor(rng, factors.template_factor, _templated):
                stats.record_emit()
                yield example

        for failure in piles.pile_of_failed_tool_calls:
            reason = drop_reason_for_service(failure["service_name"])

            def _failure(
                f: dict[str, str] = failure,
                drop_reason: str | None = reason,
            ) -> dict[str, Any] | None:
                nonlocal example_seed
                example_seed += 1
                if drop_reason:
                    stats.record_drop(drop_reason)
                    return None
                built = _generate_failure(piles, f, persona, persona_prompt, rng, example_seed)
                if built is None:
                    stats.record_drop("build_failed_failure")
                return built

            for example in _run_factor(rng, factors.failure_factor, _failure):
                stats.record_emit()
                yield example

        for refusal in piles.pile_of_refusals:
            def _refusal(r: dict[str, str] = refusal) -> dict[str, Any]:
                nonlocal example_seed
                example_seed += 1
                return _generate_refusal(piles, r, persona_prompt, rng, example_seed)

            for example in _run_factor(rng, factors.refusal_factor, _refusal):
                stats.record_emit()
                yield example

    for status in piles.pile_of_status_requests:
        device_type = status["device_type"]

        def _status(
            s: dict[str, str] = status,
            dtype: str = device_type,
        ) -> dict[str, Any] | None:
            nonlocal example_seed
            example_seed += 1
            if not is_mappable_status_device(dtype):
                stats.record_drop(f"non_v1_status:{dtype}")
                return None
            built = _generate_status(
                piles,
                s,
                piles.pile_of_system_prompts["assistant"],
                rng,
                example_seed,
            )
            if built is None:
                stats.record_drop("build_failed_status")
            return built

        for example in _run_factor(rng, factors.status_request_factor, _status):
            stats.record_emit()
            yield example

    for persona, persona_prompt in personas:
        def _datetime(p: str = persona, pp: str = persona_prompt) -> dict[str, Any]:
            nonlocal example_seed
            example_seed += 1
            return _generate_datetime(piles, p, pp, rng, example_seed)

        for example in _run_factor(rng, factors.static_factor, _datetime):
            stats.record_emit()
            yield example

        def _cancel_all(p: str = persona, pp: str = persona_prompt) -> dict[str, Any]:
            nonlocal example_seed
            example_seed += 1
            return _generate_cancel_all_timers(piles, p, pp, rng, example_seed)

        for example in _run_factor(rng, factors.static_factor, _cancel_all):
            stats.record_emit()
            yield example


def estimate_example_count(factors: GenerationFactors | None = None, piles: DatasetPiles | None = None) -> int:
    """Upper-bound count if every pile row maps successfully."""
    piles = piles or DatasetPiles.load()
    factors = factors or SMALL_FACTORS
    personas = len(piles.pile_of_system_prompts)
    specific = sum(1 for row in piles.pile_of_specific_actions if is_mappable_service(row["service_name"]))
    templated = sum(
        1 for row in piles.pile_of_templated_actions if all(is_mappable_service(s) for s in _services_from_template(row))
    )
    failures = sum(1 for row in piles.pile_of_failed_tool_calls if is_mappable_service(row["service_name"]))
    refusals = len(piles.pile_of_refusals)
    status = sum(1 for row in piles.pile_of_status_requests if is_mappable_status_device(row["device_type"]))
    total = 0
    total += int(specific * factors.static_factor * personas)
    total += int(templated * factors.template_factor * personas)
    total += int(failures * factors.failure_factor * personas)
    total += int(refusals * factors.refusal_factor * personas)
    total += int(status * factors.status_request_factor)
    total += int(2 * factors.static_factor * personas)
    return total
