"""Hypothesis strategies for property-based testing (tests only, not production sampling)."""

from __future__ import annotations

from typing import Any

from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy

from generators.capability_registry import CAPABILITIES, trainable_operations
from generators.homes import generate_home, make_entity


def home_strategy(*, min_size: int = 8, max_size: int = 32) -> SearchStrategy[dict[str, Any]]:
    return st.builds(
        lambda seed, size: generate_home(seed, size, __import__("random").Random(seed ^ size)),
        seed=st.integers(min_value=0, max_value=999_999),
        size=st.integers(min_value=min_size, max_value=max_size),
    )


def entity_strategy(capability: str = "lights") -> SearchStrategy[dict[str, Any]]:
    return st.builds(
        lambda seed: make_entity(
            name=f"Test {capability} Device",
            capability=capability,
            area="Kitchen",
            floor="Main Floor",
            rng=__import__("random").Random(seed),
        ),
        seed=st.integers(min_value=0, max_value=999_999),
    )


def valid_brightness_strategy() -> SearchStrategy[int]:
    return st.integers(min_value=0, max_value=100)


def invalid_brightness_strategy() -> SearchStrategy[int]:
    return st.one_of(
        st.integers(max_value=-1),
        st.integers(min_value=101, max_value=1000),
    )


def capability_operation_strategy() -> SearchStrategy[tuple[str, str]]:
    cap_names = list(CAPABILITIES.keys())

    @st.composite
    def _pair(draw: st.DrawFn) -> tuple[str, str]:
        cap_name = draw(st.sampled_from(cap_names))
        cap = CAPABILITIES[cap_name]
        ops = trainable_operations(cap)
        if ops:
            return cap_name, draw(st.sampled_from([op.name for op in ops]))
        return cap_name, cap.operations[0].name

    return _pair()


def scenario_strategy(
    *,
    capability: str | None = None,
    operation: str | None = None,
) -> SearchStrategy[dict[str, Any]]:
    from generators.scenarios import build_scenario

    cap_names = [capability] if capability else list(CAPABILITIES.keys())

    @st.composite
    def _scenario(draw: st.DrawFn) -> dict[str, Any]:
        cap_name = draw(st.sampled_from(cap_names))
        cap = CAPABILITIES[cap_name]
        ops = trainable_operations(cap)
        op_name = operation or draw(
            st.sampled_from([op.name for op in ops] if ops else [cap.operations[0].name])
        )
        return build_scenario(
            index=draw(st.integers(min_value=0, max_value=10_000)),
            seed=draw(st.integers(min_value=0, max_value=999_999)),
            capability=cap_name,
            operation=op_name,
            home_size=draw(st.sampled_from([8, 16, 32])),
            attempt=draw(st.integers(min_value=0, max_value=5)),
        )

    return _scenario()


def valid_spec_strategy() -> SearchStrategy[dict[str, Any]]:
    from generators.labels import scenario_to_spec
    from generators.utterances import expand_utterance

    return scenario_strategy(capability="lights", operation="turn_on").map(
        lambda scenario: scenario_to_spec(
            {
                **scenario,
                "utterance": expand_utterance(
                    {
                        **scenario_to_spec(scenario),
                        "category": "clean_direct",
                    }
                ),
            }
        )
    )


def corrupted_spec_strategy(field: str) -> SearchStrategy[dict[str, Any]]:
    from generators.validator import corrupt_spec

    return valid_spec_strategy().filter(
        lambda spec: bool((spec.get("expected") or {}).get("calls"))
    ).map(lambda spec: corrupt_spec(spec, field))
