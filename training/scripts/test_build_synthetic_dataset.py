"""Contract tests for the eval-directed synthetic dataset builder."""

from __future__ import annotations

import sys
import threading
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import json

from build_synthetic_dataset import (
    CATEGORY_WEIGHTS,
    _CLEAN_DIRECT_START,
    _framed_utterance,
    _is_generic_no_area_hint,
    _judge_prompt,
    build_specs,
    curate,
    expand_utterance,
    framing_response_format,
    judge_batch,
    judge_resilient,
    judge_response_format,
    load_user_utterances,
    openai_complete,
    request_seed,
    template_seed,
    render_example,
    run_pipeline,
    validate_spec,
    validate_utterance,
    verbalize_batch,
    verbalize_resilient,
)


def test_specs_are_label_first_deterministic_and_balanced() -> None:
    first = build_specs(100, seed=73)
    second = build_specs(100, seed=73)

    assert first == second
    assert Counter(spec["category"] for spec in first) == CATEGORY_WEIGHTS
    assert all(spec["utterance"] is None for spec in first)
    assert all(spec["expected"] and spec["home"]["entities"] for spec in first)
    assert len({spec["home"]["home_id"] for spec in first}) == 100


def test_specs_include_contrastive_identity_and_hard_cases() -> None:
    specs = build_specs(200, seed=91)
    categories = {spec["category"] for spec in specs}

    assert categories == set(CATEGORY_WEIGHTS)
    assert any(spec["contrastive_group"] for spec in specs)
    assert any("'" in entity["name"] for spec in specs for entity in spec["home"]["entities"])
    assert any(spec["expected"]["kind"] == "no_action" for spec in specs)
    assert any(len(spec["expected"].get("calls", [])) >= 2 for spec in specs)
    assert any(spec["category"] == "stt_corrupted" and spec["spoken_targets"] for spec in specs)
    assert all(
        entity["entity_id"].split(".", 1)[0] in {"light", "fan", "switch", "cover", "lock"}
        for spec in specs
        for entity in spec["home"]["entities"]
    )


def test_entity_names_match_their_actual_capability() -> None:
    nouns = {
        "light": "light",
        "fan": "fan",
        "switch": "outlet",
        "blinds": "blinds",
        "garage_door": "garage door",
        "lock": "lock",
    }
    for spec in build_specs(1_000, seed=100):
        for entity in spec["home"]["entities"]:
            assert nouns[entity["kind"]] in entity["name"].casefold()


def test_stt_specs_use_varied_corruptions_but_keep_canonical_labels() -> None:
    stt = [spec for spec in build_specs(1_000, seed=97) if spec["category"] == "stt_corrupted"]

    assert len({spec["stt_corruption"] for spec in stt}) >= 6
    for spec in stt:
        canonical = spec["target_names"][0]
        spoken = spec["spoken_targets"][canonical]
        assert spoken.casefold() != canonical.casefold()
        assert spec["expected"]["calls"][0]["arguments"]["name"] == canonical


def test_contrastive_trios_share_context_but_change_expected_behavior() -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for spec in build_specs(200, seed=95):
        if spec["contrastive_group"]:
            groups[spec["contrastive_group"]].append(spec)
    trio = next(rows for rows in groups.values() if len(rows) == 3)
    contexts = [
        json.dumps(row["home"]["entities"], sort_keys=True)
        for row in trio
    ]
    assert len(set(contexts)) == 1
    assert {row["expected"]["kind"] for row in trio} == {"action", "status", "no_action"}
    ambiguity = next(row for row in trio if row["category"] == "ambiguity")
    sayso_area = ambiguity["home"]["sayso_entity_area"]
    hint = ambiguity["request_hint"]
    assert _is_generic_no_area_hint(hint)
    generic = hint.rsplit("the ", 1)[-1]
    kind = {
        "light": "light",
        "fan": "fan",
        "outlet": "switch",
        "blinds": "blinds",
        "garage door": "garage_door",
        "door": "lock",
    }[generic]
    in_area = [
        entity
        for entity in ambiguity["home"]["entities"]
        if entity["kind"] == kind and entity["area"] == sayso_area
    ]
    expected = ambiguity["expected"]
    if len(in_area) == 0:
        assert expected["response"] == "area_unavailable"
    elif len(in_area) == 1:
        assert expected["kind"] == "action"
        assert expected["calls"][0]["arguments"]["name"] == in_area[0]["name"]
    else:
        assert expected["response"] == "clarify"
        assert sum(generic in entity["aliases"] for entity in in_area) >= 2


def test_ground_truth_renders_in_canonical_sayso_shape() -> None:
    specs = build_specs(200, seed=14)
    for spec in specs:
        assert validate_spec(spec) is None
        spec["utterance"] = "synthetic spoken request"
        example = render_example(spec)
        assert example["tools"]
        calls = [call for message in example["messages"] for call in message.get("tool_calls", [])]
        assert len(calls) == len(spec["expected"].get("calls", []))
        for call in calls:
            assert call["type"] == "function"
            assert isinstance(call["function"]["arguments"], str)
            assert isinstance(json.loads(call["function"]["arguments"]), dict)


def test_rendered_split_families_keep_contrasts_together_without_one_global_seed() -> None:
    specs = build_specs(300, seed=98)
    trio = next(
        [spec for spec in specs if spec["contrastive_group"] == group]
        for group in {spec["contrastive_group"] for spec in specs if spec["contrastive_group"]}
    )
    for spec in trio:
        spec["utterance"] = request_seed(spec)
    families = {
        (
            render_example(spec)["metadata"]["template_family"],
            render_example(spec)["metadata"]["phrasing_family"],
            render_example(spec)["metadata"]["seed"],
        )
        for spec in trio
    }
    assert len(families) == 1

    ordinary = [spec for spec in specs if not spec["contrastive_group"]][:2]
    for spec in ordinary:
        spec["utterance"] = request_seed(spec)
    assert len({render_example(spec)["metadata"]["seed"] for spec in ordinary}) == 2


def test_no_action_exclusions_and_apostrophes_survive_serialization() -> None:
    specs = build_specs(300, seed=19)
    no_action = next(spec for spec in specs if spec["expected"]["kind"] == "no_action")
    no_action["utterance"] = "which lamp did you mean"
    assert not any(message.get("tool_calls") for message in render_example(no_action)["messages"])

    exclusion = next(spec for spec in specs if spec["excluded_names"])
    exclusion["utterance"] = "do the first two but leave the other one alone"
    rendered = render_example(exclusion)
    args = [
        json.loads(call["function"]["arguments"])
        for message in rendered["messages"]
        for call in message.get("tool_calls", [])
    ]
    assert not set(exclusion["excluded_names"]) & {arg.get("name") for arg in args}

    apostrophe = next(
        spec
        for spec in specs
        if any("'" in call["arguments"].get("name", "") for call in spec["expected"].get("calls", []))
    )
    apostrophe["utterance"] = "use the named device"
    line = json.dumps(render_example(apostrophe), ensure_ascii=False)
    assert "'" in line
    assert json.loads(line)


def test_schema_or_context_mismatch_is_rejected() -> None:
    spec = next(spec for spec in build_specs(100, seed=8) if spec["expected"].get("calls"))
    spec["expected"]["calls"][0]["arguments"]["name"] = "Invented Device"
    assert validate_spec(spec) == "unknown_canonical_entity"


def test_verbalizer_only_fills_language_for_existing_labels() -> None:
    specs = build_specs(100, seed=27)[:3]
    labels_before = [json.dumps(spec["expected"], sort_keys=True) for spec in specs]

    def complete(prompt: str) -> dict:
        assert "authoritative" in prompt
        assert '"home":' not in prompt
        requested = [spec for spec in specs if spec["candidate_id"] in prompt]
        return {
            "items": [
                {
                    "candidate_id": spec["candidate_id"],
                    "utterance": f"spoken {spec['candidate_id']} {template_seed(spec)}",
                }
                for spec in requested
            ]
        }

    verbalized, rejected = verbalize_batch(specs, complete)
    assert not rejected
    assert [json.dumps(spec["expected"], sort_keys=True) for spec in verbalized] == labels_before
    assert all(spec["utterance"].startswith("spoken candidate_") for spec in verbalized)


def test_language_seed_is_derived_from_calls_targets_and_exclusions() -> None:
    for spec in build_specs(300, seed=28):
        seed = request_seed(spec).casefold()
        expected = spec["expected"]
        if expected["kind"] == "no_action":
            assert seed == spec["request_hint"].casefold()
            continue
        for name in spec["target_names"]:
            assert spec["spoken_targets"].get(name, name).casefold() in seed
        for name in spec["excluded_names"]:
            assert name.casefold() in seed
            assert "leave" in seed
        if expected["kind"] == "status":
            assert "status" in seed


def test_verbalizer_protects_identity_and_value_slots() -> None:
    spec = next(
        spec
        for spec in build_specs(300, seed=29)
        if spec["target_names"] and not spec["spoken_targets"]
    )
    protected = template_seed(spec)
    assert "<TARGET_1>" in protected
    assert spec["target_names"][0] not in protected

    verbalized, rejected = verbalize_batch(
        [spec], lambda _prompt: {"utterances": [f"Hey, could you {protected} please?"]}
    )
    assert not rejected
    assert spec["target_names"][0] in verbalized[0]["utterance"]
    assert "<TARGET_1>" not in verbalized[0]["utterance"]

    verbalized, rejected = verbalize_batch(
        [spec], lambda _prompt: {"utterances": ["turn on a different lamp"]}
    )
    assert not verbalized
    assert rejected == {spec["candidate_id"]: "verbalizer_missing_item"}


def test_llm_framing_cannot_change_authoritative_semantics() -> None:
    specs = [spec for spec in build_specs(100, seed=30) if spec["expected"]["kind"] == "action"][:2]
    verbalized, rejected = verbalize_batch(
        specs,
        lambda prompt: (
            {"framings": [["Hey, could you please", "for me?"], ["Uh", "please."]]}
            if '"framings"' in prompt
            else {}
        ),
    )
    assert not rejected
    assert verbalized[0]["utterance"].startswith("Hey, could you please ")
    assert request_seed(specs[0]) in verbalized[0]["utterance"]
    assert request_seed(specs[1]) in verbalized[1]["utterance"]

    verbalized, rejected = verbalize_batch(
        [specs[0]], lambda _prompt: {"framings": [["don't", "instead"]]}
    )
    assert not verbalized
    assert rejected == {specs[0]["candidate_id"]: "verbalizer_missing_item"}


def test_framing_discards_model_generated_action_words() -> None:
    spec = next(
        spec
        for spec in build_specs(100, seed=33)
        if request_seed(spec).startswith("open ")
    )
    verbalized, rejected = verbalize_batch(
        [spec], lambda _prompt: {"framings": [["Could you please open", "?"]]}
    )
    assert not rejected
    assert verbalized[0]["utterance"].casefold().count("open") == 1
    assert verbalized[0]["utterance"].startswith("Could you please open ")


def test_singleton_verbalizer_salvages_extra_model_fragments_safely() -> None:
    spec = next(spec for spec in build_specs(100, seed=36) if spec["excluded_names"])
    verbalized, rejected = verbalize_batch(
        [spec],
        lambda _prompt: {
            "framings": [
                ["Could you please turn on the", "and", "but leave the", "alone?"],
                ["", ""],
            ]
        },
    )
    assert not rejected
    assert verbalized[0]["utterance"] == f"Could you please {request_seed(spec)}?"


def test_verbalizer_splits_malformed_batches_without_relabeling() -> None:
    specs = build_specs(100, seed=34)[:4]

    def complete(prompt: str) -> dict:
        items = _payload(prompt)
        if len(items) > 1:
            return {"framings": []}
        return {"framings": [["Please", "."]]}

    verbalized = verbalize_resilient(specs, complete, attempts=1)
    assert [spec["candidate_id"] for spec in verbalized] == [spec["candidate_id"] for spec in specs]
    assert all(
        request_seed(original).casefold() in row["utterance"].casefold()
        for original, row in zip(specs, verbalized)
    )


def test_verbalizer_retries_a_transiently_malformed_singleton() -> None:
    spec = build_specs(100, seed=35)[0]
    calls = 0
    prompts: list[str] = []

    def complete(prompt: str) -> dict:
        nonlocal calls
        calls += 1
        prompts.append(prompt)
        return {"framings": [["Please", "."]]} if calls == 5 else {"framings": []}

    assert verbalize_resilient([spec], complete)[0]["candidate_id"] == spec["candidate_id"]
    assert calls == 5
    assert len(set(prompts)) == 5


def test_verbalizer_retries_completion_exceptions() -> None:
    spec = next(spec for spec in build_specs(100, seed=37) if spec["expected"]["kind"] == "action")
    calls = 0

    def complete(_prompt: str) -> dict:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RuntimeError("transient parse failure")
        return {"framings": [["Please", "."]]}

    assert verbalize_resilient([spec], complete)[0]["candidate_id"] == spec["candidate_id"]
    assert calls == 3


def test_verbalizer_falls_back_to_seed_framing_when_singleton_exhausted() -> None:
    spec = build_specs(100, seed=38)[0]

    verbalized = verbalize_resilient([spec], lambda _prompt: {"framings": []}, attempts=1)
    assert verbalized[0]["candidate_id"] == spec["candidate_id"]
    assert verbalized[0]["expected"] == spec["expected"]
    assert verbalized[0]["utterance"] == _framed_utterance(spec, ["", ""])
    assert request_seed(spec).casefold() in verbalized[0]["utterance"].casefold()


def test_verbalizer_fails_closed_on_missing_or_extra_ids() -> None:
    specs = build_specs(100, seed=31)[:2]
    verbalized, rejected = verbalize_batch(
        specs,
        lambda _prompt: {"items": [{"candidate_id": "invented", "utterance": "turn it on"}]},
    )
    assert not verbalized
    assert rejected == {spec["candidate_id"]: "verbalizer_missing_item" for spec in specs}


def test_compact_verbalizer_response_preserves_input_order() -> None:
    specs = build_specs(100, seed=32)[:3]
    utterances = [f"spoken row {index} {template_seed(spec)}" for index, spec in enumerate(specs)]
    verbalized, rejected = verbalize_batch(specs, lambda _prompt: {"utterances": utterances})
    assert not rejected
    assert [spec["utterance"] for spec in verbalized] == [
        f"spoken row {index} {request_seed(spec)}" for index, spec in enumerate(specs)
    ]

    verbalized, rejected = verbalize_batch(specs, lambda _prompt: {"utterances": utterances[:2]})
    assert not verbalized
    assert set(rejected) == {spec["candidate_id"] for spec in specs}


def _matching_utterance(spec: dict) -> str:
    return expand_utterance(spec)


def test_deterministic_language_gate_accepts_matching_hard_cases() -> None:
    for spec in build_specs(300, seed=81):
        spec["utterance"] = _matching_utterance(spec)
        assert validate_utterance(spec) is None, spec


def test_deterministic_language_gate_rejects_target_and_distinction_errors() -> None:
    specs = build_specs(300, seed=82)
    action = next(spec for spec in specs if spec["expected"]["kind"] == "action")
    action["utterance"] = "turn on an invented device"
    assert validate_utterance(action) == "missing_expected_target"

    exclusion = next(spec for spec in specs if spec["excluded_names"])
    exclusion["utterance"] = _matching_utterance(exclusion).split(" but leave", 1)[0]
    assert validate_utterance(exclusion) == "missing_exclusion"

    status = next(spec for spec in specs if spec["expected"]["kind"] == "status")
    status["utterance"] = f"turn on {status['target_names'][0]}"
    assert validate_utterance(status) == "status_not_query"


def test_validate_utterance_rejects_please_what_framing() -> None:
    status = next(spec for spec in build_specs(100, seed=82) if spec["expected"]["kind"] == "status")
    status["utterance"] = f"Could you please what is the status of {status['target_names'][0]}"
    assert validate_utterance(status) == "unnatural_framing"
    status["utterance"] = f"Okay, can you what is the status of {status['target_names'][0]}"
    assert validate_utterance(status) == "unnatural_framing"
    assert _framed_utterance(status, ["Could you please", ""]) is None


def test_judge_prompt_requires_full_score_range_and_naturalness_rubric() -> None:
    spec = next(spec for spec in build_specs(100, seed=82) if spec["expected"]["kind"] == "status")
    spec["utterance"] = f"Could you please {request_seed(spec)}"
    prompt = _judge_prompt([spec])
    assert "Do not default to 555" in prompt
    assert "Could you please what is the status" in prompt
    assert "exact_call_count" in prompt
    assert '"scores":["543"]' in prompt
    assert '"scores":["555"]' not in prompt


def test_independent_judge_applies_fixed_quality_thresholds() -> None:
    specs = build_specs(100, seed=88)[:2]
    for spec in specs:
        spec["utterance"] = _matching_utterance(spec)

    def complete(prompt: str) -> dict:
        assert "exact_call_count" in prompt
        assert _payload(prompt)[0]["expected"] == specs[0]["expected"]
        return {
            "items": [
                {
                    "candidate_id": specs[0]["candidate_id"],
                    "accept": True,
                    "correctness": 5,
                    "clarity": 5,
                    "naturalness": 4,
                    "difficulty": 4,
                    "semantic_key": "turn-on-canonical-target",
                },
                {
                    "candidate_id": specs[1]["candidate_id"],
                    "accept": True,
                    "correctness": 5,
                    "clarity": 5,
                    "naturalness": 3,
                    "difficulty": 5,
                    "semantic_key": "awkward-request",
                },
            ]
        }

    accepted, rejected = judge_batch(
        specs,
        complete,
        generator_model="verbalizer-a",
        judge_model="judge-b",
    )
    assert [spec["candidate_id"] for spec in accepted] == [specs[0]["candidate_id"]]
    assert accepted[0]["quality"]["naturalness"] == 4
    assert rejected == {specs[1]["candidate_id"]: "judge_below_threshold"}


def test_judge_must_be_independent_and_fails_closed() -> None:
    spec = build_specs(100, seed=89)[0]
    spec["utterance"] = _matching_utterance(spec)
    try:
        judge_batch([spec], lambda _prompt: {}, generator_model="same", judge_model="same")
    except ValueError as error:
        assert str(error) == "judge model must differ from generator model"
    else:
        raise AssertionError("same-model judge was accepted")

    accepted, rejected = judge_batch(
        [spec], lambda _prompt: {"items": []}, generator_model="one", judge_model="two"
    )
    assert not accepted
    assert rejected == {spec["candidate_id"]: "judge_missing_item"}


def test_compact_judge_scores_preserve_order_and_fixed_floor() -> None:
    specs = build_specs(100, seed=90)[:3]
    for spec in specs:
        spec["utterance"] = _matching_utterance(spec)
    accepted, rejected = judge_batch(
        specs,
        lambda _prompt: {"scores": ["555", "445", "553"]},
        generator_model="one",
        judge_model="two",
    )
    assert [spec["candidate_id"] for spec in accepted] == [specs[0]["candidate_id"], specs[1]["candidate_id"]]
    assert accepted[0]["quality"]["semantic_key"] == request_seed(specs[0])
    assert rejected == {specs[2]["candidate_id"]: "judge_below_threshold"}


def test_judge_accepts_compact_score_objects() -> None:
    specs = build_specs(100, seed=102)[:2]
    for spec in specs:
        spec["utterance"] = _matching_utterance(spec)
    accepted, rejected = judge_batch(
        specs,
        lambda _prompt: {
            "scores": [
                {"correctness": 5, "clarity": 4, "naturalness": 5},
                {"correctness": 5, "clarity": 5, "naturalness": 3},
            ]
        },
        generator_model="one",
        judge_model="two",
    )
    assert [spec["candidate_id"] for spec in accepted] == [specs[0]["candidate_id"]]
    assert rejected == {specs[1]["candidate_id"]: "judge_below_threshold"}


def test_judge_accepts_compact_integer_triples() -> None:
    specs = build_specs(100, seed=108)[:2]
    for spec in specs:
        spec["utterance"] = _matching_utterance(spec)
    accepted, rejected = judge_batch(
        specs,
        lambda _prompt: {"scores": [[5, 4, 5], [5, 5, 3]]},
        generator_model="one",
        judge_model="two",
    )
    assert [spec["candidate_id"] for spec in accepted] == [specs[0]["candidate_id"]]
    assert rejected == {specs[1]["candidate_id"]: "judge_below_threshold"}


def test_judge_accepts_singleton_score_object_without_list_wrapper() -> None:
    spec = build_specs(100, seed=104)[0]
    spec["utterance"] = _matching_utterance(spec)
    accepted, rejected = judge_batch(
        [spec],
        lambda _prompt: {"scores": {"correctness": 5, "clarity": 5, "naturalness": 4}},
        generator_model="one",
        judge_model="two",
    )
    assert [row["candidate_id"] for row in accepted] == [spec["candidate_id"]]
    assert not rejected


def test_judge_splits_malformed_batches_but_keeps_real_rejections() -> None:
    specs = build_specs(100, seed=103)[:3]
    for spec in specs:
        spec["utterance"] = _matching_utterance(spec)

    def complete(prompt: str) -> dict:
        items = _payload(prompt)
        if len(items) > 1:
            return {"scores": []}
        score = 3 if items[0]["candidate_id"] == specs[-1]["candidate_id"] else 5
        return {"scores": [{"correctness": 5, "clarity": 5, "naturalness": score}]}

    accepted, rejected = judge_resilient(
        specs,
        complete,
        generator_model="one",
        judge_model="two",
        attempts=1,
    )
    assert [spec["candidate_id"] for spec in accepted] == [
        specs[0]["candidate_id"],
        specs[1]["candidate_id"],
    ]
    assert rejected == {specs[2]["candidate_id"]: "judge_below_threshold"}


def test_judge_retries_completion_exceptions() -> None:
    spec = build_specs(100, seed=106)[0]
    spec["utterance"] = _matching_utterance(spec)
    calls = 0

    def complete(_prompt: str) -> dict:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RuntimeError("transient parse failure")
        return {"scores": {"correctness": 5, "clarity": 5, "naturalness": 5}}

    accepted, rejected = judge_resilient(
        [spec], complete, generator_model="one", judge_model="two"
    )
    assert [row["candidate_id"] for row in accepted] == [spec["candidate_id"]]
    assert not rejected
    assert calls == 3


def _judged(spec: dict, *, key: str, score: int = 5) -> dict:
    spec["utterance"] = _matching_utterance(spec)
    spec["quality"] = {
        "correctness": score,
        "clarity": score,
        "naturalness": score,
        "difficulty": score,
        "semantic_key": key,
    }
    return spec


def test_curation_removes_exact_and_semantic_duplicates_by_behavior() -> None:
    specs = build_specs(100, seed=92)
    first = _judged(specs[0], key="same meaning")
    exact = json.loads(json.dumps(first))
    exact["candidate_id"] = "exact_copy"
    semantic = json.loads(json.dumps(first))
    semantic["candidate_id"] = "semantic_copy"
    semantic["utterance"] = "a different paraphrase"
    distinct = _judged(specs[1], key="different meaning")

    selected, drops = curate([first, exact, semantic, distinct], min_count=2, max_count=3)
    assert {spec["candidate_id"] for spec in selected} == {first["candidate_id"], distinct["candidate_id"]}
    assert drops["exact_duplicate"] == 1
    assert drops["semantic_duplicate"] == 1


def test_curation_excludes_heldout_user_prompts() -> None:
    specs = build_specs(100, seed=99)[:2]
    first = _judged(specs[0], key="first")
    second = _judged(specs[1], key="second")

    selected, drops = curate(
        [first, second],
        min_count=1,
        max_count=2,
        excluded_utterances={f"  {first['utterance'].upper()}  "},
    )
    assert [spec["candidate_id"] for spec in selected] == [second["candidate_id"]]
    assert drops["heldout_overlap"] == 1


def test_load_user_utterances_reads_canonical_training_rows(tmp_path: Path) -> None:
    path = tmp_path / "heldout.jsonl"
    path.write_text(
        json.dumps({"messages": [{"role": "user", "content": "Is Joe's fan on?"}]})
        + "\n"
        + json.dumps(
            {"messages": [{"role": "user", "content": [{"type": "text", "text": "Open it."}]}]}
        )
        + "\n"
    )
    assert load_user_utterances(path) == {"Is Joe's fan on?", "Open it."}


def test_curation_keeps_contrastive_behavior_and_never_lowers_floor() -> None:
    specs = build_specs(300, seed=93)
    pair = [spec for spec in specs if spec["contrastive_group"]][:2]
    assert len(pair) == 2
    for spec in pair:
        _judged(spec, key="tiny wording contrast")
        spec["utterance"] = "turn on the light"
    selected, _drops = curate(pair, min_count=2, max_count=2)
    assert len(selected) == 2

    try:
        curate(pair[:1], min_count=2, max_count=3)
    except ValueError as error:
        assert "quality floor" in str(error)
    else:
        raise AssertionError("curation lowered its minimum")


def test_ranked_curation_respects_requested_category_mix() -> None:
    specs = [_judged(spec, key=spec["candidate_id"]) for spec in build_specs(200, seed=94)]
    selected, _drops = curate(specs, min_count=100, max_count=100)
    assert Counter(spec["category"] for spec in selected) == CATEGORY_WEIGHTS
    assert all("rank_score" in spec["quality"] for spec in selected)


def test_curation_fails_instead_of_publishing_without_hard_category_coverage() -> None:
    specs = [
        _judged(spec, key=spec["candidate_id"])
        for spec in build_specs(200, seed=96)
        if spec["category"] != "ambiguity"
    ]
    try:
        curate(specs, min_count=100, max_count=120)
    except ValueError as error:
        assert "coverage floor" in str(error)
    else:
        raise AssertionError("curation published without ambiguity coverage")


def test_openai_transport_parses_json_without_new_dependency() -> None:
    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"choices": [{"message": {"content": '```json\n{"items": []}\n```'}}]}

    def post(url: str, **kwargs: object) -> Response:
        assert url == "http://llm/v1/chat/completions"
        assert kwargs["headers"] == {"Authorization": "Bearer secret"}
        return Response()

    assert openai_complete(
        "prompt", base_url="http://llm/v1", model="model-a", api_key="secret", post=post
    ) == {"items": []}


def test_openai_transport_ignores_model_explanation_after_first_json() -> None:
    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {
                "choices": [
                    {"message": {"content": '{"scores":["555"]}\nExplanation with {"extra":true}.'}}
                ]
            }

    assert openai_complete(
        "prompt", base_url="http://llm/v1", model="judge", post=lambda *_args, **_kwargs: Response()
    ) == {"scores": ["555"]}


def test_framing_response_schema_locks_count_and_safe_language() -> None:
    response_format = framing_response_format(64)
    framings = response_format["json_schema"]["schema"]["properties"]["framings"]
    assert response_format["type"] == "json_schema"
    assert framings["minItems"] == framings["maxItems"] == 64
    assert framings["items"]["prefixItems"][0]["enum"]
    assert framings["items"]["prefixItems"][1]["enum"]


def test_judge_response_schema_locks_count_and_three_scores() -> None:
    response_format = judge_response_format(16)
    scores = response_format["json_schema"]["schema"]["properties"]["scores"]
    assert scores["minItems"] == scores["maxItems"] == 16
    assert scores["items"]["pattern"] == "^[1-5]{3}$"


def _payload(prompt: str) -> list[dict]:
    return json.loads(prompt.split("ITEMS:\n", 1)[1])


def _utterance_from_payload(item: dict) -> str:
    return item["template_seed"]


def test_resumable_end_to_end_pipeline_writes_audited_outputs(tmp_path: Path) -> None:
    calls = Counter()

    def generate(prompt: str) -> dict:
        calls["generate"] += 1
        return {
            "items": [
                {"candidate_id": item["candidate_id"], "utterance": _utterance_from_payload(item)}
                for item in _payload(prompt)
            ]
        }

    def judge(prompt: str) -> dict:
        calls["judge"] += 1
        return {"scores": ["555" for _item in _payload(prompt)]}

    report = run_pipeline(
        out_dir=tmp_path,
        count=100,
        seed=101,
        batch_size=20,
        generator_complete=generate,
        judge_complete=judge,
        generator_model="generator-a",
        judge_model="judge-b",
        min_count=40,
        max_count=60,
    )
    assert report["candidate_count"] == 100
    assert report["curated_count"] == 60
    assert report["audit"]["deterministically_valid_rows"] == 60
    assert report["audit"]["excluded_entity_call_violations"] == 0
    assert report["audit"]["exact_duplicate_keys"] == 0
    assert report["audit"]["semantic_duplicate_keys"] == 0
    assert sum(1 for _line in (tmp_path / "sayso_candidates_100.jsonl").open()) == 100
    curated = [json.loads(line) for line in (tmp_path / "sayso_curated_60.jsonl").read_text().splitlines()]
    assert len(curated) == 60
    assert all(example["metadata"]["quality"]["correctness"] == 5 for example in curated)

    before = calls.copy()
    assert run_pipeline(
        out_dir=tmp_path,
        count=100,
        seed=101,
        batch_size=20,
        generator_complete=generate,
        judge_complete=judge,
        generator_model="generator-a",
        judge_model="judge-b",
        min_count=40,
        max_count=60,
    ) == report
    assert calls == before


def test_pipeline_parallelizes_inference_but_keeps_one_checkpoint_writer(tmp_path: Path) -> None:
    barrier = threading.Barrier(2)

    def generate(prompt: str) -> dict:
        barrier.wait(timeout=2)
        return {
            "items": [
                {"candidate_id": item["candidate_id"], "utterance": item["template_seed"]}
                for item in _payload(prompt)
            ]
        }

    report = run_pipeline(
        out_dir=tmp_path,
        count=100,
        seed=105,
        batch_size=50,
        workers=2,
        generator_complete=generate,
        judge_complete=lambda prompt: {"scores": ["555" for _item in _payload(prompt)]},
        generator_model="generator-a",
        judge_model="judge-b",
        min_count=40,
        max_count=60,
    )
    assert report["curated_count"] == 60
    rows = [json.loads(line) for line in (tmp_path / "sayso_candidates_100.jsonl").read_text().splitlines()]
    assert len(rows) == len({row["candidate_id"] for row in rows}) == 100


def test_candidate_checkpoint_can_resume_with_a_distinct_generator(tmp_path: Path) -> None:
    def generate(prompt: str) -> dict:
        return {
            "items": [
                {"candidate_id": item["candidate_id"], "utterance": item["template_seed"]}
                for item in _payload(prompt)
            ]
        }

    run_pipeline(
        out_dir=tmp_path,
        count=100,
        seed=107,
        batch_size=50,
        generator_complete=generate,
        generator_model="generator-a",
        judge_model="judge",
        stage="generate",
    )
    path = tmp_path / "sayso_candidates_100.jsonl"
    lines = path.read_text().splitlines()[:50]
    path.write_text("\n".join(lines) + "\n")
    report = run_pipeline(
        out_dir=tmp_path,
        count=100,
        seed=107,
        batch_size=50,
        generator_complete=generate,
        generator_model="generator-b",
        judge_model="judge",
        stage="generate",
    )
    assert report["candidate_count"] == 100
    models = {json.loads(line)["generator_model"] for line in path.read_text().splitlines()}
    assert models == {"generator-a", "generator-b"}


def test_synthetic_v2_trl_config_reuses_the_proven_1070_lora_recipe() -> None:
    text = (ROOT / "configs" / "lfm25-230m-synthetic-v2-trl.yml").read_text()
    assert "model_name_or_path: /srv/models/LFM2.5-230M-Base" in text
    assert "data_files: /srv/datasets/sayso_v2/sayso_train_first_10000_render.jsonl" in text
    assert "output_dir: /srv/training-runs/SaySo-LFM2.5-230M-Base-First" in text
    assert "use_peft: true" in text
    assert "use_rslora: true" in text
    assert "lora_target_modules:\n  - all-linear" in text
    assert "assistant_only_loss: true" in text
    assert "num_train_epochs: 3" in text


def _specs_for(category: str, *, count: int = 200, seed: int = 42) -> list[dict]:
    return [spec for spec in build_specs(count, seed=seed) if spec["category"] == category]


def test_recipe_clean_direct_labels_one_device_one_tool() -> None:
    specs = _specs_for("clean_direct")
    for spec in specs:
        assert len(spec["expected"]["calls"]) == 1
        assert spec["expected"]["calls"][0]["name"] in {
            "HassTurnOn",
            "HassTurnOff",
            "HassLightSet",
            "HassFanSetSpeed",
        }
        utterance = expand_utterance(spec)
        assert _CLEAN_DIRECT_START.match(utterance.strip())
        spec["utterance"] = utterance
        assert validate_utterance(spec) is None


def test_recipe_conversational_uses_natural_voice_not_bare_commands() -> None:
    specs = _specs_for("conversational")
    for spec in specs:
        utterance = expand_utterance(spec)
        assert not _CLEAN_DIRECT_START.match(utterance.strip())
        spec["utterance"] = utterance
        assert validate_utterance(spec) is None


def test_recipe_entity_identity_preserves_apostrophe_canonical_names() -> None:
    apostrophe = [spec for spec in _specs_for("entity_identity") if spec["subcategory"] == "apostrophe"]
    assert apostrophe
    for spec in apostrophe:
        name = spec["expected"]["calls"][0]["arguments"]["name"]
        assert "'" in name
        assert "apostrophe" not in name.casefold()
        spec["utterance"] = expand_utterance(spec)
        assert validate_utterance(spec) is None


def test_recipe_multi_action_exclusion_never_calls_excluded_device() -> None:
    specs = _specs_for("multi_action_exclusion")
    for spec in specs:
        assert spec["excluded_names"]
        called = {call["arguments"].get("name") for call in spec["expected"]["calls"]}
        assert not called.intersection(spec["excluded_names"])
        spec["utterance"] = expand_utterance(spec)
        assert validate_utterance(spec) is None


def test_recipe_stt_corrupted_labels_name_canonical_entity() -> None:
    specs = _specs_for("stt_corrupted")
    for spec in specs:
        canonical = spec["target_names"][0]
        spoken = spec["spoken_targets"][canonical]
        assert spoken.casefold() != canonical.casefold()
        assert spec["expected"]["calls"][0]["arguments"]["name"] == canonical
        spec["utterance"] = expand_utterance(spec)
        assert validate_utterance(spec) is None


def test_recipe_status_uses_get_live_context_never_turn_on() -> None:
    specs = _specs_for("status")
    for spec in specs:
        calls = spec["expected"]["calls"]
        assert calls
        assert all(call["name"] == "GetLiveContext" for call in calls)
        assert not any(call["name"] == "HassTurnOn" for call in calls)
        spec["utterance"] = expand_utterance(spec)
        assert validate_utterance(spec) is None


def test_recipe_ambiguity_defaults_to_sayso_entity_area() -> None:
    specs = _specs_for("ambiguity")
    assert specs
    for spec in specs:
        assert spec["home"]["sayso_entity_area"]
        system = render_example({**spec, "utterance": expand_utterance(spec)})["messages"][0]["content"]
        assert spec["home"]["sayso_entity_area"] in system
    zero = next(
        spec
        for spec in specs
        if spec["subcategory"] == "zero_lights"
        and spec["expected"].get("response") == "area_unavailable"
    )
    rendered = render_example({**zero, "utterance": expand_utterance(zero)})
    assert not any(message.get("tool_calls") for message in rendered["messages"])
    assert "has no lights available" in rendered["messages"][-1]["content"].casefold()


def test_generic_no_area_ambiguity_labels_follow_sayso_entity_area() -> None:
    """Regression: seed=42 contrastive rows must not house-wide-clarify (recipe 7)."""
    for spec in build_specs(200, seed=42):
        if spec["category"] != "ambiguity":
            continue
        hint = spec.get("request_hint", "")
        if not _is_generic_no_area_hint(hint):
            continue
        generic = hint.rsplit("the ", 1)[-1]
        kind = {
            "light": "light",
            "fan": "fan",
            "outlet": "switch",
            "blinds": "blinds",
            "garage door": "garage_door",
            "door": "lock",
        }[generic]
        sayso_area = spec["home"]["sayso_entity_area"]
        in_area = [
            entity
            for entity in spec["home"]["entities"]
            if entity["kind"] == kind and entity["area"] == sayso_area
        ]
        expected = spec["expected"]
        if len(in_area) == 0:
            assert expected["response"] == "area_unavailable", (
                f"{spec['subcategory']} {hint!r} in {sayso_area!r} must be area_unavailable"
            )
            assert not expected.get("calls")
        elif len(in_area) == 1:
            assert expected["kind"] == "action", (
                f"{spec['subcategory']} {hint!r} with one {kind} in {sayso_area!r}"
            )
            assert expected["calls"][0]["arguments"]["name"] == in_area[0]["name"]
        else:
            assert expected["response"] == "clarify", (
                f"{spec['subcategory']} {hint!r} with {len(in_area)} {kind}s in {sayso_area!r}"
            )
            assert not expected.get("calls")


def test_recipe_unsupported_omits_thermostat_keeps_refuse_clarify_media() -> None:
    specs = _specs_for("unsupported_no_action")
    hints = {spec["request_hint"].casefold() for spec in specs}
    assert "set the thermostat to 72 degrees" not in hints
    assert any("smoke alarm" in hint for hint in hints)
    assert any(hint.endswith("set the light to") for hint in hints)
    assert any("play music" in hint for hint in hints)
    for spec in specs:
        spec["utterance"] = expand_utterance(spec)
        assert validate_utterance(spec) is None


def test_system_context_includes_sayso_conversation_entity_area() -> None:
    spec = build_specs(100, seed=11)[0]
    spec["utterance"] = expand_utterance(spec)
    system = render_example(spec)["messages"][0]["content"]
    assert "This SaySo conversation entity area is" in system
    assert spec["home"]["sayso_entity_area"] in system


def test_banned_please_what_status_and_chatml_labels_rejected() -> None:
    status = next(spec for spec in build_specs(100, seed=12) if spec["category"] == "status")
    status["utterance"] = f"Could you please what is the status of {status['target_names'][0]}"
    assert validate_utterance(status) == "unnatural_framing"
    bad = dict(status)
    bad["utterance"] = "evals/cases/foo turn on light"
    assert validate_utterance(bad) == "banned_content"
    bad["utterance"] = "<tool_call>{\"name\":\"HassTurnOn\"}</tool_call> turn on the kitchen light please"
    assert validate_utterance(bad) == "banned_content"
