"""Tests for scripts/wake_prepare_real_data.py."""

from __future__ import annotations

import json
import struct
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import wake_prepare_real_data as prepare  # noqa: E402

import pytest  # noqa: E402

SAMPLE_RATE = 16000
WINDOW_SAMPLES = SAMPLE_RATE * 2


def _wav(path: Path, amplitude: int = 1000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = struct.pack("<" + "h" * WINDOW_SAMPLES, *([amplitude] * WINDOW_SAMPLES))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples)


def _cluster(
    index: int,
    detail: str,
    label: str,
    score: float,
    stems: list[str],
    transcripts: list[str],
) -> dict:
    return {
        "index": index,
        "label": label,
        "detail": detail,
        "best_score": score,
        "best_stem": stems[0],
        "stems": stems,
        "transcripts": transcripts,
        "review": "suggested",
    }


def _clusters() -> list[dict]:
    return [
        _cluster(0, "positive_isolated", "positive", 0.74, ["p0a", "p0b"], ["Say so."]),
        _cluster(1, "positive_isolated", "positive", 0.52, ["p1a"], ["Say so turn on the TV"]),
        _cluster(2, "negative_preceded", "negative", 0.25, ["n2a", "n2b"], ["assistant tools say so"]),
        _cluster(3, "negative_no_phrase", "negative", 0.32, ["n3a"], ["Good boy."]),
        _cluster(4, "negative_no_phrase", "negative", 0.29, ["n4a"], ["New players available."]),
        _cluster(5, "negative_no_phrase_near_miss", "negative", 0.65, ["n5a"], ["Stay soon."]),
    ]


def test_select_eval_stems_fills_every_case_with_a_distinct_window() -> None:
    chosen, gaps = prepare.select_eval_stems(_clusters())
    assert set(chosen) == {case for case, _, _ in prepare.EVAL_CASE_PLAN}

    # One carrier cluster has to cover both carrier cases, on two different windows.
    assert chosen["neg_say_so_carrier_if_you"][1] == "n2a"
    assert chosen["neg_say_so_carrier_leading_word"][1] == "n2b"

    # continuous_command must prefer an utterance with words after the phrase.
    assert chosen["pos_continuous_kitchen"][0]["index"] == 1

    # Distinct utterances wherever the spool can supply them: the two ambient
    # negatives must not be two windows of the same utterance.
    assert chosen["neg_tv_conversation"][0]["index"] != chosen["neg_distance_noise"][0]["index"]

    # Unfillable cases are reported, not silently dropped.
    assert {case for case, _ in gaps} == set(prepare.EVAL_CASE_GAPS)


def test_select_eval_stems_reports_a_gap_when_a_class_is_absent() -> None:
    only_positives = [c for c in _clusters() if c["detail"] == "positive_isolated"]
    chosen, gaps = prepare.select_eval_stems(only_positives)
    assert set(chosen) == {"pos_say_so_two_word", "pos_continuous_kitchen"}
    assert "neg_tv_conversation" in {case for case, _ in gaps}


def test_build_eval_corpus_copies_audio_writes_fixture_and_records_held_out(tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    spool.mkdir()
    for cluster in _clusters():
        for stem in cluster["stems"]:
            _wav(spool / f"{stem}.wav")

    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    (eval_root / "cases.json").write_text(
        json.dumps(
            {
                "version": 1,
                "cases": [
                    {
                        "id": "pos_say_so_two_word",
                        "category": "positive_say_so_two_word",
                        "audio": "audio/positive_say_so_two_word.wav",
                        "expect_detection": True,
                    },
                    {
                        "id": "pos_continuous_kitchen",
                        "category": "continuous_command",
                        "audio": "audio/continuous_kitchen.wav",
                        "expect_detection": True,
                        "command_transcript": "turn off the kitchen",
                        "transcript_fixture": "fixtures/stt/continuous_kitchen.json",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    held_out = tmp_path / "held-out.json"

    result = prepare.build_eval_corpus(spool, _clusters(), eval_root, held_out)

    assert (eval_root / "audio" / "positive_say_so_two_word.wav").is_file()
    assert (eval_root / "audio" / "continuous_kitchen.wav").is_file()
    fixture = json.loads((eval_root / "fixtures/stt/continuous_kitchen.json").read_text())
    assert fixture["text"] == "turn off the kitchen"
    assert fixture["stub"] is True

    payload = json.loads(held_out.read_text())
    # Every window of every cluster the corpus draws on is held out, not just the
    # one window that became a fixture.
    assert {"p0a", "p0b", "p1a"} <= set(payload["stems"])
    assert len(result["populated"]) == 2


def test_build_eval_corpus_refuses_a_root_without_cases(tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    spool.mkdir()
    with pytest.raises(ValueError, match="no cases.json"):
        prepare.build_eval_corpus(spool, _clusters(), tmp_path / "empty", None)


def test_build_training_data_seeds_originals_and_respects_held_out(tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    spool.mkdir()
    for cluster in _clusters():
        for stem in cluster["stems"]:
            _wav(spool / f"{stem}.wav")

    held_out = tmp_path / "held-out.json"
    held_out.write_text(json.dumps({"stems": ["p0a", "p0b"]}), encoding="utf-8")
    model_dir = tmp_path / "output" / "sayso"

    manifest = prepare.build_training_data(
        spool,
        _clusters(),
        model_dir,
        held_out,
        repeat=3,
        negative_repeat=2,
        index_base=900000,
        dry_run=False,
    )

    positives = sorted((model_dir / "positive_train").glob("*.wav"))
    negatives = sorted((model_dir / "negative_train").glob("*.wav"))

    # The held-out cluster (index 0) is seeded nowhere, not even into a test split
    # which would select the operating threshold.
    assert manifest["held_out_clusters"] == 1
    assert [entry["cluster_index"] for entry in manifest["positive_utterances"]] == [1]
    assert [entry["cluster_index"] for entry in manifest["negative_utterances"]] == [2, 3, 4, 5]

    assert len(positives) == 3
    assert len(negatives) == 8
    # Plain original naming is load bearing: augment's startup cleanup deletes
    # every clip_dddddd_rN.wav, so an _r0-named seed never survives to training.
    assert positives[0].name == "clip_900000.wav"
    assert [p.name for p in positives] == ["clip_900000.wav", "clip_900001.wav", "clip_900002.wav"]
    assert negatives[0].name == "clip_900003.wav"

    written_manifest = json.loads((model_dir / "real_clip_manifest.json").read_text())
    assert written_manifest["repeat"] == {"positive": 3, "negative": 2}
    assert written_manifest["held_out_clusters"] == 1


def test_seed_split_names_survive_augments_startup_cleanup(tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    spool.mkdir()
    _wav(spool / "p0a.wav")
    out = tmp_path / "positive_train"
    files, index = prepare.seed_split(["p0a"], spool, out, repeat=1, index=900000, dry_run=False)

    assert files == 1 and index == 900001
    # The exact pattern run_augment deletes at startup: clip_dddddd_rN.wav.
    # A seed matching it is erased before it is ever trained on, so it must not.
    import re as _re
    cleanup = _re.compile(r"^clip_\d{6}_r\d+\.wav$")
    seeded = out / "clip_900000.wav"
    assert seeded.is_file() and not cleanup.match(seeded.name)
    with wave.open(str(seeded), "rb") as wf:
        assert wf.getnframes() == WINDOW_SAMPLES


def test_main_dry_run_writes_nothing(tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    spool.mkdir()
    _wav(spool / "p0a.wav")
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps({"version": 1, "clusters": [_clusters()[0]]}), encoding="utf-8"
    )
    model_dir = tmp_path / "output" / "sayso"

    rc = prepare.main(
        [
            "training-data",
            "--spool",
            str(spool),
            "--ledger",
            str(ledger),
            "--model-dir",
            str(model_dir),
            "--dry-run",
        ]
    )
    assert rc == 0
    assert not model_dir.exists()


def test_main_refuses_to_seed_twice(tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    spool.mkdir()
    _wav(spool / "p0a.wav")
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps({"version": 1, "clusters": [_clusters()[0]]}), encoding="utf-8"
    )
    model_dir = tmp_path / "output" / "sayso"
    argv = [
        "training-data",
        "--spool",
        str(spool),
        "--ledger",
        str(ledger),
        "--model-dir",
        str(model_dir),
    ]
    assert prepare.main(argv) == 0
    assert prepare.main(argv) == 1


def test_main_rejects_a_ledger_without_clusters(tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    spool.mkdir()
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"version": 1, "clusters": []}), encoding="utf-8")
    assert prepare.main(["training-data", "--spool", str(spool), "--ledger", str(ledger)]) == 2


def test_all_windows_seeds_every_window_of_each_cluster(tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    spool.mkdir()
    for cluster in _clusters():
        for stem in cluster["stems"]:
            _wav(spool / f"{stem}.wav")

    manifest = prepare.build_training_data(
        spool,
        _clusters(),
        tmp_path / "output" / "sayso",
        None,
        repeat=1,
        negative_repeat=1,
        index_base=900000,
        dry_run=False,
        all_windows=True,
    )
    # 3 positive windows and 5 negative windows, one clip each.
    assert len(manifest["positive_utterances"]) == 3
    assert len(manifest["negative_utterances"]) == 5
    assert manifest["all_windows"] is True


def test_clear_seeded_leaves_generated_clips_alone(tmp_path: Path) -> None:
    model_dir = tmp_path / "output" / "sayso"
    (model_dir / "positive_train").mkdir(parents=True)
    generated = model_dir / "positive_train" / "clip_000123.wav"
    generated_round = model_dir / "positive_train" / "clip_000123_r0.wav"
    seeded = model_dir / "positive_train" / "clip_900001.wav"
    seeded_round = model_dir / "positive_train" / "clip_900001_r1.wav"
    for path in (generated, generated_round, seeded, seeded_round):
        path.write_bytes(b"")
    (model_dir / "real_clip_manifest.json").write_text("{}", encoding="utf-8")

    removed = prepare.clear_seeded(model_dir, 900000)

    # Both the seeded original and the augmentation round derived from it go;
    # generated TTS clips below the index base are untouched either way.
    assert removed == 2
    assert generated.exists() and generated_round.exists()
    assert not seeded.exists() and not seeded_round.exists()
    assert not (model_dir / "real_clip_manifest.json").exists()


def test_copy_live_backgrounds_skips_carriers_and_held_out_clusters(tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    spool.mkdir()
    for cluster in _clusters():
        for stem in cluster["stems"]:
            _wav(spool / f"{stem}.wav")

    out = tmp_path / "data" / "live-backgrounds"
    copied = prepare.copy_live_backgrounds(
        spool, _clusters(), out, {"n3a"}, dry_run=False
    )

    names = sorted(p.name for p in out.glob("*.wav"))
    # Phrase-free n3a (held out) and n4a, plus the near-miss n5a; the carrier
    # cluster (n2a/n2b) and every positive are excluded.
    assert names == ["bg_n4a.wav", "bg_n5a.wav"]
    assert copied == 2


def test_main_replace_reseeds_without_touching_generated_clips(tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    spool.mkdir()
    _wav(spool / "p0a.wav")
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"version": 1, "clusters": [_clusters()[0]]}), encoding="utf-8")
    model_dir = tmp_path / "output" / "sayso"
    argv = [
        "training-data",
        "--spool",
        str(spool),
        "--ledger",
        str(ledger),
        "--model-dir",
        str(model_dir),
        "--repeat",
        "2",
    ]
    assert prepare.main(argv) == 0
    assert len(list((model_dir / "positive_train").glob("clip_9*.wav"))) == 2

    assert prepare.main(argv + ["--replace"]) == 0
    assert len(list((model_dir / "positive_train").glob("clip_9*.wav"))) == 2


def test_main_requires_replace_when_splits_exist(tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    spool.mkdir()
    _wav(spool / "p0a.wav")
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"version": 1, "clusters": [_clusters()[0]]}), encoding="utf-8")
    model_dir = tmp_path / "output" / "sayso"
    argv = ["training-data", "--spool", str(spool), "--ledger", str(ledger), "--model-dir", str(model_dir)]
    assert prepare.main(argv) == 0
    assert prepare.main(argv) == 1
