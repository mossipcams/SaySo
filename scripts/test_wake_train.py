"""Colocated tests for scripts/wake_train.py."""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import wake_train  # noqa: E402
from satellite.sayso.wake.eval import write_synthetic_wav  # noqa: E402
from satellite.sayso.wake.livekit import HOP_SAMPLES, SAMPLE_RATE, WINDOW_SAMPLES  # noqa: E402


def _write_pcm_wav(path: Path, peak: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.full(SAMPLE_RATE * 2, peak, dtype="<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.tobytes())


def _write_record(spool: Path, capture_id: str, *, peak: int = 2000) -> None:
    record_dir = spool / "records" / capture_id
    record_dir.mkdir(parents=True)
    _write_pcm_wav(record_dir / "window.wav", peak)
    meta = {
        "capture_id": capture_id,
        "score": 0.12,
        "sampling_reason": "near_threshold",
        "hashes": {},
    }
    (record_dir / "record.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _write_seed(seed_dir: Path) -> None:
    _write_pcm_wav(seed_dir / "positive" / "seed_pos.wav", 7000)
    _write_pcm_wav(seed_dir / "negative" / "seed_neg.wav", 200)


def _fixture_paths(tmp_path: Path) -> dict[str, Path]:
    spool = tmp_path / "spool"
    seed_dir = tmp_path / "seed"
    work_dir = tmp_path / "work"
    eval_root = tmp_path / "eval"
    splits_path = tmp_path / "splits.json"
    baseline_path = tmp_path / "baseline.json"
    recipe_path = tmp_path / "recipe.yaml"
    capture_id = "real_capture_001"
    _write_record(spool, capture_id, peak=2000)
    _write_seed(seed_dir)
    splits_path.write_text(
        json.dumps(
            {
                "version": 1,
                "groups": [
                    {
                        "id": "eval_holdout_cases",
                        "split": "eval",
                        "holdout": True,
                        "members": ["pos_sayso_near"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    baseline_path.write_text(
        json.dumps(
            {
                "version": 1,
                "status": "blocked",
                "blocker": "fixtures only",
                "deployed_threshold": {"recall": 0.5, "fpph": 0.1},
            }
        ),
        encoding="utf-8",
    )
    recipe_path.write_text("model_name: sayso\n", encoding="utf-8")
    eval_root.mkdir()
    return {
        "spool": spool,
        "seed_dir": seed_dir,
        "work_dir": work_dir,
        "eval_root": eval_root,
        "splits_path": splits_path,
        "baseline_path": baseline_path,
        "recipe_path": recipe_path,
        "capture_id": capture_id,
    }


def _good_eval_report() -> dict:
    return {
        "summary": {"total": 2, "passed": 2, "failed": 0, "skipped": 0, "errors": 0},
        "aggregate": {"background_duration_seconds": SAMPLE_RATE * 2},
        "results": [
            {
                "case_id": "pos",
                "category": "positive_sayso",
                "status": "passed",
                "detection_ok": True,
                "detected": True,
                "detection_sample": WINDOW_SAMPLES,
            },
            {
                "case_id": "background",
                "category": "negative_tv_conversation",
                "status": "passed",
                "detection_ok": True,
                "detected": False,
            },
        ],
    }


def _run_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    paths: dict[str, Path] | None = None,
    **kwargs,
) -> wake_train.PipelineResult:
    paths = paths or _fixture_paths(tmp_path)
    kwargs.setdefault("spool", paths["spool"])
    kwargs.setdefault("seed_dir", paths["seed_dir"])
    kwargs.setdefault("work_dir", paths["work_dir"])
    kwargs.setdefault("recipe_path", paths["recipe_path"])
    kwargs.setdefault("eval_root", paths["eval_root"])
    kwargs.setdefault("baseline_path", paths["baseline_path"])
    kwargs.setdefault("splits_path", paths["splits_path"])
    kwargs.setdefault("require_known_real", paths["capture_id"])
    kwargs.setdefault("eval_runner", lambda *_args, **_kw: _good_eval_report())
    return wake_train.run_pipeline(**kwargs)


def test_fixture_pipeline_ingest_train_eval_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_fixture(tmp_path, monkeypatch)
    assert result.status == "insufficient_evidence"
    assert result.bundle_dir is not None
    candidate = json.loads((result.bundle_dir / "candidate.json").read_text(encoding="utf-8"))
    assert candidate["status"] == "insufficient_evidence"
    assert (result.bundle_dir / "sayso.onnx").is_file()
    assert paths_capture_reaches_training(tmp_path, result)


def paths_capture_reaches_training(tmp_path: Path, result: wake_train.PipelineResult) -> bool:
    run_dir = tmp_path / "work" / "runs" / result.run_key
    state = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    snapshot_root = tmp_path / "work" / "snapshots" / state["snapshot_id"]
    mapping = json.loads((snapshot_root / "mapping.json").read_text(encoding="utf-8"))
    return "real_capture_001" in mapping


def test_identical_rerun_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture_paths(tmp_path)
    first = _run_fixture(tmp_path, monkeypatch, paths=paths)
    snapshots_after_first = list((tmp_path / "work" / "snapshots").glob("*"))
    second = _run_fixture(tmp_path, monkeypatch, paths=paths)
    assert first.run_key == second.run_key
    assert second.noop is True
    assert second.status == first.status
    snapshots_after_second = list((tmp_path / "work" / "snapshots").glob("*"))
    assert len(snapshots_after_second) == len(snapshots_after_first)


def test_teacher_failure_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_fixture(
        tmp_path,
        monkeypatch,
        teacher=wake_train.StubTeacher(fail=True),
    )
    assert result.status == "rejected"
    assert result.bundle_dir is None


def test_split_leakage_rejected(tmp_path: Path) -> None:
    splits = wake_train.load_splits(_fixture_paths(tmp_path)["splits_path"])
    leaked = wake_train.Example(
        source_id="pos_sayso_near",
        wav_path=tmp_path / "x.wav",
        label="positive",
        split="train",
        origin="manual",
    )
    good = wake_train.Example(
        source_id="seed_pos",
        wav_path=tmp_path / "y.wav",
        label="positive",
        split="train",
        origin="trusted_seed",
    )
    with pytest.raises(RuntimeError, match="split leakage"):
        wake_train.select_examples([leaked, good], splits)


def test_select_examples_excludes_corpus_holdout_sessions(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "corpus_splits.json").write_text(
        json.dumps(
            {
                "version": 1,
                "seed": 1,
                "holdout_session_ids": ["holdout_session"],
                "assignments": {
                    "holdout_session": "holdout",
                    "train_session": "train",
                },
            }
        ),
        encoding="utf-8",
    )
    holdout_example = wake_train.Example(
        source_id="evt_hold",
        wav_path=tmp_path / "hold.wav",
        label="positive",
        split="train",
        origin="corpus_event",
        meta={"session_id": "holdout_session"},
    )
    train_example = wake_train.Example(
        source_id="evt_train",
        wav_path=tmp_path / "train.wav",
        label="positive",
        split="train",
        origin="corpus_event",
        meta={"session_id": "train_session"},
    )
    neg = wake_train.Example(
        source_id="evt_neg",
        wav_path=tmp_path / "neg.wav",
        label="negative",
        split="train",
        origin="corpus_event",
        meta={"session_id": "train_session"},
    )
    selected = wake_train.select_examples(
        [holdout_example, train_example, neg],
        {},
        corpus_root=corpus,
    )
    source_ids = {ex.source_id for ex in selected}
    assert "evt_hold" not in source_ids
    assert {"evt_train", "evt_neg"} <= source_ids


def test_omitted_real_features_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_fixture(
        tmp_path,
        monkeypatch,
        omit_feature_for="real_capture_001",
    )
    assert result.status == "rejected"
    assert "source-to-feature" in (result.reason or "")


def test_failed_export_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_fixture(
        tmp_path,
        monkeypatch,
        trainer=wake_train.StubTrainer(export_ok=False),
    )
    assert result.status == "rejected"
    assert result.reason == "export failed"


def test_empty_eval_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_fixture(
        tmp_path,
        monkeypatch,
        eval_runner=lambda *_a, **_k: {
            "summary": {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0},
            "aggregate": {"background_duration_seconds": 0.0},
            "results": [],
        },
    )
    assert result.status == "rejected"


def test_worse_candidate_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_fixture(
        tmp_path,
        monkeypatch,
        trainer=wake_train.StubTrainer(quality="bad"),
        eval_runner=lambda *_a, **_k: {
            "summary": {"total": 2, "passed": 0, "failed": 2, "skipped": 0, "errors": 0},
            "aggregate": {"background_duration_seconds": SAMPLE_RATE * 2},
            "results": [
                {
                    "case_id": "pos",
                    "category": "positive_sayso",
                    "status": "failed",
                    "detection_ok": False,
                    "detected": False,
                },
                {
                    "case_id": "background",
                    "category": "negative_tv_conversation",
                    "status": "failed",
                    "detection_ok": False,
                    "detected": True,
                },
            ],
        },
    )
    assert result.status == "rejected"
    assert result.status != "qualified"


def test_late_detection_does_not_qualify(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eval_root = tmp_path / "late_eval"
    wake_train._make_fixture_eval(eval_root, late_detection=True)
    late_report = _good_eval_report()
    late_report["results"][0]["detection_sample"] = WINDOW_SAMPLES + HOP_SAMPLES * 4
    paths = _fixture_paths(tmp_path)
    baseline = json.loads(paths["baseline_path"].read_text(encoding="utf-8"))
    baseline["status"] = "ok"
    paths["baseline_path"].write_text(json.dumps(baseline), encoding="utf-8")
    result = wake_train.run_pipeline(
        spool=paths["spool"],
        seed_dir=paths["seed_dir"],
        work_dir=paths["work_dir"],
        recipe_path=paths["recipe_path"],
        eval_root=paths["eval_root"],
        baseline_path=paths["baseline_path"],
        splits_path=paths["splits_path"],
        require_known_real=paths["capture_id"],
        eval_root_override=eval_root,
        eval_runner=lambda *_a, **_k: late_report,
    )
    assert result.status == "rejected"
    assert result.status != "qualified"


def test_interruption_rejected_without_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_fixture(tmp_path, monkeypatch, interrupt_after="snapshot")
    assert result.status == "rejected"
    assert result.bundle_dir is None


def test_schedule_refused_while_baseline_blocked(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    reason = wake_train.schedule_refused_message(
        seed_dir=paths["seed_dir"],
        baseline_path=paths["baseline_path"],
        eval_root=paths["eval_root"],
    )
    assert reason is not None
    rc = wake_train.main(
        [
            "--spool",
            str(paths["spool"]),
            "--seed-dir",
            str(paths["seed_dir"]),
            "--work-dir",
            str(paths["work_dir"]),
            "--schedule",
        ]
    )
    assert rc == 2


def test_automatic_positive_labels_fail_closed(tmp_path: Path) -> None:
    class BadTeacher(wake_train.StubTeacher):
        def label_window(self, wav_path: Path, meta: dict) -> wake_train.TeacherVerdict:
            return wake_train.TeacherVerdict("positive", "bad", {})

    paths = _fixture_paths(tmp_path)
    rows = wake_train.ingest_sources(paths["spool"], paths["seed_dir"])
    with pytest.raises(RuntimeError, match="automatic positive"):
        wake_train.label_records(rows, BadTeacher())


def test_resolve_teacher_stub_mode() -> None:
    teacher = wake_train._resolve_teacher(True, None)
    assert isinstance(teacher, wake_train.StubTeacher)


def test_resolve_trainer_stub_mode() -> None:
    trainer = wake_train._resolve_trainer(True, None)
    assert isinstance(trainer, wake_train.StubTrainer)


def test_resolve_teacher_real_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wake_train, "_load_whisper_model_class", lambda: MagicMock())
    teacher = wake_train._resolve_teacher(False, None)
    assert isinstance(teacher, wake_train.FasterWhisperTeacher)


def test_resolve_trainer_real_mode() -> None:
    trainer = wake_train._resolve_trainer(False, None)
    assert isinstance(trainer, wake_train.LiveKitTrainer)


def test_faster_whisper_missing_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> None:
        raise RuntimeError(
            "faster-whisper is required for non-stub training; "
            "install satellite/models/requirements-wake-train.txt on the host"
        )

    monkeypatch.setattr(wake_train, "_load_whisper_model_class", _raise)
    with pytest.raises(RuntimeError, match="faster-whisper"):
        wake_train.FasterWhisperTeacher()


def test_livekit_missing_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _raise() -> None:
        raise RuntimeError(
            "livekit-wakeword is required for non-stub training; "
            "install satellite/models/requirements-wake-train.txt on the host"
        )

    monkeypatch.setattr(wake_train, "_import_livekit_wakeword", _raise)
    trainer = wake_train.LiveKitTrainer()
    snapshot = wake_train.Snapshot(
        snapshot_id="snap",
        root=tmp_path / "snap",
        mapping={"a": "clip_000001"},
        recipe_path=tmp_path / "recipe.yaml",
        counts={"positive": 1, "negative": 1},
        hashes={},
        seed=1,
    )
    with pytest.raises(RuntimeError, match="livekit-wakeword"):
        trainer.train_and_export(snapshot, tmp_path / "out")


def test_faster_whisper_never_emits_positive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wav_path = tmp_path / "window.wav"
    _write_pcm_wav(wav_path, 5000)

    class _Word:
        def __init__(self, word: str, start: float, end: float, probability: float) -> None:
            self.word = word
            self.start = start
            self.end = end
            self.probability = probability

    class _Segment:
        def __init__(self, text: str) -> None:
            self.text = text
            self.words = [_Word("sayso", 0.0, 0.5, 0.99)]

    class _Info:
        language = "en"
        language_probability = 0.99
        no_speech_prob = 0.01

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([_Segment("sayso")], _Info())
    monkeypatch.setattr(wake_train, "_load_whisper_model_class", lambda: MagicMock(return_value=mock_model))

    teacher = wake_train.FasterWhisperTeacher()
    verdict = teacher.label_window(wav_path, {})
    assert verdict.label is None
    assert verdict.reason == "no_automatic_positive"


def test_livekit_staging_excludes_holdout_from_train(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    examples = [
        wake_train.Example(
            "seed_pos",
            paths["seed_dir"] / "positive" / "seed_pos.wav",
            "positive",
            "train",
            "trusted_seed",
        ),
        wake_train.Example(
            "seed_neg_a",
            paths["seed_dir"] / "negative" / "seed_neg.wav",
            "negative",
            "train",
            "trusted_seed",
        ),
        wake_train.Example(
            "real_capture_001",
            paths["spool"] / "records" / "real_capture_001" / "window.wav",
            "negative",
            "train",
            "manual",
        ),
    ]
    snapshot = wake_train.build_snapshot(
        examples,
        work_dir=paths["work_dir"],
        recipe_path=paths["recipe_path"],
        seed=1,
        teacher_audit={"teacher_version": "stub", "policy_version": "v1"},
    )
    model_dir = tmp_path / "output" / "sayso"
    mapping = wake_train._stage_snapshot_clips(snapshot, model_dir)
    clip_labels = json.loads((snapshot.root / "manifest.json").read_text())["clip_labels"]
    positive_clips = sorted(clip for clip, label in clip_labels.items() if label == "positive")
    negative_clips = sorted(clip for clip, label in clip_labels.items() if label == "negative")
    assert len(positive_clips) == 1
    singleton_positive = positive_clips[0]
    assert (model_dir / "positive_train" / f"{singleton_positive}.wav").is_file()
    assert not (model_dir / "positive_test" / f"{singleton_positive}.wav").exists()
    holdout = negative_clips[-1]
    assert (model_dir / "negative_test" / f"{holdout}.wav").is_file()
    assert not (model_dir / "negative_train" / f"{holdout}.wav").exists()


def test_false_activations_per_hour_counts_activation_samples() -> None:
    report = {
        "aggregate": {"background_duration_seconds": 3600.0},
        "results": [
            {
                "category": "negative_tv_conversation",
                "detected": True,
                "activation_samples": [1000, 50000],
            },
            {
                "category": "negative_distance_noise",
                "detected": False,
                "activation_samples": [],
            },
        ],
    }
    assert wake_train._false_activations_per_hour(report) == pytest.approx(2.0)


def test_acquire_lock_second_call_fails(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    lock = wake_train.acquire_lock(work_dir)
    assert lock.is_file()
    with pytest.raises(RuntimeError, match="another wake_train run holds the lock"):
        wake_train.acquire_lock(work_dir)
    wake_train.release_lock(lock)


def test_pipeline_eval_uses_strict_and_production_refractory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(wake_train, "production_refractory_seconds", lambda: 2.0)

    def _run_wake_eval(**kwargs):
        captured.update(kwargs)
        return _good_eval_report()

    monkeypatch.setattr(wake_train, "run_wake_eval", _run_wake_eval)
    _run_fixture(tmp_path, monkeypatch, eval_runner=None)
    assert captured.get("strict") is True
    assert captured.get("refractory_seconds") == 2.0


def test_source_feature_count_assert(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    examples = [
        wake_train.Example("a", paths["spool"] / "records" / "real_capture_001" / "window.wav", "positive", "train", "seed"),
        wake_train.Example("b", paths["seed_dir"] / "negative" / "seed_neg.wav", "negative", "train", "seed"),
    ]
    snapshot = wake_train.build_snapshot(
        examples,
        work_dir=paths["work_dir"],
        recipe_path=paths["recipe_path"],
        seed=1,
        teacher_audit={"teacher_version": "stub", "policy_version": "v1"},
    )
    feature = snapshot.root / "clips" / f"{snapshot.mapping['a']}_r0.wav"
    feature.unlink()
    with pytest.raises(RuntimeError, match="source-to-feature"):
        wake_train.assert_source_feature_counts(snapshot)


def _write_corpus_event(corpus: Path, event_id: str, session_id: str, *, label: str) -> None:
    from satellite.sayso.wake.corpus import import_spool_record
    from satellite.sayso.wake.livekit import WINDOW_SAMPLES

    spool = corpus / "spool"
    record_dir = spool / "records" / event_id
    record_dir.mkdir(parents=True)
    _write_pcm_wav(record_dir / "window.wav", 3000)
    (record_dir / "record.json").write_text(
        json.dumps(
            {
                "capture_id": event_id,
                "session_id": session_id,
                "score": 0.4,
                "sampling_reason": "near_threshold",
                "sample_start": 0,
                "sample_end": WINDOW_SAMPLES,
                "label": label,
                "hashes": {},
            }
        ),
        encoding="utf-8",
    )
    import_spool_record(record_dir, corpus)


def test_corpus_holdout_excluded_from_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture_paths(tmp_path)
    corpus = tmp_path / "corpus"
    _write_corpus_event(corpus, "hold_pos", "holdout_session", label="positive")
    _write_corpus_event(corpus, "hold_neg", "holdout_session", label="negative")
    _write_corpus_event(corpus, "train_pos", "train_session", label="positive")
    _write_corpus_event(corpus, "train_neg", "train_session", label="negative")
    (corpus / "corpus_splits.json").write_text(
        json.dumps(
            {
                "version": 1,
                "seed": 1,
                "holdout_session_ids": ["holdout_session"],
                "assignments": {
                    "holdout_session": "holdout",
                    "train_session": "train",
                },
            }
        ),
        encoding="utf-8",
    )
    rows = wake_train.ingest_corpus_rows(corpus, seed=1)
    source_ids = {row["capture_id"] for row in rows}
    assert "hold_pos" not in source_ids
    assert "hold_neg" not in source_ids
    assert {"train_pos", "train_neg"} <= source_ids


def test_living2_seed_preserved_without_replace_flag(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    rows = wake_train.ingest_sources(paths["spool"], paths["seed_dir"])
    origins = {row.get("origin") for row in rows}
    assert "trusted_seed" in origins


def test_replace_living2_omits_seed(tmp_path: Path) -> None:
    paths = _fixture_paths(tmp_path)
    rows = wake_train.ingest_sources(paths["spool"], paths["seed_dir"], corpus_root=None)
    assert any(row.get("origin") == "trusted_seed" for row in rows)
    rows = wake_train.ingest_sources(paths["spool"], None)
    assert not any(row.get("origin") == "trusted_seed" for row in rows)
