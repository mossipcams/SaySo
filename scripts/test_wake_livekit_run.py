"""Colocated tests for scripts/wake_livekit_run.py (stub CLI; no GPU/TTS)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import wake_hard_negative_mine as mine  # noqa: E402
from scripts import wake_livekit_run  # noqa: E402

_PROD_CONFIG = _REPO_ROOT / "satellite" / "models" / "sayso.yaml"
_SMOKE_CONFIG = _REPO_ROOT / "satellite" / "models" / "sayso-smoke.yaml"


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_prod_yaml_livekit_contract() -> None:
    cfg = _load(_PROD_CONFIG)
    assert cfg["model_name"] == "sayso"
    assert set(cfg["target_phrases"]) == {"SaySo", "Sayso"}
    assert "say so" not in cfg["target_phrases"]
    assert "say so" in cfg["custom_negative_phrases"]
    assert "tts_backend" not in cfg
    assert cfg["n_samples"] == 25000
    assert cfg["n_samples_val"] == 5000
    assert cfg["n_background_samples"] == 2000
    assert cfg["n_background_samples_val"] == 500
    assert cfg["augmentation"]["clip_duration"] == 2.0
    assert cfg["augmentation"]["rounds"] == 3
    assert cfg["model"]["model_type"] == "conv_attention"
    assert cfg["model"]["model_size"] == "small"
    assert cfg["steps"] == 100000
    assert cfg["max_negative_weight"] == 5000
    assert cfg["target_fp_per_hour"] == 0.05
    assert cfg["batch_n_per_class"] == {
        "positive": 50,
        "adversarial_negative": 50,
        "ACAV100M_sample": 2048,
        "background_noise": 50,
    }


def test_smoke_yaml_includes_generate_scale() -> None:
    cfg = _load(_SMOKE_CONFIG)
    assert cfg["n_samples"] == 100
    assert cfg["n_samples_val"] == 20
    assert cfg["model"]["model_type"] == "dnn"
    assert cfg["model"]["model_size"] == "tiny"
    assert cfg["steps"] == 500
    assert cfg["target_fp_per_hour"] == 1.0


def test_rewrite_config_paths_under_work_dir(tmp_path: Path) -> None:
    cfg = _load(_PROD_CONFIG)
    rewritten = wake_livekit_run.rewrite_config_for_work_dir(cfg, tmp_path / "run")
    work = tmp_path / "run"
    assert rewritten["data_dir"] == str(work / "data")
    assert rewritten["output_dir"] == str(work / "output")
    assert rewritten["augmentation"]["background_paths"] == [str(work / "data" / "backgrounds")]
    assert rewritten["augmentation"]["rir_paths"] == [str(work / "data" / "rirs")]


def test_run_pipeline_stage_order(tmp_path: Path) -> None:
    calls: list[tuple[str, Path]] = []

    def _stub_runner(stage: str, config_path: Path) -> None:
        calls.append((stage, config_path))

    executed = wake_livekit_run.run_pipeline(
        _PROD_CONFIG,
        tmp_path / "work",
        runner=_stub_runner,
    )
    assert executed == list(wake_livekit_run.RUN_STAGES)
    assert [stage for stage, _ in calls] == list(wake_livekit_run.RUN_STAGES)
    recipe = tmp_path / "work" / "recipe.yaml"
    assert recipe.is_file()
    assert all(path == recipe for _, path in calls)
    rewritten = _load(recipe)
    assert rewritten["output_dir"] == str(tmp_path / "work" / "output")
    assert "output-living2" not in rewritten["output_dir"]


def test_main_single_stage_uses_rewritten_config(tmp_path: Path) -> None:
    calls: list[str] = []

    def _stub_runner(stage: str, _config_path: Path) -> None:
        calls.append(stage)

    with patch.object(wake_livekit_run, "run_stage", side_effect=_stub_runner):
        rc = wake_livekit_run.main(
            [
                "generate",
                "--config",
                str(_PROD_CONFIG),
                "--work-dir",
                str(tmp_path / "host"),
            ]
        )
    assert rc == 0
    assert calls == ["generate"]
    recipe = tmp_path / "host" / "recipe.yaml"
    assert recipe.is_file()


def test_main_run_invokes_full_pipeline(tmp_path: Path) -> None:
    with patch.object(wake_livekit_run, "run_pipeline", return_value=list(wake_livekit_run.RUN_STAGES)) as mock_run:
        rc = wake_livekit_run.main(
            [
                "run",
                "--config",
                str(_SMOKE_CONFIG),
                "--work-dir",
                str(tmp_path / "smoke"),
            ]
        )
    assert rc == 0
    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    assert args[0] == _SMOKE_CONFIG.resolve()
    assert args[1] == (tmp_path / "smoke").resolve()


def test_default_config_path() -> None:
    assert wake_livekit_run._DEFAULT_CONFIG == _PROD_CONFIG


def test_cli_help_via_direct_script_invocation() -> None:
    script = _REPO_ROOT / "scripts" / "wake_livekit_run.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "livekit.wakeword stage" in proc.stdout


def _write_acav_rows(path: Path, n: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n, 16, 96), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr


def _seed_augment_outputs(work_dir: Path, model_name: str = "sayso") -> dict[str, Path]:
    output_dir = work_dir / "output" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train_neg": output_dir / "negative_features_train.npy",
        "test_neg": output_dir / "negative_features_test.npy",
        "pos_train": output_dir / "positive_features_train.npy",
        "pos_test": output_dir / "positive_features_test.npy",
        "validation": output_dir / "validation_set_features.npy",
        "bg_test": output_dir / "background_features_test.npy",
    }
    _write_acav_rows(paths["train_neg"], 8, seed=1)
    _write_acav_rows(paths["test_neg"], 3, seed=2)
    _write_acav_rows(paths["pos_train"], 4, seed=3)
    _write_acav_rows(paths["pos_test"], 2, seed=4)
    _write_acav_rows(paths["validation"], 5, seed=5)
    _write_acav_rows(paths["bg_test"], 2, seed=6)
    return paths


def _write_miner_pair(tmp_path: Path, *, n_source: int = 20, top_k: int = 3) -> tuple[Path, Path]:
    source_path = tmp_path / "acav_source.npy"
    _write_acav_rows(source_path, n_source, seed=10)
    out_features = tmp_path / "selected.npy"
    provenance_path = tmp_path / "selected.provenance.json"
    config = mine.MineConfig(
        features_path=source_path,
        output_features_path=out_features,
        provenance_path=provenance_path,
        model_path=None,
        top_k=top_k,
        chunk_size=4,
        diversity_gap=0,
    )

    def scorer(batch: np.ndarray) -> np.ndarray:
        return batch.reshape(batch.shape[0], -1).mean(axis=1).astype(np.float64)

    mine.run_mine(config, scorer=scorer)
    return out_features, provenance_path


def test_run_pipeline_with_overlay_stage_order(tmp_path: Path) -> None:
    calls: list[str] = []
    work = tmp_path / "work"
    overlay_features, overlay_provenance = _write_miner_pair(tmp_path / "mine")
    _seed_augment_outputs(work)

    def _stub_runner(stage: str, _config_path: Path) -> None:
        calls.append(stage)

    executed = wake_livekit_run.run_pipeline(
        _PROD_CONFIG,
        work,
        runner=_stub_runner,
        hard_negative_features=overlay_features,
        hard_negative_provenance=overlay_provenance,
        overlay_chunk_rows=2,
    )
    assert executed == list(wake_livekit_run.pipeline_stages(with_overlay=True))
    assert calls == ["setup", "generate", "augment", "train", "export", "eval"]
    manifest = work / wake_livekit_run.OVERLAY_MANIFEST_NAME
    assert manifest.is_file()


def test_overlay_appends_train_negatives_only(tmp_path: Path) -> None:
    work = tmp_path / "work"
    recipe = wake_livekit_run.write_work_config(_PROD_CONFIG, work)
    paths = _seed_augment_outputs(work)
    overlay_features, overlay_provenance = _write_miner_pair(tmp_path / "mine", top_k=4)
    before = {
        name: path.read_bytes()
        for name, path in paths.items()
        if name != "train_neg"
    }
    train_before = np.load(paths["train_neg"])
    overlay = np.load(overlay_features)

    wake_livekit_run.apply_hard_negative_overlay(
        recipe,
        work,
        overlay_features,
        overlay_provenance,
        chunk_rows=2,
    )

    merged = np.load(paths["train_neg"])
    assert merged.shape[0] == train_before.shape[0] + overlay.shape[0]
    np.testing.assert_array_equal(merged[: train_before.shape[0]], train_before)
    np.testing.assert_array_equal(merged[train_before.shape[0] :], overlay)
    for name, path in paths.items():
        if name == "train_neg":
            continue
        assert path.read_bytes() == before[name]


def test_overlay_append_failure_removes_tmp_file(tmp_path: Path) -> None:
    work = tmp_path / "work"
    recipe = wake_livekit_run.write_work_config(_PROD_CONFIG, work)
    paths = _seed_augment_outputs(work)
    overlay_features, overlay_provenance = _write_miner_pair(tmp_path / "mine", top_k=4)
    train_path = paths["train_neg"]
    tmp_path_file = train_path.with_name(f"{train_path.name}.overlay.tmp")
    train_before = train_path.read_bytes()
    call_count = 0
    original_asarray = np.asarray

    def _failing_asarray(obj: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        nonlocal call_count
        call_count += 1
        if call_count > 2:
            raise RuntimeError("simulated chunk failure")
        return original_asarray(obj, *args, **kwargs)

    with patch("scripts.wake_livekit_run.np.asarray", side_effect=_failing_asarray):
        with pytest.raises(RuntimeError, match="simulated chunk failure"):
            wake_livekit_run.apply_hard_negative_overlay(
                recipe,
                work,
                overlay_features,
                overlay_provenance,
                chunk_rows=2,
            )
    assert not tmp_path_file.exists()
    assert train_path.read_bytes() == train_before


def test_overlay_bounded_copy_materializes_chunks(tmp_path: Path) -> None:
    work = tmp_path / "work"
    recipe = wake_livekit_run.write_work_config(_PROD_CONFIG, work)
    _seed_augment_outputs(work)
    overlay_features, overlay_provenance = _write_miner_pair(tmp_path / "mine", top_k=5)
    max_rows = 0
    original_asarray = np.asarray

    def _tracking_asarray(obj: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        nonlocal max_rows
        arr = original_asarray(obj, *args, **kwargs)
        if arr.ndim == 3 and tuple(arr.shape[1:]) == (16, 96):
            max_rows = max(max_rows, arr.shape[0])
        return arr

    with patch("scripts.wake_livekit_run.np.asarray", side_effect=_tracking_asarray):
        wake_livekit_run.apply_hard_negative_overlay(
            recipe,
            work,
            overlay_features,
            overlay_provenance,
            chunk_rows=2,
        )
    assert max_rows <= 2


def test_overlay_rejects_sha256_mismatch(tmp_path: Path) -> None:
    overlay_features, overlay_provenance = _write_miner_pair(tmp_path / "mine")
    doc = json.loads(overlay_provenance.read_text(encoding="utf-8"))
    doc["output"]["sha256"] = "0" * 64
    overlay_provenance.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        wake_livekit_run.verify_hard_negative_provenance(overlay_provenance, overlay_features)


def test_overlay_rejects_malformed_provenance(tmp_path: Path) -> None:
    overlay_features, _ = _write_miner_pair(tmp_path / "mine")
    bad = tmp_path / "bad.provenance.json"
    bad.write_text('{"output": {}}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        wake_livekit_run.verify_hard_negative_provenance(bad, overlay_features)


def test_overlay_does_not_mutate_miner_source_or_validation(tmp_path: Path) -> None:
    work = tmp_path / "work"
    recipe = wake_livekit_run.write_work_config(_PROD_CONFIG, work)
    paths = _seed_augment_outputs(work)
    overlay_features, overlay_provenance = _write_miner_pair(tmp_path / "mine")
    source_path = tmp_path / "mine" / "acav_source.npy"
    source_before = source_path.read_bytes()
    overlay_before = overlay_features.read_bytes()
    provenance_before = overlay_provenance.read_text(encoding="utf-8")
    validation_before = paths["validation"].read_bytes()

    wake_livekit_run.apply_hard_negative_overlay(
        recipe,
        work,
        overlay_features,
        overlay_provenance,
    )

    assert source_path.read_bytes() == source_before
    assert overlay_features.read_bytes() == overlay_before
    assert overlay_provenance.read_text(encoding="utf-8") == provenance_before
    assert paths["validation"].read_bytes() == validation_before


def test_main_run_passes_overlay_args(tmp_path: Path) -> None:
    overlay_features = tmp_path / "overlay.npy"
    overlay_provenance = tmp_path / "overlay.provenance.json"
    overlay_features.write_bytes(b"x")
    overlay_provenance.write_text("{}", encoding="utf-8")
    with patch.object(wake_livekit_run, "run_pipeline") as mock_run:
        rc = wake_livekit_run.main(
            [
                "run",
                "--config",
                str(_SMOKE_CONFIG),
                "--work-dir",
                str(tmp_path / "smoke"),
                "--hard-negative-features",
                str(overlay_features),
                "--hard-negative-provenance",
                str(overlay_provenance),
            ]
        )
    assert rc == 0
    _, kwargs = mock_run.call_args
    assert kwargs["hard_negative_features"] == overlay_features
    assert kwargs["hard_negative_provenance"] == overlay_provenance


def test_main_rejects_partial_overlay_flags(tmp_path: Path) -> None:
    rc = wake_livekit_run.main(
        [
            "run",
            "--config",
            str(_SMOKE_CONFIG),
            "--work-dir",
            str(tmp_path / "smoke"),
            "--hard-negative-features",
            str(tmp_path / "only.npy"),
        ]
    )
    assert rc == 2


def test_default_runner_setup_uses_config_flag(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("data_dir: x\n", encoding="utf-8")
    captured: list[list[str]] = []

    def _fake_run(argv: list[str], **kwargs: Any) -> Any:
        captured.append(list(argv))

        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Result()

    with patch("scripts.wake_livekit_run.subprocess.run", side_effect=_fake_run):
        wake_livekit_run.run_stage("setup", recipe)
        wake_livekit_run.run_stage("generate", recipe)

    setup_argv = captured[0]
    generate_argv = captured[1]

    setup_stage_idx = setup_argv.index("setup")
    config_flag = next(
        (flag for flag in ("--config", "-c") if flag in setup_argv[setup_stage_idx + 1 :]),
        None,
    )
    assert config_flag is not None, "setup must pass --config or -c before the recipe path"
    flag_idx = setup_argv.index(config_flag)
    assert flag_idx > setup_stage_idx
    assert setup_argv[flag_idx + 1] == str(recipe)

    generate_stage_idx = generate_argv.index("generate")
    assert generate_argv[generate_stage_idx + 1] == str(recipe)
    assert "--config" not in generate_argv
    assert "-c" not in generate_argv
