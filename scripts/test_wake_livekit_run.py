"""Colocated tests for scripts/wake_livekit_run.py (stub CLI; no GPU/TTS)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
    assert cfg["max_negative_weight"] == 3000
    assert cfg["target_fp_per_hour"] == 0.1
    assert cfg["batch_n_per_class"] == {
        "positive": 50,
        "adversarial_negative": 50,
        "ACAV100M_sample": 1024,
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
