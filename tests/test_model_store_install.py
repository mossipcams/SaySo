import os
import subprocess as _subprocess
from types import SimpleNamespace

import custom_components.sayso.model_store as model_store
from custom_components.sayso.const import (
    LLAMA_CPP_PACKAGE,
    LLAMA_CPP_VERSION,
    LLAMA_CPP_WHEEL_INDEX,
)


def _fake_run(calls):
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stderr=b"")

    return run


def test_install_does_not_mutate_environ(monkeypatch):
    before = os.environ.copy()
    monkeypatch.delenv("UV_EXTRA_INDEX_URL", raising=False)
    monkeypatch.setattr(
        model_store,
        "subprocess",
        SimpleNamespace(run=_fake_run([]), PIPE=_subprocess.PIPE),
    )
    assert model_store._install_llama_cpp() is True
    assert os.environ == before


def test_install_preserves_existing_extra_index(monkeypatch):
    monkeypatch.setenv("UV_EXTRA_INDEX_URL", "https://preset.example/index")
    before = os.environ.copy()
    monkeypatch.setattr(
        model_store,
        "subprocess",
        SimpleNamespace(run=_fake_run([]), PIPE=_subprocess.PIPE),
    )
    assert model_store._install_llama_cpp() is True
    assert os.environ == before


def test_install_uses_pinned_requirement_and_wheel_index(monkeypatch):
    monkeypatch.setenv("UV_EXTRA_INDEX_URL", "https://preset.example/index")
    calls = []
    monkeypatch.setattr(
        model_store,
        "subprocess",
        SimpleNamespace(run=_fake_run(calls), PIPE=_subprocess.PIPE),
    )
    assert model_store._install_llama_cpp() is True
    llama_args, kwargs = next(
        (args, kw)
        for args, kw in calls
        if any(LLAMA_CPP_PACKAGE in part for part in args)
    )
    assert f"{LLAMA_CPP_PACKAGE}=={LLAMA_CPP_VERSION}" in llama_args
    env = kwargs["env"]
    assert "https://preset.example/index" in env["UV_EXTRA_INDEX_URL"]
    assert env["UV_EXTRA_INDEX_URL"].endswith(LLAMA_CPP_WHEEL_INDEX)
    strategy_index = llama_args.index("--index-strategy")
    assert llama_args[strategy_index + 1] == "unsafe-first-match"
    constraint_index = llama_args.index("--constraint")
    assert (
        os.path.basename(llama_args[constraint_index + 1]) == "package_constraints.txt"
    )


def test_install_failure_returns_false(monkeypatch):
    def run(args, **kwargs):
        return SimpleNamespace(returncode=1, stderr=b"boom")

    monkeypatch.setattr(
        model_store, "subprocess", SimpleNamespace(run=run, PIPE=_subprocess.PIPE)
    )
    assert model_store._install_llama_cpp() is False


def test_target_added_when_not_venv(monkeypatch):
    calls = []
    monkeypatch.setattr(model_store.pkg_util, "is_virtual_env", lambda: False)
    monkeypatch.setattr(
        model_store.site,
        "getusersitepackages",
        lambda: "/fake/user/site",
    )
    monkeypatch.delenv("UV_SYSTEM_PYTHON", raising=False)
    monkeypatch.delenv("UV_PYTHON", raising=False)
    monkeypatch.setattr(
        model_store,
        "subprocess",
        SimpleNamespace(run=_fake_run(calls), PIPE=_subprocess.PIPE),
    )
    assert model_store._install_llama_cpp() is True
    llama_args, _ = next(
        (args, kw)
        for args, kw in calls
        if any(LLAMA_CPP_PACKAGE in part for part in args)
    )
    target_index = llama_args.index("--target")
    assert llama_args[target_index + 1] == "/fake/user/site"


def test_no_target_when_uv_system_python_set(monkeypatch):
    calls = []
    monkeypatch.setattr(model_store.pkg_util, "is_virtual_env", lambda: False)
    monkeypatch.setattr(
        model_store.site,
        "getusersitepackages",
        lambda: "/fake/user/site",
    )
    monkeypatch.setenv("UV_SYSTEM_PYTHON", "1")
    monkeypatch.delenv("UV_PYTHON", raising=False)
    monkeypatch.setattr(
        model_store,
        "subprocess",
        SimpleNamespace(run=_fake_run(calls), PIPE=_subprocess.PIPE),
    )
    assert model_store._install_llama_cpp() is True
    llama_args, _ = next(
        (args, kw)
        for args, kw in calls
        if any(LLAMA_CPP_PACKAGE in part for part in args)
    )
    assert "--target" not in llama_args
