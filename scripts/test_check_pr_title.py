"""Tests for scripts/check_pr_title.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_pr_title  # noqa: E402


def test_rejects_add_prefix_title() -> None:
    ok, _ = check_pr_title.validate_title(
        "Add compiled tool schema identity, envelope validation, and the v1 reference artifact"
    )
    assert not ok


def test_rejects_harden_prefix_title() -> None:
    ok, _ = check_pr_title.validate_title(
        "Harden llama.cpp tool path for schema envelope validation"
    )
    assert not ok


def test_accepts_feat() -> None:
    ok, msg = check_pr_title.validate_title(
        "feat: add compiled tool schema identity and v1 artifact"
    )
    assert ok, msg


def test_accepts_fix_with_scope() -> None:
    ok, _ = check_pr_title.validate_title(
        "fix(conversation): fail fast on invalid envelope"
    )
    assert ok


def test_accepts_perf_deps_chore_docs_ci() -> None:
    for title in (
        "perf: reduce schema compile latency",
        "deps: bump homeassistant to 2026.8.3",
        "chore: refresh lockfile",
        "docs: document schema lock plan",
        "ci: require conventional pull request titles",
    ):
        ok, msg = check_pr_title.validate_title(title)
        assert ok, msg


def test_accepts_build_refactor_test_style_revert() -> None:
    for title in (
        "build: pin setuptools",
        "refactor(schema): extract fingerprint helper",
        "test: cover envelope validation",
        "style: format compat matrix",
        "revert: undo experimental schema path",
    ):
        ok, msg = check_pr_title.validate_title(title)
        assert ok, msg


def test_accepts_breaking_bang() -> None:
    ok, msg = check_pr_title.validate_title("feat(schema)!: remove legacy envelope")
    assert ok, msg


def test_rejects_missing_description() -> None:
    ok, _ = check_pr_title.validate_title("feat:")
    assert not ok


def test_rejects_unknown_type() -> None:
    ok, _ = check_pr_title.validate_title("feature: add something")
    assert not ok


def test_main_reads_env_pr_title(monkeypatch) -> None:
    monkeypatch.delenv("PR_TITLE", raising=False)
    monkeypatch.setenv("PR_TITLE", "feat: env title")
    assert check_pr_title.main(["check_pr_title.py"]) == 0


def test_main_reads_argv() -> None:
    assert check_pr_title.main(["check_pr_title.py", "fix: argv title"]) == 0


def test_main_fails_on_invalid_argv() -> None:
    assert check_pr_title.main(["check_pr_title.py", "Add something"]) == 1
