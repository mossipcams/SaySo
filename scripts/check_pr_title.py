"""Validate pull request titles against Conventional Commits."""

from __future__ import annotations

import os
import re
import sys

ALLOWED_TYPES = frozenset(
    {
        "build",
        "chore",
        "ci",
        "deps",
        "docs",
        "feat",
        "fix",
        "perf",
        "refactor",
        "revert",
        "style",
        "test",
    }
)

# type(scope)!: description — scope allows word chars, dots, dashes, slashes.
_PR_TITLE_RE = re.compile(
    r"^(?P<type>[a-z]+)"
    r"(?:\((?P<scope>[\w./-]+)\))?"
    r"(?P<breaking>!)?"
    r":\s+"
    r"(?P<description>.+)$"
)


def validate_title(title: str) -> tuple[bool, str]:
    """Return (ok, error_message)."""
    stripped = title.strip()
    if not stripped:
        return False, "PR title must not be empty."

    match = _PR_TITLE_RE.match(stripped)
    if not match:
        return False, (
            "PR title must follow Conventional Commits: "
            "type[(scope)][!]: description "
            f"(got {stripped!r})"
        )

    commit_type = match.group("type")
    if commit_type not in ALLOWED_TYPES:
        allowed = ", ".join(sorted(ALLOWED_TYPES))
        return False, f"Unknown commit type {commit_type!r}. Allowed: {allowed}"

    description = match.group("description").strip()
    if not description:
        return False, "PR title description must not be empty after the colon."

    return True, ""


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv[1:]
    title = " ".join(args) if args else os.environ.get("PR_TITLE", "")

    ok, message = validate_title(title)
    if ok:
        return 0
    print(message, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
