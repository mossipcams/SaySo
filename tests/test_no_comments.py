import ast
import io
import subprocess
import tokenize
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCOPED_NODES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _tracked(suffix: str) -> list[Path]:
    listing = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return [REPO_ROOT / name for name in listing if name.endswith(suffix)]


def _python_offenses(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    offenses = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type != tokenize.COMMENT:
            continue
        if token.start == (1, 0) and token.string.startswith("#!"):
            continue
        offenses.append(f"{token.start[0]}: comment")
    for node in ast.walk(ast.parse(text)):
        if (
            isinstance(node, SCOPED_NODES)
            and ast.get_docstring(node, clean=False) is not None
        ):
            offenses.append(f"{getattr(node, 'lineno', 1)}: docstring")
    return offenses


def _shell_offenses(path: Path) -> list[str]:
    offenses = []
    previous = ""
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        is_shebang = stripped.startswith("#!") and (number == 1 or "<<" in previous)
        if stripped.startswith("#") and not is_shebang:
            offenses.append(f"{number}: comment")
        previous = line
    return offenses


def _report(offenses_by_path: dict[Path, list[str]]) -> str:
    return "\n".join(
        f"{path.relative_to(REPO_ROOT)}:{offense}"
        for path, offenses in offenses_by_path.items()
        for offense in offenses
    )


def test_python_sources_have_no_comments_or_docstrings() -> None:
    offenses = {path: _python_offenses(path) for path in _tracked(".py")}
    offenses = {path: found for path, found in offenses.items() if found}
    assert not offenses, (
        "comments or docstrings in tracked Python sources:\n" + _report(offenses)
    )


def test_shell_sources_have_no_comments() -> None:
    offenses = {path: _shell_offenses(path) for path in _tracked(".sh")}
    offenses = {path: found for path, found in offenses.items() if found}
    assert not offenses, "comments in tracked shell sources:\n" + _report(offenses)
