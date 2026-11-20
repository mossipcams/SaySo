# 0006. CI gate and agent rules

## Status

Accepted.

## Context

Before this change, CI did not run ruff or the root test suite, so a pull request could merge with lint errors or failing tests. `AGENTS.md` had no delegation rule, no rule against code comments, and a loose plan rule that allowed "a sibling scratch file". Comments and docstrings were common across the Python sources.

## Decision

- The Python CI job runs `ruff check .`, `ruff format --check .`, and `pytest tests evals/tests --ignore=tests/test_realistic_v3.py` on pull requests and pushes to `main`. Ruff is declared in the dev extra. Lint is narrowed to the `E9`, `F63`, `F7`, and `F82` rules, which catch syntax and undefined-name errors. Format excludes live in `pyproject.toml`. The job pins Home Assistant to `CURRENT_HA_VERSION`.
- The satellite suite stays in its own job because it needs the linux-voice-assistant checkout.
- `AGENTS.md` requires implementation work to go through the model-router delegate, with no native subagents and no bypass without per-request approval. Pull requests run the CI gate through the delegate.
- `AGENTS.md` bans code comments and docstrings in `.py` and `.sh` files, except the shebang line. Comments and docstrings were stripped with an AST and tokenize pass, not regex, so string literals are unchanged.
- Plans for active work are `docs/PLAN_<feature>.md`. They are deleted once shipped, and their decisions are recorded as an ADR.

Rejected:

- A mass lint fix to make the full ruff rule set pass.
- Regex-based comment removal, which would damage `#` inside strings.
- Keeping executed plans. They drifted from the code: the code-improvement plan contradicted `pyproject.toml` on the Home Assistant floor.

## Consequences

- CI checks syntax and undefined names only. Style is not enforced by the lint step, but format is checked.
- Comments and docstrings that later commits reintroduced (60 comments, 5 `noqa` or `type: ignore` directives, and 19 docstrings) were removed in `f66c02f` and `8070e56`. `tests/test_no_comments.py` (`f659bcb`) now enforces the rule for tracked `.py` and `.sh` files.
- GitNexus is not referenced anywhere in this repository. Its removal from the workspace skills is recorded outside this repository.

## Shipped in

- `3d5932a` ci: add ruff and root pytest to GitHub Actions
- `7bb6acf` ci: make Python job ruff and pytest gates pass without mass lint
- `9478ece` ci: pin python job homeassistant to CURRENT_HA_VERSION
- `a309572` ci: exclude docs from ruff format check
- `dcf213f` docs(agents): always save an md plan before implementing
- `6c30d14` chore(agent): remove all code comments and docstrings; mandate model-router delegation and no-comments rule in AGENTS.md
- `9f91300` docs(agents): local agent runs the full delegation loop
- `f66c02f` style: remove comments from the integration, evals, tests and scripts
- `8070e56` style: remove comments from the satellite
- `f659bcb` test: forbid comments and docstrings in tracked sources
