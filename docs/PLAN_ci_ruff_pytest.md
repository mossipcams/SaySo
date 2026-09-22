# PLAN: Add ruff and pytest to CI

## Goal

GitHub Actions CI on this repo must run:

- `ruff check .`
- `ruff format --check .`
- `pytest`

on pull requests and pushes to `main`, in addition to the jobs already in
`.github/workflows/ci.yml`.

## Current state

`.github/workflows/ci.yml` already has:

- Conventional PR title check
- Satellite unit tests (`pytest -q satellite/sayso` with linux-voice-assistant patches)
- Home Assistant compatibility matrix tests
- A manual live llama.cpp job

There is no ruff install or ruff config in the repo. Root `pyproject.toml`
defines pytest `testpaths` as `tests`, `evals/tests`, and `satellite/sayso`.

## Scope

- `.github/workflows/ci.yml` — add a job that installs ruff (and test extras as needed) and runs the three commands
- `pyproject.toml` — only if needed to declare ruff as a test/dev extra or a minimal `[tool.ruff]` so the CI install is reproducible

Do not:

- Remove or rewrite the existing satellite, compat, PR-title, or live-llama jobs
- Mass-format or mass-lint-fix the Python tree to make ruff pass
- Change runtime wiring, evals, or training
- Commit `context.json`

## Approach

Add one CI job (split lint vs test only if install needs differ) on
`ubuntu-latest` that:

1. Checks out the repo
2. Sets up Python using the existing pinned `actions/setup-python` action
3. Installs ruff (and `.[test]` / other extras required for collection)
4. Runs `ruff check .`
5. Runs `ruff format --check .`
6. Runs `pytest`

Reuse existing checkout/setup-python pin SHAs. Keep `permissions: contents: read`.

If a root `pytest` cannot collect without the satellite LVA checkout, install
what that collection needs rather than dropping `pytest` or replacing it with
a narrower path. Keep the existing specialized satellite job.

## Files to touch

- `docs/PLAN_ci_ruff_pytest.md` (this plan)
- `.github/workflows/ci.yml`
- `pyproject.toml` (only if ruff extra/config is required)

## Verification

- Workflow YAML still parses and includes the three exact commands
- Ruff is installed in the new job before those commands run
- Existing satellite / compat / pr-title / live-llama jobs keep their current commands
- No runtime Python behavior change
