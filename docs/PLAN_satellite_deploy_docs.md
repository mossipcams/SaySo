# Plan: document the satellite deploy

## Scope

Record the deploy procedure found while moving the Pi to LVA v1.1.18. The README
says the venv must be rebuilt from the patched tree but not how, and plain
`uv sync` breaks the satellite.

Docs only. No code, tests, or config.

## Files

- `satellite/README.md`: add a `## Deploying` section directly before `## Commands`.
- `docs/PLAN_satellite_deploy_docs.md`: this plan.

## Section text

````markdown
## Deploying

`/opt/sayso-satellite` is not a git checkout. Deploy from a clone on the
satellite, such as `~/SaySo`:

1. Run `git pull --ff-only origin main` in the clone.
2. Copy `satellite/sayso/`, `satellite/patches/`, `pyproject.toml` and `uv.lock`
   into `/opt/sayso-satellite/`. Leave `models/` alone.
3. Rebuild `/opt/sayso-satellite/linux-voice-assistant` at the tag in
   `pyproject.toml`, then apply `patches/0001-*` and `patches/0002-*` in order with
   `git apply`. Build it beside the live tree and swap it in, so a failed build
   leaves the running code untouched.
4. Update dependencies from `/opt/sayso-satellite`:

   ```bash
   uv sync --frozen --inexact --no-install-package linux-voice-assistant
   ```

   Do not run plain `uv sync` here. It replaces the patched LVA install with the
   unpatched tag from `uv.lock`, and the satellite no longer starts.
5. Restart with `sudo systemctl restart sayso-satellite`.

Importing LVA opens the audio server, so checks run outside the service need the
same `XDG_RUNTIME_DIR` and `PULSE_SERVER` values as the service.
````

## Verification

- `ruff check .` and `ruff format --check .`
- `pytest tests evals/tests --ignore=tests/test_realistic_v3.py`
- `pytest -q satellite/sayso` against v1.1.18 with both patches applied
