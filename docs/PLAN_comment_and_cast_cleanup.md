# Plan: remove existing comments and loose type casts

User request (2026-11-20): "Remove old comments fix typecasts". Follow-up to PR #212
(Batch 5). The commits land on `ajax/review-batch5` and update that PR.

## 1. Measured scope

Counted with `tokenize` (comments) and `ast` (docstrings) over `git ls-files '*.py'`.
A shebang on line 1 is allowed. Every other `#` comment and every docstring breaks
`AGENTS.md` "No code comments".

| Area | `#` comments | Directive comments (`noqa`, `type: ignore`) | Docstrings |
| --- | --- | --- | --- |
| `custom_components` | 2 | 0 | 0 |
| `evals` | 1 | 0 | 0 |
| `satellite` | 32 | 5 | 10 |
| `scripts` | 25 | 0 | 8 |
| `tests` | 0 | 0 | 1 |

Files: `custom_components/sayso/core/area_context.py`, `evals/tests/test_scalar_domain.py`,
`tests/test_tool_key_order.py`, `scripts/{export_dma_kws_onnx,test_wake_stage_prefix,
test_wake_stage_tts_tv,test_wake_tts_distant,wake_stage_prefix,wake_stage_tts_tv,
wake_tts_distant}.py`, `satellite/sayso/{config,events,test_events}.py`,
`satellite/sayso/wake/{dma_kws_verifier,livekit,test_cascade,test_mining}.py`.
`.sh` files: only shebangs, but the delegate rechecks.

Loose casts in `custom_components/sayso` (added in S5b to satisfy mypy):

| Site | Cast | Honest fix |
| --- | --- | --- |
| `__init__.py:443` | `entry.runtime_data = cast(SaySoRuntimeData, None)` | HA deletes `runtime_data` after a successful unload (`config_entries.py:1044`, HA 2026.8.3). Drop the assignment. Any reader that can see an unloaded or failed entry (`diagnostics.py:124`, unload itself) reads it with `getattr(entry, "runtime_data", None)` |
| `conversation.py:253` | `cast("list[dict[str, Any]] \| None", tools)` | Widen the engine, `client.py` and core `prefix_hash` signatures to `Sequence[Mapping[str, Any]] \| None`, since the runtime value is the compiled tuple |
| `conversation.py:646` | `cast(BaseException, failure.error)` | Narrow `failure.error` (`BaseException \| None`) with an explicit branch |
| `repairs.py:70,74` | `cast(FlowResult, ...)` | Make `DownloadSteps` generic over its flow context and result types, so the repair flow and the config/options flows each get their own type |
| `schema.py:20` | `convert = cast(Any, None)` | Declare the optional import's type up front (`Callable[..., Any] \| None`) |
| `schema.py:91` | `cast(Iterable[Any], container)` | Narrow with `isinstance` checks where the validator's container is read |

## 2. Slices

| Slice | Scope | Verify |
| --- | --- | --- |
| C1 | Comments and docstrings in `custom_components`, `evals`, `tests`, `scripts` (10 files) | CI gate |
| C2 | Comments, directives and docstrings in `satellite/sayso` (7 files) | `~/.cache/sayso-venvs/satellite/bin/pytest -q satellite/sayso` plus the CI gate |
| C3 | New `tests/test_no_comments.py`: every tracked `.py` has no comment token (except a line-1 shebang) and no docstring, and every tracked `.sh` has no `#` line except a line-1 shebang. This guard is needed because comments came back after `6c30d14` | the new test plus the CI gate |
| C4 | The six casts above, removed with the honest fix in each row | `python -m mypy custom_components/sayso` and the CI gate, plus a `rg 'cast\('` count. Any cast that has to stay is listed with its reason |

Rules for C1/C2:
- **Strip by token.** Use `tokenize` and `ast`, never a regex, so `#` inside strings is safe.
- **Empty bodies.** A body left empty after a docstring is removed gets `...` for a protocol or abstract stub, and `pass` otherwise.
- **Comments that carry information.** Where a comment explains non-obvious code, rename or restructure instead, as `AGENTS.md` says.
- **Directives.** A `# noqa: E402` after a path tweak becomes an ordinary import order, or the lines are left as they are with no comment. Satellite is outside ruff, so no lint enforces it. A `# type: ignore` becomes a typed `getattr` or `cast` at that test site.

## 3. Verification (every slice)

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/pytest tests evals/tests --ignore=tests/test_realistic_v3.py -q
.venv/bin/pytest tests/test_realistic_v3.py -k perfect_run -q
~/.cache/sayso-venvs/satellite/bin/pytest -q satellite/sayso
.venv/bin/python -m mypy custom_components/sayso
.venv/bin/lint-imports
.venv/bin/python -m evals.generate_realistic --check
```

Then push `ajax/review-batch5` (no force) and update the #212 body.

## 4. Risks

- The deployed satellite is synced from `~/SaySo` (`/opt/sayso-satellite`). Removing comments doesn't change runtime behaviour, and no redeploy is needed.
- Wake scripts under `scripts/` are run on the llm VM from the ajax-optimization copy, which has diverged. Only this repo's copies change.
