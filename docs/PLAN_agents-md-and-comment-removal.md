# Plan: AGENTS.md router contract, no-comments rule, comment/docstring strip, GitNexus removal

User request (2026-09-30):
1. Update `AGENTS.md` to always use the Ajax Model Router, like `ajax-cli/AGENTS.md`.
2. Add a "never add code comments" rule to `AGENTS.md`.
3. Remove all code comments from the repo (user confirmed: docstrings too).
4. Remove GitNexus everywhere and remove the `impeccable` skill from pi.

## Scope

### SaySo repo (this worktree) — routed through model-router, delegate-executed

- `AGENTS.md`:
  - Add a **Delegation** section mirroring `ajax-cli/AGENTS.md`: all
    implementation writes go through the `model-router` skill (one `EXECUTION`
    decision, delegate implements, parent reviews the delta); no native
    subagents; bypass only with explicit per-request user approval; PR
    requests route through the delegate with the repo verification gate.
  - Add a **No code comments** rule: never add `#` comments or docstrings to
    code; existing comments are being stripped by this change.
- Comment/docstring strip, all `.py` and `.sh` files in the repo:
  - Remove every `#` comment (full-line and trailing) except shebang lines
    (`#!...` at line 1).
  - Remove every docstring (module/class/function first-statement string).
  - `# type: ignore` and `# noqa` included — CI runs only ruff with
    `select = ["E9", "F63", "F7", "F82"]` (syntax errors), so none are
    load-bearing.
  - Implementation must be AST/tokenize-based (Python `ast` + `tokenize`),
    not regex, so `#` and `"""` inside string literals are never touched.
  - No behavior change: code, strings, and formatting of kept lines unchanged.
- Not in scope: `.md`, `.toml`, `.yaml`, `.json` files; eval case content;
  commits in this repo (user did not ask).

### ajax-cli repo — routed through model-router, delegate-executed, PR requested

- Remove the `<!-- gitnexus:start -->…<!-- gitnexus:end -->` block from
  `AGENTS.md`.
- Remove GitNexus references from `CLAUDE.md`.
- Delete `ajax-cli/.claude/skills/gitnexus/` (7 skill dirs) and
  `ajax-cli/.gitnexus/` (index).
- Delegate commits, pushes, and opens the PR with `scripts/gh-pr-create`
  (user explicitly requested a PR).
- Historical mentions in `ajax-cli/.planning/agent-plans/*.md` are past
  records — left untouched.

### Workspace level — parent does directly (no repo writes)

- Delete `~/.agents/skills/gitnexus-*` (8 skill dirs) and
  `~/.agents/skills/impeccable` (removes both from pi's loaded skills).
- Remove GitNexus MCP config entries from `~/.claude/settings.json` and
  `~/.codex/config.toml` after inspecting them.

## Files to touch

- SaySo: `AGENTS.md`, all `*.py` (238 files), all `*.sh` (4 files), this plan.
- ajax-cli: `AGENTS.md`, `CLAUDE.md`, `.claude/skills/gitnexus/**`, `.gitnexus/**`.
- Workspace: `~/.agents/skills/gitnexus-*`, `~/.agents/skills/impeccable`,
  `~/.claude/settings.json`, `~/.codex/config.toml`.

## Verification

- SaySo (delegate):
  - `python -m pytest` (full suite: `tests/`, `evals/tests/`, `satellite/sayso`).
  - `ruff check .`
  - AST/tokenize sweep: zero `COMMENT` tokens (other than shebangs) and zero
    docstrings remain in any `.py` file.
  - `git diff --stat` sanity: only comment/docstring/AGENTS.md lines changed.
- ajax-cli (delegate):
  - `grep -ri gitnexus AGENTS.md CLAUDE.md` returns nothing; both dirs gone.
  - Docs-only change: no compiler/lint gate required per ajax-cli AGENTS.md.
- Parent: risk-proportional review of both deltas before acceptance.
