# Queue LFM v5 corpus generation on the LLM VM behind the running wake-word generation

Date: 2026-09-23. Scope: host job orchestration on `ssh llm`, plus one bounded
generator bug fix (blocker found while verifying) + regression test, landed via
branch/PR so the VM can pull it. No wake-pipeline changes, no training launch,
no eval changes.

## Context (observed on the VM, 2026-09-23 ~23:16)

- Wake-word generate (LiveKit, CPU) is running:
  - PID `151804`: `python -m livekit.wakeword generate /srv/llm/wake/runs/livekit-restart-20260923/cpu-gen/w1-train.yaml`
  - At 23:16 it was at 43% of `negative_train` (2000 clips, ~1.9 clip/s) → ETA ~23:26-23:35.
  - `w2-val` (the second generate) already finished 23:10:37 ("Generation complete!").
  - GPU is held by vLLM (`gpu status` → `serve`); the wake run is intentionally CPU-only.
- LFM data generation = the in-repo generator per `training/README.md` "v5 full SFT on the
  ROCm host": run `python -m generators.cli --config configs/generation/full_sft_v5.yaml`
  on the VM with `HIP_VISIBLE_DEVICES=` (CPU only, XTX hidden), then export the rendered
  prompt/completion view with `scripts/export_llamafactory_view.py`.
- The generator code (v5) is on `origin/main` @ `359e4f8` (PR #104). The VM checkout
  `/srv/llm/sayso` (→ `/srv/llm/data/sayso`) has HEAD `cfdbc70` (main, stale) plus a
  **mixed, partially applied working tree**: untracked v5 files identical to
  `origin/main`, and tracked files (e.g. `training/generators/planning.py`,
  `row_generation.py`) still at the older pre-PR content. Running generation from that
  mixed tree is unsafe.
- `training/.venv` (265 MB) exists; a generator smoke ran successfully there today
  (`training/datasets/sayso_generator_smoke.jsonl`, 2026-09-23 12:51).

## Blocker found while verifying (2026-09-23 ~23:30)

The committed v5 generator (`origin/main` @ `359e4f8`) crashes deterministically at
plan time, on Mac and VM (identical code):

```text
generators/planning.py:199 _pick_capability_operation
    cap_name = rng.choice(tier_caps)
IndexError: Cannot choose from an empty sequence
```

The `ordinary` branch of `_pick_capability_operation` draws a tier from
`TIER_PROPORTIONS` ({1: 0.80, 2: 0.15, 3: 0.05}) but filters that tier's
capabilities to those with an *action* op (`_action_ops`: tool-backed, not a
settings op, not `query_state`). Tier 3 (`lawn_mowers`, `todo_lists`,
`buttons`) has no action-backed ops, so a tier-3 draw (5% of ordinary, and
ordinary is 22% of the v5 recipe) leaves `tier_caps` empty and crashes. The
VM's 12:51 smoke only passed because it ran the older pre-PR `planning.py`,
which had an unsafe fallback to the capability's first operation; PR #104's
stricter filter removed the fallback without excluding the empty tier. The
crash is deterministic within a few draws (tier-3 weight 0.05), so every
40k run (and every `--dry-run`) fails.

### Fix (implemented directly in the worktree — user said: don't delegate)

- `training/generators/planning.py`, `ordinary` branch only: draw the tier
  only from tiers that actually contain at least one capability with an
  action op (same `_action_ops` predicate), keeping `TIER_PROPORTIONS`
  relative weighting among eligible tiers (renormalized over {1, 2}).
  Tier-3 capabilities (settings/query-only) simply do not appear in the
  ordinary draw — the intent the existing comment already stated ("a slot
  whose only op is query_state ... gets re-picked forever"), minus the
  crash. No other family changes; no registry changes.
- Regression test (new file `training/tests/test_planner_tier_draw.py`):
  hammer `_pick_capability_operation("ordinary", rng)` across seeds/draws and
  assert the chosen capability always has an action op (pre-fix code fails
  within the first ~20 draws).

Verification (local):
1. `cd training && .venv/bin/python -m generators.cli --config configs/generation/smoke.yaml --dry-run` → exit 0.
2. `python3 -m pytest training/tests/ -q` → green.
3. Full v5 recipe dry-run `configs/generation/full_sft_v5.yaml --dry-run` → exit 0 (40k plan builds).

Landing: commit the fix + test + this plan on `ajax/training-data` (worktree
content == origin/main; PR #104 lineage), push, `gh pr create --base main`,
merge; VM `git reset --hard origin/main`, re-verify dry-run, then arm the
gated run below.

## Plan

1. **Backup the drift** (HDD): copy the 68 changed/untracked files from
   `/srv/llm/sayso` to `/srv/llm/data/archive/pre-v5gen-20260923/` preserving paths.
2. **Sync the VM checkout to `origin/main`** (fetched, `359e4f8`):
   `git reset --hard origin/main` in `/srv/llm/sayso`. The in-flight wake job reads only
   from `/srv/llm/wake/runs/...` and the wake venv — it does not read the repo at
   runtime (config loaded at start), so the reset is safe while it runs.
   Verify: `git status --porcelain` empty; quick generator smoke in the venv
   (`python -m generators.cli --config configs/generation/smoke.yaml --dry-run`).
3. **Install a one-shot watcher** on the VM
   (`/srv/llm/lfm/runs/v5-gen-20260923/wait-wake-and-gen.sh`), started detached
   (`setsid nohup`), that:
   1. Polls every 30 s until PID `151804` exits (confirm twice, 5 s apart; 12 h cap).
   2. Fail-closed: requires "Generation complete!" in
      `cpu-gen/w1-train.log`; otherwise writes an ABORT note to the log and exits 0.
   3. Runs LFM v5 generation (CPU only):
      `cd /srv/llm/sayso/training && HIP_VISIBLE_DEVICES= .venv/bin/python -m generators.cli --config configs/generation/full_sft_v5.yaml`
      → `training/datasets/sayso_full_sft_v5_20260922.jsonl` (+ manifest).
   4. Exports the rendered view:
      `HIP_VISIBLE_DEVICES= .venv/bin/python scripts/export_llamafactory_view.py datasets/sayso_full_sft_v5_20260922.jsonl -o datasets/sayso_full_sft_v5_20260922_rendered.jsonl --dataset-name sayso_v5_rendered --dataset-info-out datasets/dataset_info.sayso_v5_rendered.json`
   5. Copies both artifacts + manifest to the canonical dataset root
      `/srv/llm/data/lfm/datasets/` (per README host layout).
   All steps logged to `/srv/llm/lfm/runs/v5-gen-20260923/lfm-v5-gen.log`.
4. **Verify the watcher is armed** (`ps` + log header) and report.

## Files touched

- Host `llm` only: backup dir, git state of `/srv/llm/sayso`, new
  `/srv/llm/lfm/runs/v5-gen-20260923/{wait-wake-and-gen.sh, lfm-v5-gen.log}`,
  generated artifacts under `training/datasets/` and `/srv/llm/data/lfm/datasets/`.
- Repo: `training/generators/planning.py`, `training/tests/test_planner_tier_draw.py`,
  this plan file. No `context.json` commit. No other code.

## Verification

- Pre-launch: VM `git status` clean at `359e4f8`; smoke `--dry-run` exits 0.
- Watcher armed: process visible, log shows waiting on PID 151804.
- On wake completion: watcher log shows wake-done marker, then generation start;
  progress visible in `training/datasets/sayso_full_sft_v5_20260922.jsonl` (growing)
  and the log tail.
- On completion: manifest row count = 40,000 (accepted), rendered view row count
  matches, artifacts copied to `/srv/llm/data/lfm/datasets/`. Then (separate step,
  after review) record the corpus + sha256 in `training/TRAINING_LOG.md`.

## What this is NOT

- Not a training launch. No GPU is touched; vLLM keeps serving.
- Not a change to the wake pipeline. If w1 fails, LFM generation does not start;
  the wake log + watcher log show why.
- The watcher is single-shot for PID 151804; a manually restarted wake generate
  would not re-arm it.
