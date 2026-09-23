# Generate v4 40k corpus

Scope: run `generators.cli` with `training/configs/generation/full_sft_v4.yaml`. Write `training/datasets/sayso_full_sft_v4_20260922.jsonl` + manifest. No training launch, no eval/runtime edits.

## Why now

Generator changes for #94 (bare room phrasing) and #102 (letter-spaced TV STT) are in the tree. The recipe already has junk (#97), per-tool floors (#39), `bare_name_rate`, and real-home mix.

## Command

```bash
cd training && .venv/bin/python -m generators.cli --config configs/generation/full_sft_v4.yaml
```

## After it finishes

Audit allocations, GetDateTime floor, junk share, rows with gold `name="TV"` plus room phrasing, rows whose utterance is `T V`/`T.V.`/`teevee` with gold still `TV`. Record in `training/TRAINING_LOG.md` + `training/README.md` only if generation succeeds.

Kill the process if it burns the attempt budget without writing artifacts (previous 40k hang).

## Real-home quota vs family gold

When balanced real-home mixing forces `P=remaining_quota/rows_remaining` near 1.0 on
late ordinary slots, `gold_matches_family` can fail on the fixture home. The pipeline
retries that slot once with `real_home_selected=False` (same attempt counter) so
leftover ordinary fills use synthetic homes without lowering `real_home_rate` in YAML.

## 2026-09-22 generate attempt

40k hung at ~25MB RSS / 100% CPU. Killed. 8k dry-run then failed: **7968/8000**, shortfall **absence 254/286**, `duplicate_semantic_id` 151419. Unavailable/unsupported/junk/datetime/absence now stay synthetic via `REAL_HOME_EXCLUDED_FAMILIES`; retry 40k after verification. Do not shrink family shares.
