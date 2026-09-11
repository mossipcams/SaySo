# 40k Grounded Gauntlet execution plan

## Scope (v4)

Repair the deterministic 40,000-row generation failure, build and audit the
replacement corpus on `192.168.1.140`, and launch a fresh Base rsLoRA run only
after every existing data and overlap gate passes. Keep the 15-row gauntlet
evaluation held out and preserve prior datasets and checkpoints. This run is
referred to as v4 to mark the regenerated, gated corpus on
`sayso_40k_grounded_gauntlet_20260911`.

## Files

- `training/generators/duplicates.py`
- `training/generators/test_coverage_gates.py`
- Named host bundles using `sayso_40k_grounded_gauntlet_20260911`

## Implementation

Bound repeated semantic scenarios with the existing duplicate limit instead of
rejecting the second distinct phrasing. Keep exact utterance/home caps, verify
locally, generate and audit the named remote corpus, then reuse the proven Run
012 guarded launcher with a fresh output directory.

## Verification

- Focused and complete relevant training suites pass.
- Two 1,200-row smoke generations are byte-identical.
- Remote canonical and rendered files contain exactly 40,000 rows.
- Quotas, 10% real-home mixing, grounding families, schemas, and held-out
  overlap checks pass.
- Token audit and 12-step GPU smoke pass before the Base trainer starts.

## Current state (post-regeneration)

- Canonical SHA-256: `c181ca96eb1a584142ea9605f56819cc7ef8731f86b71234dacd73b7abf47a15`.
- Render SHA-256: `8fde3c75faf28b0c79c7c85f61ebfa8a5601f948c418e2b68b1f59682a4f9ccc`.
- Manifest SHA-256: `bdb08d8c1d3065d29e448d538601713df7e34c5af2a6350082b36e6592307a04`.
- Quotas: no shortfall; `uncovered_operations: []`; 16 grounding families present
  (88 rows total, requested 3% achieved 0.22%); real-home rows 4,000 at the
  requested 10% achieved 10.0%; 35,642 tool calls positive; 4,871 negative.
- Held-out gauntlet eval (15 rows, dated 2026-09-11 02:24) remains untouched.
- Launcher script `/srv/training-runs/run-40k-grounded-gauntlet-train.sh`
  already exists and guards on the new corpus path.

## Next steps

1. Reuse Run 012's launcher, token-audit script, and 12-step GPU smoke.
2. Run the held-out overlap check (gauntlet eval + frozen realistic + recipe
   lock) against the v4 corpus.
3. Run the token audit and the 12-step GPU smoke.
4. Launch the Base trainer and capture results in `TRAINING_LOG.md`.

## v4 launch — 2026-09-11 18:47 UTC

All gates passed and the Base rsLoRA trainer is running on `192.168.1.140`.

**Data and bundle**

- Data: `/srv/datasets/sayso_40k_grounded_gauntlet_20260911/`
  - Canonical SHA-256: `c181ca96eb1a584142ea9605f56819cc7ef8731f86b71234dacd73b7abf47a15`
  - Render SHA-256: `8fde3c75faf28b0c79c7c85f61ebfa8a5601f948c418e2b68b1f59682a4f9ccc`
  - Manifest SHA-256: `bdb08d8c1d3065d29e448d538601713df7e34c5af2a6350082b36e6592307a04`
- Bundle: `/srv/training-runs/sayso_40k_grounded_gauntlet_20260911-source/`
  - Launcher: `launch.sh` (Run 012 guarded launcher, fresh corpus and output dir)
  - Source commit: `6fdd46b`
  - Token audit: `token-audit.json` (40,000 rows, max 3,795 tokens)
  - Held-out audit: `audit_heldout_overlap.txt` (PASS, 0 overlap across 4 eval sets)

**Gates passed**

- Render hash matches token audit; 40,000 rows; max 3,795 ≤ 8,192; no truncation.
- 11 quotas: no shortfall; `uncovered_operations: []`; real-home rate requested
  10.0% achieved 10.0%; 16 grounding families present.
- Held-out overlap check PASS: gauntlet eval (6 utterances), recipe-lock
  (35), v3 gold (35), v3 shadow (100) — 0 overlap each.
- 12-step GPU smoke on the longest 3,795-token row: finite losses (1.362 →
  0.001071), finite grad norm after initial FP16 backoff, nonzero LoRA B
  weights.

**Trainer status**

- Recipe: Base, rsLoRA rank 32, FP16/SDPA, lr 2e-4 cosine, batch 1, accum 16,
  2 epochs (5,000 optimizer steps), save every 250 steps, retain 20
  checkpoints, assistant-only loss.
- Output: `/srv/training-runs/SaySo-LFM2.5-230M-40k-grounded-gauntlet/`
- PID: `162739`, started 2026-09-11 18:47:45 UTC.
- Throughput: ~25 s / optimizer step on GTX 1070 (4,827 MiB / 8 GiB peak).
- Convergence (TensorBoard at step 40): loss 0.0780, grad_norm 3.20,
  mean_token_accuracy 0.977, entropy 0.072, learning_rate 2.0e-4.
- ETA: ~34.7 hours total → ~2026-09-13 05:30 UTC.

**Next**

- Watch the next 250-step checkpoint (`step-250`) to confirm FP16 stability.
- After completion, evaluate both epoch checkpoints (step 2,500 and 5,000)
  on gauntlet eval, recipe-lock, v3 gold, and v3 shadow; no automatic
  promotion.
