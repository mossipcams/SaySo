# 40k Grounded Gauntlet execution plan

## Scope

Repair the deterministic 40,000-row generation failure, build and audit the
replacement corpus on `192.168.1.140`, and launch a fresh Base rsLoRA run only
after every existing data and overlap gate passes. Keep the 15-row gauntlet
evaluation held out and preserve prior datasets and checkpoints.

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
