# SaySo training lifecycle

Run training from the SaySo checkout on the `llm` VM. The local `./sayso`
command wraps the existing v5 generator, static preflight, rendered-view
exporter, Unsloth trainer, and production SaySo scorer. It does not commit code,
publish weights, or deploy a model. See [training/README.md](../training/README.md)
for the observed host layout, [the model training plan](SAYSO_LFM_TRAINING_PLAN.md)
for design constraints, and [TRAINING_LOG.md](../training/TRAINING_LOG.md) for
historical runs and scores.

## Commands

Run these in order for each candidate:

```bash
# Fast CPU-only check while developing a dataset
python scripts/preflight.py

# Candidate -> promoted dataset -> evaluated model
./sayso generate
./sayso validate
./sayso canary
./sayso promote-dataset
./sayso train
./sayso eval
./sayso promote-model
```

`./sayso generate` builds the configured `full_sft_v5.yaml` dataset under
`training/datasets/candidates/<run-id>.jsonl`. Each run also has a manifest and
metadata record. Generation never changes the promoted-dataset pointer.

`./sayso validate` runs `scripts/preflight.py` against that candidate using
`training/configs/preflight.yaml` and the accepted reference in
`training/configs/training_baseline.json`. It checks row and tool-call structure,
pinned tools and targets, positive coverage, negative ratios, duplicates and
conflicts, family-distribution drift, required fixtures, and overlap with eval
utterances. A failed check leaves the candidate unvalidated.

`./sayso canary` requires successful validation. It renders and stages the
candidate, then runs 250 steps through the same Unsloth trainer and training
settings used for full training; only the step limit and output directory differ.
The shared scorer evaluates the smoke suite and records recipe-lock, gold, shadow,
and grounding diagnostics. The smoke gate checks suite requirements and compares
important category and tool-validity rates with the accepted baseline using its
configured tolerance.

`./sayso promote-dataset` succeeds only when validation and canary both passed
for the unchanged dataset. It writes the local promoted-dataset record, including
the run ID, dataset and rendered-data hashes, source commit, recipe, and canary
metrics.

`./sayso train` reads that explicit promotion record, checks the source and
rendered-data hashes, stages the exact rendered dataset, and launches full
training through `/srv/llm/bin/gpu train lfm`. The trainer independently requires
the promotion record and verifies the rendered-data hash before training. If a
newer candidate exists, this command still trains the latest explicitly promoted
dataset, not the unpromoted candidate.

`./sayso eval` scores the completed full checkpoint on the locked 120-case
promotion suite using the shared runner and scorer. The suite enforces its
existing overall and category gates, including media, ambiguity, unavailable,
multi-action, status, and basic tool selection behavior. A failure clears the
final-eval pass state.

`./sayso promote-model` writes the latest model-promotion record only after final
evaluation passes. Promotion records model and dataset run IDs, source commit,
recipe, checkpoint path, and scorer metrics. It does not publish or deploy the
checkpoint.

## Local records and baselines

Candidate datasets are stored under ignored `training/datasets/candidates/`.
Per-run records and the current-run pointer are under ignored
`training/runs/promotion/`. The same directory holds the latest
`promoted_dataset.json` and `promoted_model.json`; each per-run JSON record keeps
the candidate's earlier gate results. These are local operational files, not a
database or Git-tracked approval system.

The accepted dataset distribution and model-eval baseline live in
`training/configs/training_baseline.json`. A run reads this file but never
rewrites it. Update it manually only after reviewing and accepting a dataset or
model; a failed run cannot replace the baseline.

`./train.sh` is a compatibility shortcut for `./sayso train`. It deliberately
fails unless an accepted dataset promotion already exists. Do not launch model
training from the Unsloth Studio UI because that bypasses the GPU lock and
promotion checks.

For code changes that affect generation, training rows, prompt/template
formatting, loss masking, or training configuration, generate a fresh candidate
and run the complete sequence. Unrelated code changes do not require a canary.
