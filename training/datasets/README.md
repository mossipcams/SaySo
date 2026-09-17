# Generated datasets live here (gitignored except this README).

Do not commit large generated JSONL files.

Typical local outputs:

- `sayso_train_first_10000.jsonl` — deterministic 10k train
- `sayso_train_supplement.jsonl` — corrective 500–800 rows
- `sayso_shadow_eval.jsonl` — 100–150 shadow eval rows
- `sayso_test_balanced.jsonl` — 2,500 held-out prompts (do not train on these)
- `*_render.jsonl` — TRL dict-argument views of the same rows

Canonical model-eval cases live in `evals/cases/`, not here.

Generators:

- `python training/scripts/build_synthetic_dataset.py`
- `python training/scripts/generate_training_supplement.py`
- `python training/scripts/generate_balanced_test_data.py`
- `python training/scripts/split_dataset.py`
