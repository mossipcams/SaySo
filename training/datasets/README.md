# Generated datasets live here (gitignored except this README).

Do not commit large generated JSONL files.

Typical local outputs:

- `sayso_quality_eval_recipe_lock.jsonl` — 38 locked gold cases
- `sayso_quality_eval_v3_gold.jsonl` — v3 locked gold cases
- `sayso_quality_eval_v3_shadow.jsonl` — v3 shadow eval rows
- `sayso_train_first_10000.jsonl` — deterministic 10k train
- `sayso_train_supplement.jsonl` — corrective 500–800 rows
- `sayso_shadow_eval.jsonl` — 100–150 shadow eval rows
- `sayso_test_balanced.jsonl` — 2,500 held-out prompts (do not train on these)
- `*_render.jsonl` — TRL dict-argument views of the same rows

Generators:

- `python training/scripts/generate_recipe_lock_eval.py`
- `python training/scripts/generate_v3_quality_eval.py`
- `python training/scripts/generate_training_supplement.py`
- `python training/scripts/generate_balanced_test_data.py`
