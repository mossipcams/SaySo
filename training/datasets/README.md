# Generated datasets live here (gitignored except this README).

Do not commit generated JSONL files. SaySo training data is generated and
promoted from ajax-optimization `sayso-training/`; the manifests here record
older corpora.

Older and development outputs may also be present:

- `sayso_train_first_10000.jsonl` — deterministic 10k train
- `sayso_train_supplement.jsonl` — corrective 500–800 rows
- `sayso_shadow_eval.jsonl` — 100–150 shadow eval rows
- `sayso_test_balanced.jsonl` — 2,500 held-out prompts (do not train on these)
- `*_render.jsonl` — TRL dict-argument views of the same rows

Canonical model-eval cases live in `evals/cases/`, not here.

Other data utilities (not the full-training entry point):

- `python training/scripts/split_dataset.py`
