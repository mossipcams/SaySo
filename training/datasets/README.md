# Generated datasets live here (gitignored except this README).

Run `python training/scripts/generate_dataset.py` after pinning upstream with
`python training/scripts/pin_upstream.py`.

Expected outputs after generation and adaptation:

- `home_llm_v2_english_small.jsonl` — adapted Home-LLM V2 source
- `sayso_train.jsonl`, `sayso_val.jsonl`, `sayso_test.jsonl` — leakage-resistant splits
- `sayso_test_balanced.jsonl` — deterministic balanced held-out test set
- `synthetic_v2/sayso_curated_10000.jsonl` — first 10k curated train set from `build_synthetic_dataset.py` (gitignored)

Do not commit large generated JSONL files.
