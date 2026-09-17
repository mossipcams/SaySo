# SaySo evaluations

One case schema, one execution pipeline, one scorer. Model adapters only
invoke the model. Rendering, parsing, and validation stay in production
modules (`sayso_contract` / `custom_components/sayso`). Evaluation never
executes Home Assistant actions.

```text
python -m evals.cli run --suite smoke --adapter endpoint --server http://127.0.0.1:8080
python -m evals.cli run --suite promotion --adapter endpoint --server http://127.0.0.1:8080
python -m evals.cli run --category ambiguity --adapter endpoint --server http://127.0.0.1:8080
python -m evals.cli run --tag grounding --adapter endpoint --server http://127.0.0.1:8080
python -m evals.cli run --case-id <id> --adapter endpoint --server http://127.0.0.1:8080
```

`--suite smoke` is checkpoint selection (~24 promotion cases). `--suite
promotion` is the locked original 120 realistic cases (10 per category).
Diagnostic flags (`--category`, `--tag`, `--case-id`) filter the same
cases; they are not a third runner.

Training-time smoke uses `evals.adapters.InMemoryAdapter` through
`evals.runner.evaluate`. Validation loss remains a trainer metric.

Each run writes `evals/results/<run-id>/` (`metadata.json`, `summary.json`,
`outcomes.jsonl`, `raw/`). Historical scores keep the
`production_contract_fingerprint` recorded at that run.

Do not train on case IDs or utterances from `evals/cases/`. `evals/archive/`
is historical evidence and is never scanned for cases.
