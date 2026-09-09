# OHF deterministic linguistics review

Validated on 2026-09-09 against the repaired training baseline `1883de8`.
This is a generator review, not a model-quality evaluation or a replacement for
the running training corpus.

## Reproduce

Install the training dependencies (including `hassil==3.12.0`), then run:

```sh
PYTHONPATH=training python -m pytest -q training/tests training/scripts training/evals training/generators
PYTHONPATH=training python -m generators.cli --count 2000 --seed 20260909 --output /tmp/sayso-ohf-verified.jsonl
```

All **299 tests passed**, including cross-process reproducibility, recipe-lock
coverage, shadow-generator validation, alias and scope preservation, multi-call
rendering, combined settings, timer durations, and negative-example checks.
No existing test assertions were removed or weakened. The imported grammar JSON
was reproduced exactly from its pinned upstream checkout.

Canonical sample SHA256:
`4cfe69e3d066d58579f6f5f0a9e03cbef74aa4c5367181926bd00b9ffd9385c6`.

## Audited sample

| Measurement | Result |
|---|---:|
| Accepted rows | 2,000 |
| Distinct requests, ignoring case | 1,898 |
| Action/state-call rows | 1,863 |
| No-call rows | 137 |
| Rows using OHF grammar | 890 |
| OHF source files actually selected | 21 |
| OHF source/block/template combinations | 57 |
| Rows using an explicit fallback | 981 |
| Multi-call rows | 162 |
| Exclusion rows | 24 |
| Actual alias rows | 32 |
| Conversational rows | 923 |

OHF and fallback counts can overlap for multi-call requests. State questions use
the explicit GetLiveContext renderer; negative request hints are not included in
the grammar-provenance totals. The audit found no actionable timer refusals,
offered withheld tools, excluded targets called, invalid tool arguments, duplicate
candidate IDs, or normalized overlap with the frozen training eval prompts.

Representative generated requests:

- “please switch off master bedroom ikea light”
- “Could you set my Kitchen ceiling lights to red?”
- “set a timer for 8 minutes called bread”
- “Can you start a timer named cookies for seven minutes for me?”

The adapter rejects recognition-only fragments before conversational framing,
protects literal names from grammar cleanup, and handles articles before
possessive names and timer duration agreement. Unsupported slot combinations
retain the semantic fallback. Grammar variety and these audits do not establish
model accuracy; that still requires the unchanged held-out model evaluations.
