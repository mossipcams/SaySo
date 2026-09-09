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

All **322 tests passed**, including cross-process reproducibility, recipe-lock
coverage, shadow-generator validation, alias and scope preservation, multi-call
rendering, combined settings, timer durations, and negative-example checks.
No existing test assertions were removed or weakened. The imported grammar JSON
was reproduced exactly from its pinned upstream checkout.

Canonical sample SHA256:
`746caf87e28698584d10e8783b41345e17e1a7f796966eb88160b8699d9651ec`.

## Audited sample

| Measurement | Result |
|---|---:|
| Accepted rows | 2,000 |
| Distinct requests, ignoring case | 1,874 |
| Action/state-call rows | 1,869 |
| No-call rows | 131 |
| Rows using OHF grammar | 901 |
| OHF source files actually selected | 21 |
| OHF source/block/template combinations | 60 |
| Rows using an explicit fallback | 974 |
| Multi-call rows | 120 |
| Exclusion rows | 14 |
| Actual alias rows | 24 |
| Conversational rows | 974 |

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

## Generation-grammar defects removed

The upstream grammars are written for *recognition*, so their optional articles,
prepositions and units are independently droppable. Sampling them freely produced
recognition-valid but ungrammatical requests. Measured over the same 2,000-row
sample, before and after this pass:

| Defect | Before | After |
|---|---:|---:|
| Double space left by the dropped-article STT transform | 196 | 0 |
| `a`/`an` disagreement (`an timer`, `set an 9-minute timer`) | 9 | 0 |
| Domain ids spoken as nouns (`switchs`, `climates`) | 7 | 0 |
| Missing preposition (`temperature Kitchen Light to 5000`) | 5 | 0 |
| Bare `k` unit (`2700 k`) | 9 | 0 |
| Degree sign in spoken text (`sixty nine °`) | 15 | 0 |
| `create the timer` instead of `create a timer` | 7 | 0 |
| `make X to blue` / `set X blue` value binding | 30 | 0 |

The double space was the worst of these: it marked every article-dropped row with
a token the model could learn instead of the phrasing. Candidates violating a
generation rule are resampled rather than dropped, so OHF coverage rose (890 to
901 rows, 57 to 60 templates) while the defects went to zero.
`test_generated_requests_are_grammatical_english` locks these in.

The adapter rejects recognition-only fragments before conversational framing,
protects literal names from grammar cleanup, and handles articles before
possessive names and timer duration agreement. Unsupported slot combinations
retain the semantic fallback. Grammar variety and these audits do not establish
model accuracy; that still requires the unchanged held-out model evaluations.
