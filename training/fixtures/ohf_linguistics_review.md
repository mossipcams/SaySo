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
`5004843f27656eb538ccd03b35988eb0169bc7cfb47421edcd1f6bb67549e0bb`.

## Audited sample

| Measurement | Result |
|---|---:|
| Accepted rows | 2,000 |
| Distinct requests, ignoring case | 1,873 |
| Action/state-call rows | 1,868 |
| No-call rows | 132 |
| Rows using OHF grammar | 898 |
| OHF source files actually selected | 21 |
| OHF source/block/template combinations | 56 |
| Rows using an explicit fallback | 976 |
| Multi-call rows | 115 |
| Exclusion rows | 18 |
| Actual alias rows | 22 |
| Conversational rows | 971 |

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
| Verbless scene fragments (`can you my garage bright lights on for me?`) | 59 | 0 |
| `change to scene the X` parser word order | 23 | 0 |

The double space was the worst of these: it marked every article-dropped row with
a token the model could learn instead of the phrasing. Candidates violating a
generation rule are resampled rather than dropped, so OHF coverage held (890 to
898 rows) while the defects went to zero.
`test_generated_requests_are_grammatical_english` locks these in.

Counts for the verbless and `to scene` rows are from a 6,000-row sample at seed
777, where they are frequent enough to measure; the rest are from the 2,000-row
sample above.

## Known limits of this pass

These are corpus-shape problems, not grammar defects, and are out of scope here:

- **State questions are one frame.** `GetLiveContext` is 22.5% of rows and every
  one of them is built from "the status of {name}" — 11 phrasings before STT
  noise. `HassGetState` is excluded from `ohf_extract.INTENTS`, so the natural
  forms ("is the kitchen light on", "which lights are on in the kitchen",
  "how many lights are on upstairs") never appear. Adopting it needs a free
  `{state}` slot that carries no `GetLiveContext` argument, which the current
  "every slot must bind an argument" contract rejects.
- **Media, vacuum, fan and volume calls have 4-39 frames each** and one phrasing
  covers 23-57% of their rows; `HassVacuum*` is also excluded from extraction.
- **Assistant replies are 34 distinct forms, 71% of them the literal "Done."**
  Nothing in the corpus confirms *what* was done, so the response side is far
  narrower than the request side.
- HVAC modes are spoken raw ("the thermostat is heat"), and clarification
  replies never name the candidates ("Which device did you mean?").

The adapter rejects recognition-only fragments before conversational framing,
protects literal names from grammar cleanup, and handles articles before
possessive names and timer duration agreement. Unsupported slot combinations
retain the semantic fallback. Grammar variety and these audits do not establish
model accuracy; that still requires the unchanged held-out model evaluations.
