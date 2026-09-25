# v6 dataset: diagnose first, then targeted data quality (plan)

Date: 2026-09-25. Replaces the earlier v6 draft (LLM generation, full rewrite),
which failed review:
- It skipped diagnosis.
- It had no trustworthy yardstick.
- It used an LLM to generate and judge (circular, and ~1.6B prompt tokens).
- It overclaimed constrained decoding and the HA oracle.
- It had no fast iteration loop.

Data comes from the in-repo generator only (no LLM generation or judging).

## Goal

Accurate tool calls across a wide range of commands, where quality means rows
that are correct, unambiguous, realistic, informative and leak-free. Every change
has to show up on a trustworthy dev measurement before it ships.

## Step 1: diagnose v5b (as soon as training ends, ~11:40 CDT)

1. Promotion 120 (already queued in the overnight chain) plus a failure
   breakdown by type (wrong device, act-instead-of-ask, refusal, argument
   convention, parse).
2. **In-distribution check:** 300 held-out v5 rows (same generator and recipe,
   new seed, no prompt overlap with train), scored on the first supervised
   turn (decision class, tool, exact arguments).
   - High in-distribution, low eval → the gap is data distribution
     (realism/coverage): data work pays off.
   - Low in-distribution too → the model isn't learning the task
     (230M × ~5k-token prompts). Then prompt length, model size or training
     settings come before data.
3. Read the failures (per the verify-before-reporting rule) before naming causes.

## Step 2: a yardstick we can trust

- **Outcome grading** next to exact-argument grading, so equivalent calls
  pass and convention drift is separated from real mistakes (the legacy
  `eval_outcome.py` approach).
- **Dev set for checkpoint selection**, never the locked suite. The smoke suite
  (a subset of promotion) stops being used for selection.
  - **val-ID:** held-out generator rows (the Step 1 set, fixed).
  - **val-OOD:** generator rows built from held-out vocabulary pools (nicknames,
    owner names, brands, phrasings, rooms) split 80/20 before generation, plus
    the real home's holdout entities.
  - **real-dev:** the ~130 real satellite command transcripts, hand-labeled.
- **Locked test:** promotion 120 + regressions, used for promotion only.

## Step 3: fast ablations

A 10k-row, 1-epoch recipe (~2h on the XTX) and the fixed dev sets. Each data
change is tested against the same baseline recipe before any full 40k run.

## Step 4: targeted generator fixes, one at a time

Ordered by what Step 1's failures show. Current candidates, each tied to an
observed failure:

| Candidate | Evidence | Generator change |
|---|---|---|
| Nickname aliases | 98% of alias rows use word-overlap aliases; eval aliases are nicknames | nickname vocabulary + alias_distractor prefers nicknames (drafted as v5c, uncommitted) |
| Settings-request ambiguity | clarify drew only on/off/status; 6 of 10 eval ambiguity cases are settings requests | settings ops in clarify/follow_up (drafted as v5c) |
| Relative adjustments | "a little brighter", "turn it down" absent from templates | extend the OHF-backed grammar with relative brightness/volume/temperature |
| Phrasing variety | 15% "can you", 14% "could you"; few questions, fragments or first-person | more OHF sentence templates and phrasing transforms in `utterances.py` |
| Real STT errors | real captures show "t v", "liver room", "pd", "turn out" | error model in `stt_noise.py` fitted to the labeled real commands |
| Junk realism | v5 junk is templated fillers; real false wakes are background speech | junk built from real non-command transcript shapes (lengths, fragments) |
| Tool results | post-tool turns see `{"result": "Success"}` | real HA intent/GetLiveContext response format from integration traces |

Checks for every change:
- **Resolver consistency:** one decision function mirroring HA semantics,
  including the exact-name rule and locks without a device class.
- **Shortcut probe:** a classifier on non-semantic features must be near chance.
- **Human review:** a 100-row stratified sample, about 15 minutes.
- **HA matching** as an executability check (confirms a call resolves, not that
  it's what the user meant) where it's cheap to add.

Flip-one-fact siblings and scene reuse are candidates like the others. They
ship only if an ablation shows they move the dev set.

## Step 5: only if Step 1 points past the data

Prompt compaction (a contract change, must be trained on), model size, training
settings. Constrained decoding (GBNF from the offered tools, arguments and
entity names) is a separate small integration change. It prevents invalid
outputs (made-up arguments or devices), not valid-but-wrong choices, so it
won't fix most of the observed failures.

## Later: real-usage loop

Integration traces → review/label → real-data set (dev/test first; train only
with approval).

## Needs your OK

- Hand-labeling the ~130 real command transcripts (I pre-label from the
  generator's resolver where it applies; you review).
- Whether real transcripts may shape training style/noise, or stay dev/test only.
- Installing HA in a new test venv, if the executability check is wanted.
