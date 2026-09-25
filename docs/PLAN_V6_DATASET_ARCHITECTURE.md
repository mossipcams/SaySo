# v6 dataset: data quality first (proposal)

Date: 2026-09-25. Status: **proposal for review**. No code until approved. The v5b
run keeps training as the baseline.

## What "quality" means here

A training row is good only if all five hold:

1. **Correct:** the label is what Home Assistant would execute successfully *and*
   what a careful person wants.
2. **Unambiguous:** given the prompt, the label is the one best answer, in one
   canonical form (no convention noise).
3. **Realistic:** the input looks like production (real HA prompt, real homes'
   naming, real speech and real STT errors, real tool results).
4. **Informative:** it teaches something other rows don't (not a near-duplicate,
   not decidable from surface features).
5. **Leak-free:** nothing but the semantics predicts the label.

For a 230M model, label noise and shortcuts cost more than missing quantity:
20k verified rows beat 40k with 10% wrong labels.

## What the evidence says about v5 on each criterion

| Criterion | Evidence (this week) | Status |
|---|---|---|
| Correct | 10% of clarify rows asked "which one?" for an exact name; locks labeled with a device class HA locks don't have; trailing-space names; raw timestamps in replies | **Unmeasured.** Found by hand; the true error rate is unknown |
| Unambiguous | Eval and corpus disagree on conventions (lock args, `floor` added, area vs. names); the exact-arg grader punishes any drift | Conventions live implicitly in code |
| Realistic | See real transcripts below; tool results are faked as `{"result": "Success"}`; synthetic names follow "{Area} {Brand} {Noun}" templates | Largely synthetic |
| Informative | 33,905 throwaway homes for 40k rows; openers are 15% "can you", 14% "could you" | Low contrast |
| Leak-free | Tool count, casing, junk format and alias overlap all predicted the label | Found one at a time, by hand |

### Real usage (satellite STT captures, `live-20260924/stt_capture`)

358 non-empty transcripts from the command path:

- **Only ~36% look like commands** (keyword heuristic: 129 of 358). The rest is
  background conversation and false wakes ("all the workers started sanding…",
  "yeah", "okay"). v5 junk is 6% and
  templated ("uh huh", "right then.").
- **Real STT errors differ from our noise list.** "turn on the **t v**",
  "come on in the **liver room**", "turn the **water tv** on", "the living room
  **pd**", "**try out** the living room tv", "turn **out** the…",
  "**i** turn on…".
- **Real follow-ups exist:** "no, the other one".
- Commands concentrate on living-room lights and the TV.

This is the only real signal we have about inputs, and v5 doesn't use it.

## Plan: measure, then generate with an LLM and verify with HA

### Phase 1: measurement tools, run on the existing v5 corpus first

Nothing gets rebuilt until we know v5's real error rates.

1. **HA oracle (correctness).** Load each row's home into a real Home Assistant
   test instance (`pytest-homeassistant-custom-component`: entities, areas,
   floors, aliases, device classes, supported features, exposure). Run every
   gold call through HA's own intent matching. Pass means it resolves to exactly
   the intended entities and the intent succeeds. For "ask" rows, the spoken
   reference must match 2 or more entities. For "unavailable", the tool is
   absent. For "absent", nothing matches. HA is the authority (AGENTS.md); the
   generator's own reasoning is where the bugs came from.
2. **Independent judge (ambiguity / label noise).** Qwen3.8-27B (already served
   on the VM) gets the prompt and utterance and returns decision + target. It
   never sees our label. Disagreements are sampled for human review; the
   agreement rate is our label-noise estimate.
3. **Recoverability check (noise).** After STT corruption, can the judge still
   identify the target? If not, the row is teaching a guess: relabel it "ask"
   or drop it.
4. **Shortcut probe (leak-free).** Train a small classifier on non-semantic
   features only (tool count, script count, prompt length, casing,
   punctuation, home size, satellite area, family metadata) to predict the
   decision. Target AUC ≤ 0.55. This replaces the ad-hoc audits that found
   leaks one at a time.
5. **Human review pack.** Each build gets 100 stratified rows rendered as
   readable markdown (home summary, utterance, label, oracle/judge verdicts)
   for a 15-minute spot check. Any error found becomes a generator fix and an
   audit rule.

Output: a **data-quality report** per build covering oracle validity %, judge
agreement %, recoverability %, probe AUC, duplication, coverage matrix, and a
realism comparison against real transcripts.

### Phase 2: generation core (the LLM writes the requests, HA decides the labels)

Templates can't produce the long tail of how people talk; they gave v5 its
"can you … for me?" monotony and 2% true nicknames. Verified LLM generation is
the established pattern for function-calling data (e.g. Salesforce APIGen:
diverse generation, then format, execution and semantic checks). Here:

1. **Coverage grid picks what to ask.** Capability × operation × referring style
   (exact name, nickname, area + noun, bare noun, description, whole area, floor)
   × decision (act / ask / refuse-unavailable / refuse-unsupported / absent /
   status / ignore). Exact counts per cell, so rare tools and rare shapes are
   guaranteed, with no rejection loops or probabilistic overlays.
2. **Scene sampler.** Home graph + exposed catalog (realistic withholding) +
   satellite area + timers, independent of the cell's decision. Registry-style
   naming (mixed casing, bare names, model strings like "LEVDS-Switch-92D6",
   owner names, nicknames from a split-able vocabulary). Each scene serves about
   6 cells.
3. **Counterfactual siblings.** For each (scene, cell), the minimal scene edits
   that flip the decision with the utterance fixed: tool withheld, second device
   sharing the alias, device moved out of the area, feature missing.
4. **The LLM writes the utterances.** Qwen3.8-27B writes 5–10 per cell, from the
   cell spec and the scene (not the label). Style guidance comes from the real
   satellite transcripts. Required spread: terse, polite, questions, indirect
   ("it's dark in here"), relative ("a bit brighter", "turn it down"),
   multi-part, exclusions, numbers as words, self-corrections, fragments,
   first-person ("i turn on…").
5. **The label never comes from the LLM.** One resolver with HA's rules computes
   the decision and canonical call (convention table below); the **HA oracle**
   confirms the call resolves to exactly the intended entities.
6. **Meaning check (round trip).** An independent judge pass sees only the
   scene + utterance and must recover the same intent and target. Utterances
   that drifted ("dim" rendered as "turn off") or became ambiguous are
   dropped or relabelled.
7. **Templates stay only as a floor:** for cells where the LLM repeatedly fails
   verification, and for exact-count guarantees.

**Convention table (one canonical call per situation), enforced by the oracle
and aligned with the eval grader:** lock → `{name}`; TV → `device_class: [tv]` +
domain; never `floor` or `area` unless spoken; area call vs. per-device calls;
the "which one" and refusal wording. Where the eval's exact-arg grader encodes a
different convention, align the grader or grade on outcome (per the
outcome-eval memory).

**Real tool results:** capture real HA intent-tool and GetLiveContext responses
from integration traces and use their format in post-tool turns, replacing
`{"result": "Success"}`.

### Phase 3: realism layers

- **STT noise** learned from the real captures: pair the ~130 real commands
  with intents and use their error types ("t v", "liver room", "pd", "turn
  out", "i turn on"), not a fixed list. Every corrupted row passes the
  recoverability check or becomes "ask"/"ignore".
- **Junk/ignore** built from real non-command transcripts (background speech,
  partial sentences). It's sized to real usage: about 64% of real command-path
  captures are not commands, versus 6% of v5.
- **Multi-turn** from real shapes: clarify → answer, "no, the other one",
  failed call → correction.

### Phase 4: non-data accuracy levers (in parallel with the data work)

- **Constrained decoding.** llama.cpp GBNF grammar generated per request from the
  offered tools, their argument schemas, and the exposed entity names/aliases.
  The model can't emit a nonexistent device or argument (the `floor`
  hallucination; "Kitchen Worktop Lights"). It complements the integration's
  pre-execution validation, and is likely the cheapest large accuracy gain.
  Runtime change in `custom_components/sayso` (prompt/contract docs updated).
- **Prompt size.** ~5k tokens per request, mostly tool schema, for a 230M model
  to scan. Options: a more compact tool rendering in SaySo's own prompt, more
  confident routing. Any change is a contract change and must be trained on.

### Phase 5: real-usage loop

The integration already records traces. Build trace → review/label → real-data
set early: failed, corrected or low-confidence real requests become the
highest-value rows (dev/test first, train with your OK). Over time this replaces
guessing at the input distribution.

## Splits

| Split | Source | Use |
|---|---|---|
| train | generator, train vocab pools | training |
| val-ID | same pools, held-out scenes | overfitting / loss curve |
| val-OOD | held-out pools (nicknames, names, phrasings, rooms) + real-home holdout entities | **checkpoint selection** |
| real-dev | half of the hand-labeled real satellite transcripts | selection sanity on real speech |
| real-test | other half, locked | the most honest production signal |
| test (locked) | `evals/` promotion 120 + regressions | promotion only |

Today checkpoint selection uses the smoke suite, which is a subset of the locked
promotion suite. That's selection on the test set, and it stops. Vocabulary
pools (nicknames, owner names, brands, templates, rooms) are split 80/20 *before*
generation.

## Sequencing and cost

1. **Phase 1 on v5 (~1–2 days):** oracle harness, judge script, probe,
   review pack; label the ~130 real commands (the judge pre-labels, you review).
   Result: the first measured quality numbers, plus real-dev/real-test.
2. **Phase 2 core on a 4k pilot:** grid + scenes + siblings + LLM utterances +
   resolver/oracle/meaning check. Compare with v5 on the quality report
   (validity, agreement, probe AUC, diversity) before scaling.
3. **Phase 3 layers, then scale to 20–40k** verified rows, as many as pass,
   not a fixed count.
4. **Train v6;** select on val-OOD / real-dev; promote on the locked suite.
   Compare with v5b.
5. **Phase 4 (constrained decoding)** alongside steps 2–4, since it is
   independent of the data. **Phase 5** begins as soon as the integration's
   traces are reachable.

LLM generation cost: roughly 40k cells × ~8 utterances, plus judge passes, on
Qwen 27B. At the VM's measured throughput this is a multi-hour batch that needs
the GPU, so it runs between training jobs.

## Needs your OK

- Installing Home Assistant (test harness) in a **new** venv on the VM (or the
  Mac) for the oracle.
- Using the Qwen server as utterance generator and judge (needs the GPU between
  training runs).
- Constrained decoding in the integration (runtime change).
- Hand-labeling ~130 real command transcripts (I pre-label with the judge; you
  review in ~20 minutes).
- Whether real satellite transcripts may be used for training style/noise
  (train side) versus kept purely for dev/test.
