# Plan: wake-word rethink (TV false positives)

**Date:** 2026-09-27
**Status:** Draft — needs the user's decisions below before implementation
**Constraint:** no second-stage verifier (user decision, 2026-09-27).
**Supersedes:** increments 1 and 3 of `docs/PLAN_WAKE_PROSODY_NEGATIVES.md`
and the ablations proposed in `docs/HANDOFF_WAKE_TV_FALSE_POSITIVES.md`.

## Target

Measured on audio from the living-room Pi, not LiveKit's validation set:

- **TV false fires ≤ 0.5/hour**, measured over held-out TV hours. Today it is
  about 6–90/hour.
- **Real-wake recall ≥ 0.90**, measured on held-out real takes that include
  takes spoken over the TV.

## What we know (evidence, 2026-09-20 → 09-27)

1. Three recipe iterations (pass 1, pass 2/voicev1, tvneg) all sit on
   roughly the same recall-vs-TV-fires curve. The tvneg retrain moved the
   operating point; it didn't move the curve. voicev1 raised to 0.30–0.33
   matches it.
2. **Real wakes score low.** voicev1 catches 3/30 silent30 takes at 0.50.
   Even on LiveKit's own TTS validation it only reaches 0.49–0.56 recall at
   0.50. The deployed threshold of 0.22 sits inside the score range of
   ordinary room-mic speech.
3. The positive class is roughly 90k TTS rows plus 60 real takes (one
   speaker, one session). The real-mic domain is almost absent from
   training.
4. Domain has to be balanced across the two labels. Adding 96 room-mic TV
   negatives ×5 against 60 room-mic positives ×5 lowered real-wake scores
   along with TV scores (tvneg).
5. **Our metrics don't measure the problem.** LiveKit reports 0.07
   false fires per hour on generic ACAV validation features. Real TV is
   6–90/hour. The only real TV test data is 24 clips, and they were
   selected because they fired.
6. The satellite fires on a **single** 160 ms hop above threshold
   (`satellite/sayso/wake/livekit.py`). It has no consecutive-hop
   requirement.
7. Loudness is not a usable gate. TV windows peak at a median of −19.5
   dBFS and real wakes at −15.9, with heavy overlap.
8. "Say so" is 2 syllables and a common English bigram. TV said it twice
   in about 5 h. Its acoustic neighbourhood ("okay so", "so", "thank you",
   "stay still") is dense in dialogue. A perfect "say so" detector still
   fires on TV "say so".

## Root cause

The wake class was learned from TTS, so real wakes score in the same range
as room-mic speech. The threshold then had to drop into that range. Nothing
measures TV false fires directly, so each change was judged on proxies
(TTS validation, a few dozen clips) that don't track the real failure. The
short, common phrase and the single-hop firing rule make it worse.

## Options considered

| Option | Verdict |
| --- | --- |
| More recipe tweaks judged on LiveKit eval | Stop. Three iterations moved along the same curve. |
| Raise threshold | Stopgap only. It trades real wakes 1:1 for TV silence. |
| Loudness gate | Rejected. Distributions overlap (finding 7). |
| Second-stage verifier | Excluded by the user. |
| Consecutive-hop firing rule | Do. Cheap, testable by replay, and standard in openWakeWord/microWakeWord. |
| In-domain data at scale (both labels) | Do. This is the root-cause fix. |
| Change wake phrase | Recommend testing it. The phrase choice caps the achievable floor. |
| Switch framework (e.g. microWakeWord) | Defer. Data and eval are the problem whichever framework we use. Revisit only if phase 3 doesn't move the curve. |
| Post-STT no-op in the conversation agent | Optional, separate. It limits the damage a TV fire can do; it doesn't reduce fires. |

## Plan

### Phase 1 — a real yardstick (no training)

- Record **≥ 6 h of TV-only audio** on the Pi through the existing
  long-form session path (`wake_corpus.py` stage → ship). Use at least 3
  distinct shows or channels at normal volume. Split by session: 2 h
  eval-only, the rest for training in phase 3.
- **Real positives:** listen-verify the 63 + 26 mined clips. Record ≥ 50
  takes per household speaker at couch, kitchen, and doorway distances,
  half of them **with the TV on**. Split by speaker/session into train and
  eval.
- Extend the replay eval to report TV false fires/hour and real-wake recall
  per threshold, using long-form sessions and the production hop.
- Baseline voicev1 and tvneg on it.

Exit: two numbers per model per threshold, from held-out real audio.

### Phase 2 — firing rule (satellite only, no training)

- Add a consecutive-hop requirement (fire when ≥ N of the last M hops are
  ≥ threshold). N/M go in satellite config with default 1/1, so today's
  behaviour is unchanged until the config changes.
- Sweep N/M × threshold on the phase-1 replay. Pick the point that meets
  recall ≥ 0.90 with the fewest TV false fires.

Exit: measured TV false fires/hour at 0.90 recall, before vs after. If the
target is met, stop and ship.

### Phase 3 — in-domain training data, both labels

- **Positives in the room domain:** play a few thousand TTS "SaySo" clips
  (many Piper voices) through a speaker in the room and record them on the
  Pi overnight. Add the train-split real takes.
- **Negatives in the same domain:** use every 2 s window of the train-split
  TV hours, not just fires. Also re-record TTS adversarial negatives
  through the same speaker path, so the room domain appears in both labels
  at similar rates. Windows whose transcript contains "say so" are
  excluded from both classes.
- Positive augmentation mixes in **TV background** (train-split TV hours)
  so wakes spoken over the TV stay positive.
- Train with the fixed recipe (no wake-phoneme negatives). Judge only on the
  phase-1 yardstick, together with the phase-2 firing rule.

Exit: the curve moves, meaning fewer TV false fires at the same recall than
voicev1 + the firing rule. If it doesn't, revisit the framework.

### Phase 4 — phrase decision (product call, data-driven)

- Train a TTS-only candidate for an alternative phrase (e.g. "Hey SaySo")
  with the same pipeline. Replay it over the same held-out TV hours and
  compare TV false fires at matched recall. TV fires can be measured before
  anyone records real positives for the new phrase.
- Switch only if the floor for "say so" (TV saying the phrase plus its
  neighbourhood) is what keeps the target out of reach after phases 2–3.

## Files (by phase)

- Phase 1: `satellite/sayso/wake/replay.py`, `satellite/sayso/wake/eval.py`,
  `satellite/eval/` (new held-out TV/positive cases), and the colocated
  `test_*.py`.
- Phase 2: `satellite/sayso/wake/livekit.py` (firing rule), satellite
  config, and `test_livekit.py`.
- Phase 3: the train-VM data prep script in `scripts/`, plus
  `satellite/models/sayso.yaml`.
- Docs: `docs/WAKE_TRAINING_DATA_ARCHITECTURE.md` (new sets and splits).

## Decisions needed

1. Approve phase 1 data collection. It needs the TV left on while the Pi
   records, recording sessions per speaker, and listen-verification.
2. Is changing the wake phrase an acceptable outcome (phase 4)?
3. Until phase 2 ships, keep 0.22 or raise voicev1 to about 0.30 as a
   stopgap?
