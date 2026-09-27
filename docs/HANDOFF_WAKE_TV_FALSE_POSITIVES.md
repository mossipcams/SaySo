# Handoff: wake TV false positives — what we know and what's next

**Date:** 2026-09-27
**Context:** TV-session false-positive capture on the Pi satellite
(2026-09-26 21:19 → 2026-09-27 02:33 UTC), five mining passes, full
phonetic analysis, and a model-separation audit.

## What happened

The TV in the living room triggered the wake detector continuously
(~1–15 fires/10 min). We captured every `detection`-class record off the Pi
spool (379 total across 5 days), transcribed with faster-whisper `small.en`,
broke them down phonically against the target **SaySo = [S EY1 S OW1]**, and
kept everything from last Tuesday (2026-09-22) onward per user instruction.

**Labeled data now on the train VM** (`/srv/llm/data/wake/data/`):

| set | n | notes |
| --- | ---: | --- |
| `negative_tv_20260926/` | 122 | TV/speech confusables + empty/SFX + 2 phrase-exact TV. Per-clip manifest: transcript, word timestamps, phonetic span, similarity, sha256. Safe to use as-is. |
| `positive_mined_20260926/` | 63 | Auto-labelled from transcripts (60 bare "Say so." + 3 wake+command). **Human verification required before training use.** |
| `REPORT_tv_20260926.md` | — | Full phonetic analysis + usage guidance. |

Plus **26 spool `near_threshold` "Say so." clips** (22 bare, scores 0.212–0.414)
identified as additional positive candidates — not yet shipped, awaiting the
same human verification. Pi spool was ack-drained after each pass; the
manifests are the durable record. Local scratch (all 379 records + analysis
scripts): `/tmp/sayso-tv-triggers-20260926/`.

## What the analysis found

### 1. The model cannot separate SaySo from TV speech (the core problem)

| | min | median | max |
| --- | ---: | ---: | ---: |
| real SaySo (63) | 0.255 | 0.452 | 0.669 |
| TV false-positives (121) | 0.222 | 0.309 | 0.579 |

- Overlap band 0.255–0.579: 57/63 real wakes and 104/121 TV FPs inside it.
- **Best single threshold: 0.408 → 75% accuracy** (catches 48/63, admits
  31/121). No threshold works. A functioning wake model separates
  target/non-target at >99%.
- It is *not* a pure speech detector — 77 TV speech windows scored 0.009–0.086
  (correctly rejected). The failure is a gray zone: hundreds of TV windows at
  0.12–0.43, with real SaySo inside the same band.

### 2. The threshold was lowered 0.50 → 0.22 over a week to chase recall

Recorded `detect_threshold` per spool record: 0920: 0.50 → 0921: 0.28–0.50
(five values in one day) → 0922: 0.25/0.37/0.42 → 0923–0925: 0.25 →
0926–27: 0.22. At 0.50 the model caught only 3/30 real SaySo, so it was
lowered until real wakes were caught — admitting the TV gray zone at every
step. The low threshold is a symptom of missing separation, not the cause.

### 3. Linguistic structure of the collision

- **"say so" is a real English bigram** (confirmation phrase). 91 exact
  occurrences in 655 scanned clips; 0 occurrences of the minimal-pair
  neighborhood (say-no/say-go/say-show/say-soul/hey-so/stay-no — TV never
  says them; the recipe already synthesizes them as TTS negatives).
- **Isolation does not discriminate**: 86/91 "say so" occurrences are
  start-of-utterance — *including* TV's "Say so, Paul." The real signal is
  trailing context (command/silence vs name/words), invisible to a 2 s window.
- **The model matches cadence + vowels, not phonemes**: the largest confusable
  family (gratitude, 25 clips) has no /s/; "I'm sure she has." fired at 0.431
  with near-zero phonetic overlap.
- **The positive set is prosodically degenerate**: 60/63 are the identical
  bare "Say so." (one speaker, one mic, one delivery). The model learned one
  acoustic event, not a class.
- Confusable families (122 kept negatives): gratitude ×~25, okay-so ×11,
  bare "so" ×~9, stay-still/so ×5, see-you-soon ×3, plus single-word bursts.

### 4. A concrete bug: domain confound in the training recipe

`satellite/models/sayso.yaml` trains "say so" as a **clean-TTS negative**
while real-mic "Say so" is a **positive**. The model can use *acoustic
domain* as a label proxy; TV dialogue is room-mic domain, so it lands in the
positive domain. This directly explains the gray zone and is the
highest-leverage fix.

### 5. Verifier: exists in code, not deployed (removed from plan per user)

`satellite/sayso/wake/verifier.py` supports a `speech_embedding` logistic
stage (192-d, AND-gated with the DNN); the Pi config has no `verifier` key;
the legacy `sayso-verifier.npz` is the incompatible `mel_union` one that
previously misfired. The separation data above is the strongest argument for
a second stage, since trailing context is the one feature that discriminates.
User removed it from the plan on 2026-09-27 — revisit if FPs persist.

## Current state

- **Satellite:** still running on the Pi, `sayso-voicev1.onnx` @ 0.22, no
  verifier. TV off as of ~02:33 UTC 09-27; spool continues mining
  near/below-threshold.
- **Train VM:** 122 negatives + 63 unverified positives + report, all
  sha256-verified. No training run has used them yet.
- **Repo:** `docs/PLAN_WAKE_PROSODY_NEGATIVES.md` (3 increments: fix
  domain-confound + retrain → post-STT abort → satellite continuation gate),
  `docs/WAKE_TRAINING_DATA_ARCHITECTURE.md` notes the new sets.
- **Pending human work:** listen-verify the 63 positives and 26 spool
  candidates before any training use.

## What might be next (in leverage order)

1. **Fix the domain confound and retrain** (plan increment 1): add the 122 TV
   negatives as the hard-negative overlay, add pitch/energy variation to
   positive generation, ingest verified positives. Necessary; confidence in
   closing a 75%-accuracy gap with the same approach is moderate.
2. **Reconsider the second-stage verifier** (removed 09-27): the data
   supports it — it's the only stage that can see trailing context.
3. **Decide the wake word** (product call): "say so" is a dense, real English
   bigram; a rarer phrase lowers the irreducible TV floor.
4. **Keep mining** while TV is on: each pass adds hard negatives; pass 6 is a
   one-liner (pull `detection` records > last pass, transcribe, ship, ack).
5. **Threshold:** leave at 0.22 while collecting; any raise trades real
   wakes for TV silence (overlap band) and is not a fix.

## Verification notes for whoever picks this up

- Re-verify any set before training: `sha256` per manifest entry, 16 kHz
  mono, 32000 samples.
- The 63 positives are auto-labelled (whisper-transcript heuristic) — the
  label authority is the human, per `docs/WAKE_WORD_DATA.md`.
- LiveKit eval gates (recall ≥ 0.90 silent30, FPPH ≤ 0.10) apply to any
  retrain; voicev1 baseline is 24/30 @ 0.22.
- Pi has no sftp-server — use rsync. Acks go to
  `/var/lib/sayso-satellite/wake-mining/acks/`; the satellite drains acked
  records itself.


## Retrain 2026-09-27 (tvneg) — result: not better than voicev1

Plan: `docs/PLAN_WAKE_TV_RETRAIN.md`. Driver: `scripts/wake_livekit_train.py`.
Work dir `llm:/srv/llm/wake/runs/livekit-tvneg-20260927/`, archive
`…-tvneg-20260927-models/` (`sayso.onnx` sha256 `0ce07cad…c5cfa3`).

Changes vs pass 2 (voicev1): removed 6 custom negatives and 25 LiveKit
adversarial phrases that espeak to the wake phonemes (`sˈeɪ sˈoʊ`; the old
recipe trained identical TTS input as both labels); 96 TV negatives ×5 and
eMeet ×5 seeded; PitchShift ±1 st + Gain ±3 dB per augment round on all
splits. TTS positives reused from pass 2; TTS negatives regenerated clean
(29,900 — two HIP OOM batches skipped).

| | voicev1 | tvneg |
| --- | ---: | ---: |
| LiveKit optimal recall / FPPH / thr | 0.851 / 0.073 / 0.22 | 0.795 / 0.073 / 0.25 |
| silent30 strict (at own thr) | 24/30 | 16/30 |
| held-out TV (24) median / ≥ own thr | 0.257 / 24 @0.22 | 0.178 / 5 @0.25 |
| mined positives (63, unverified) median | 0.518 | 0.461 |
| AUC mined pos vs held-out TV | 0.912 | 0.871 |
| phrase-exact TV (2) | 0.47, 0.49 | 0.28, 0.57 |

Scores lower on TV, but lower on real wakes too; separation got worse. Not a
promotion candidate. One run can't attribute the drop (TV ×5 may have flipped
the room-mic domain toward negative; pitch aug; clean negatives). The clean
TTS negatives in the work dir can be reused, so ablations cost ~45 min each
instead of ~5.5 h: (a) confound fix only, (b) + TV ×1, (c) + pitch/gain.

Matched operating points (held-out TV fires / mined positives / silent30):
voicev1@0.22 24/24, 60/63, 24/30 · voicev1@0.30 7/24, 56/63, 16/30 ·
voicev1@0.33 1/24, 54/63, 15/30 · tvneg@0.20 12/24, 53/63, 20/30 ·
tvneg@0.25 5/24, 52/63, 16/30. At its own threshold tvneg cuts TV fires
24→5, but raising voicev1 to ~0.30–0.33 reaches the same or better point.
