# Plan: wake retrain for TV false positives (increment 1 of PLAN_WAKE_PROSODY_NEGATIVES)

**Date:** 2026-09-27
**Source:** `docs/HANDOFF_WAKE_TV_FALSE_POSITIVES.md`, `docs/PLAN_WAKE_PROSODY_NEGATIVES.md`

## Review findings

- **Domain confound confirmed and worse than described.** espeak-ng (Piper's
  phonemizer) gives `sˈeɪ sˈoʊ` for both the target `SaySo` and the negative
  `say so`. The recipe trained identical phoneme input as positive and
  negative. Five more negatives end in the same sequence: `if you say so`,
  `just say so`, `you don't say so`, `they say so`, and `essay so`
  (`ˈɛseɪ sˈoʊ`). Because LiveKit right-aligns positives in the 2 s window,
  a trailing wake phrase in a negative gives a contradictory label too.
- The TV set has 122 negatives (the plan says 104, which was the count before
  pass 3–5). 2 of them are phrase-exact ("Say so, Paul.", "You can say so.").
  Those contradict the positive label inside a 2 s window, so they stay
  out of training and go into eval.
- LiveKit 0.2.1 augmentation is hard-coded (EQ + tanh + RIR + background).
  It has no pitch or gain augmentation.
- The 63 mined positives and 26 spool candidates are unverified, so they
  aren't used for training. The 63 are used only as an eval signal, and
  that signal is labelled unverified.

## Scope

1. `satellite/models/sayso.yaml`: remove the 6 negatives that end in
   `seɪ sˈoʊ`. Adopt pass-2 generation params (30k/6k, `length_scales`
   0.65–1.4, `slerp_weights` 0.1–0.9, 50k steps) so the recipe matches the
   champion's.
2. `scripts/wake_livekit_train.py` (new, one-off run driver):
   - `prepare`: new work dir. Copy pass-2's original TTS positives
     (positive recipe unchanged, which saves about 8 h of TTS). Seed the
     eMeet positives ×5, the 162 LibriSpeech negatives ×1, and the TV
     negatives ×5 into `negative_train`. The TV holdout (latest 20% by
     `published_utc`) and the 2 phrase-exact clips stay out. Regenerate all
     TTS negatives and write a sha256 seed manifest.
   - `run`: generate → augment → train → export → eval in-process. The
     augmentor gets PitchShift ±2 st and Gain ±6 dB (p=0.5) added. They're
     applied to every split, so pitch/level can't become a label proxy.
   - `score`: per-clip max score for any ONNX over a wav dir, used for the
     separation comparison.
3. Docs: this plan, plus the run result in `docs/HANDOFF_WAKE_TV_FALSE_POSITIVES.md`.

## Verification

- `prepare`: counts and seed sha256 match the source manifests.
- `score` on voicev1 over the TV negatives roughly reproduces the manifest
  `livekit_score` values. This sanity-checks the scorer.
- After training: LiveKit eval metrics, strict silent30 at the candidate's
  optimal threshold (`satellite.eval.run --strict --promotion-only`), and a
  separation table (voicev1 vs candidate) over held-out TV negatives,
  phrase-exact TV, and the unverified mined positives.
- Gates are unchanged: recall ≥ 0.90 silent30, FPPH ≤ 0.10. No Pi deploy
  without the user's call.
