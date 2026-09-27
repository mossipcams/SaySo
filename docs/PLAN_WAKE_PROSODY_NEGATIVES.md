# Plan: wake prosody + negatives (TV false-positive program)

**Date:** 2026-09-27
**Status:** Draft — review before implementation
**Trigger:** TV-session false-positive capture 2026-09-26/27 (Pi spool,
`sayso-voicev1.onnx` @ 0.22) and linguistic analysis of the 167-clip labeled
set plus a 655-clip spool scan (mining report:
`llm:/srv/llm/data/wake/data/REPORT_tv_20260926.md`).

## Evidence

1. **The minimal-pair neighborhood is empty in real TV data.** 655 clips
   scanned: say-no / say-go / say-show / say-soul / say-saw / hey-so / hay-so /
   stay-no / stay-go / stay-show = **0 occurrences**; only stay-so (2) and
   stay-still (5) occur. The recipe already synthesizes these as TTS
   `custom_negative_phrases`, so no synthesis gap needs filling.
2. **The real collision class is the exact bigram "say so": 91 occurrences.**
   63 labeled positives (62 isolated), 2 TV phrase-exact negatives
   ("Say so, Paul.", "You can say so."), 26 spool `near_threshold` "Say so."
   clips (22 bare) that are unlabeled additional positive candidates
   (scores 0.212–0.414).
3. **Prosodic isolation does not discriminate.** 86/91 "say so" occurrences
   are start-of-utterance — including the TV "Say so, Paul." The 2 s window
   model cannot see the trailing context that actually separates wake
   ("Say so" + silence/command) from TV ("Say so" + name/words).
4. **The positive set is prosodically degenerate.** 60/63 positives are the
   identical bare "Say so." (one speaker, one mic, one room, one delivery).
   The model learned one acoustic event, not a class.
5. **Near-miss shapes are the bulk of observed FPs:** thank-you family ×25,
   okay-so ×11, bare "so" ×7, stay-still ×3, see-you-soon ×3 — all captured in
   `negative_tv_20260926` (104 clips, train VM).
6. **Domain confound in the training mix.** The recipe trains "say so" as a
   negative in clean TTS while real-mic "Say so" is a positive, so the model
   can use *acoustic domain* as a label proxy. TV dialogue is room-mic domain,
   so it lands in the positive domain. Evidence: "I'm sure she has." fired at
   0.431 with near-zero phonetic overlap to [S EY1 S OW1] — cadence/domain,
   not phonemes.
7. **A compatible verifier mechanism already exists in code but is not
   deployed.** `satellite/sayso/wake/verifier.py` supports
   `feature_kind=speech_embedding` (logistic over concat(mean, std) of the
   last 16 speech-embedding vectors, 192-d) and AND-gates with the DNN. The
   Pi config has no `wake_word.verifier` key. The legacy `sayso-verifier.npz`
   is `mel_union` — incompatible with the generate-first model and previously
   misfired (blessed recorded SaySo, vetoed live fires). No fit script exists
   in the repo.

## Increment 1 — break the domain confound (training data, next LiveKit run)

- Add `negative_tv_20260926` (104 TV-through-room-mic negatives) as the
  hard-negative overlay in the next generate-first run. This puts room-mic
  "say so"-ish audio in the negative class and breaks the domain proxy.
- Extend positive generation with pitch (±2 st) and energy (±6 dB) variation
  on top of the existing `length_scales` and `slerp_weights`, so the wake
  contour is a class. Optional: a second real speaker's eMeet takes.
- Ingest the 26 spool `near_threshold` "Say so." positive candidates **after
  human verification** (22 are bare "Say so.").
- Keep mining; phrase-exact TV negatives (2 so far) are the irreducible class
  and keep arriving.

Files: `satellite/models/sayso.yaml` (recipe params), generator stage of the
train-VM pipeline (`/srv/llm`, not this repo), `docs/WAKE_TRAINING_DATA_ARCHITECTURE.md`
(set notes).

Verification: LiveKit eval recall ≥ 0.90 on silent30 and FPPH ≤ 0.10; strict
eval fixtures extended with the 2 TV phrase-exact clips and 5 near-miss
shapes; compare against the voicev1 baseline (24/30 @ 0.22). No promotion
without the existing gate.

## Increment 2 — post-STT abort (optional, in-architecture)

Only if the phrase-exact TV class still triggers pipeline runs after
increment 1: in the SaySo `ConversationEntity`, a wake followed by a
non-command fragment (name, bare word, embedded "say so") returns a silent
no-op instead of an LLM turn.

Files: `custom_components/sayso/conversation.py` (+ colocated
`custom_components/sayso/test_*.py`), `evals/cases/` coverage for the abort
rule.

Verification: new tests (wake + "Paul." → no-op; wake + "turn on the lights"
→ normal routing); smoke suite (~24 cases) and the locked 120-case promotion
suite stay green.

## Increment 3 — satellite-side continuation gate (deferred)

Only if FPs persist after 1–2: gate the wake event on an acoustic check of the
~1 s after detection (silence/command continuation vs name continuation).
This is an architectural change (second inference stage on the satellite) and
needs an `ARCHITECTURE.md` update plus its own plan.

## Out of scope

- Changing the wake word itself (product decision)
- Training on the 63 auto-labelled positives before human verification
- Multi-satellite, streaming, generalized diagnostics

## Overall verification

- Offline eval path (`evals/`) and the existing test suites stay green
  throughout.
- Live A/B on the Pi: TV-session FP rate before/after each increment.
- Every model swap keeps a rollback copy on the Pi (existing practice).
