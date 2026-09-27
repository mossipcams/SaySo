# Plan: change the wake word to "Atlas" and retrain from scratch

**Date:** 2026-09-27
**Decisions (user):** the wake word is now **Atlas** (espeak `ˈætləs`). There
is no verifier. SaySo wake data is being retired, and the new generator
follows LiveKit's data-synthesis recommendations. Phases 1–3 of
`docs/PLAN_WAKE_RETHINK.md` (real yardstick, firing rule, in-domain data)
still apply to Atlas.

## Generator = LiveKit's recommended pipeline, unmodified

`satellite/models/atlas.yaml` is LiveKit's `configs/prod.yaml`
(livekit-wakeword 0.2.1) with only three changes:

- `target_phrases: ["Atlas"]`
- Atlas near-miss `custom_negative_phrases`: "at last", "at least",
  "Dallas", "cutlass", "Atlanta", and others. None of them phonemizes to
  end in `ætləs`, so no phrase is labelled both ways.
- `model_size: small`, following LiveKit's guidance of small for embedded
  devices and medium for servers.

Everything else follows LiveKit's recommendations:

- 25k/5k clips per class and 2000/500 background clips.
- Piper VITS with speaker blending across all 904 voices, at the default
  noise/length/slerp settings.
- LiveKit's own adversarial phrases from CMUdict.
- Stock augmentation: 3 rounds of EQ, distortion, room reverb, and
  background noise.
- `conv_attention`, 100k steps, `max_negative_weight` 3000, FP target
  0.1/h.

There's no custom driver: the stock `livekit.wakeword` CLI stages run from a
work dir whose `./data` symlinks to the shared setup assets.

```bash
# on llm
W=/srv/llm/wake/runs/atlas-prod-YYYYMMDD
mkdir $W && cd $W && ln -s <shared setup data> data && cp <repo>/satellite/models/atlas.yaml .
/srv/llm/bin/gpu train wake --serve-after bash -c \
  'set -e; for s in generate augment train export eval; do python -m livekit.wakeword $s atlas.yaml; done'
```

There are no real "Atlas" recordings, so v1 trains on synthetic audio,
ACAV100M, and backgrounds only. Room-mic data goes into both labels
together once real Atlas takes exist; the tvneg run showed that putting it
in one label shifts the domain.

## Later (not in this change)

- Allow "Atlas" in `satellite/sayso/config.py` (+ test). Ship the ONNX and
  switch the Pi config.
- Retire the SaySo wake data, recipes, and scripts. This needs explicit
  per-location confirmation (VM, Pi, repo/Mac) before anything is deleted.

## Verification

- LiveKit eval on the synthetic validation set: recall and false fires per
  hour at the optimal threshold.
- **Recall on real audio isn't known until Atlas takes are recorded.**
  Record ≥ 50 takes per household speaker at couch, kitchen, and doorway
  distances, half with the TV on. Split by speaker/session. No Pi deploy
  before that number exists.
- TV false fires per hour on long-form TV hours (rethink phase 1) is the
  deciding metric.
