# Plan: rebuild wake mining to check for "Atlas" and false positives

**Date:** 2026-09-27
**Context:** wake word changed to Atlas (`docs/PLAN_WAKE_ATLAS.md`); SaySo
mining data deleted.

## Current state

- **Satellite capture is phrase-agnostic.** `HardNegativeMiner`
  (`satellite/sayso/wake/mining.py`) records each scored window as
  `window.wav` plus `pre.wav`/`post.wav` (0.5 s each). It covers three
  cases: windows that fired, near-threshold windows, and a small random
  sample of low-scoring windows. The HA wake/STT outcome for each capture
  (including the command transcript) goes in `outcomes/`. Nothing there
  mentions SaySo.
- **Blocker:** `satellite/sayso/config.py` rejects any `wake_word.phrase`
  other than `"SaySo"`, so an Atlas model can't be configured.
- **Missing:** an automated check that answers two questions for every
  record: did someone say Atlas, and was it a false positive? The SaySo-era
  whisper/phonetic scripts were one-off scratch files and are gone.
  `scripts/wake_mine_report.py` still defaults to `--phrase SaySo` and
  living2 paths.

## Scope

1. `satellite/sayso/config.py`: accept any non-empty `wake_word.phrase`.
2. `scripts/wake_mine_check.py` (new). It runs over a pulled spool
   (`records/`, `outcomes/`) and does the following:
   - Verify each record's hashes (`ingest_record`).
   - Transcribe the 2 s window alone with faster-whisper `small.en`
     (plus pre + window + post for review context). The window is
     transcribed on its own because the smoke test showed whisper stretching
     an isolated word's timestamps over surrounding silence ("Atlas", 0.48 s,
     reported as 0.0–2.8 s), so timestamp overlap can't place the phrase.
   - Classify by "phrase in the window transcript" × "window fired":

     | | phrase in window | no phrase |
     | --- | --- | --- |
     | fired | `wake` | `false_positive` |
     | not fired | `missed_wake` | `near_miss` |

   - Write `check.json` into each record dir with class, transcript, words,
     and the linked STT transcript. It never sets `label`; per
     `docs/WAKE_WORD_DATA.md`, the human is the label authority.
   - Print a report: counts per class, spool time span, top false-positive
     transcripts, and missed wakes by score.
   - `--export DIR` copies window wavs into per-class dirs, with a
     `manifest.json` (sha256, score, transcript, `needs_verification` for
     `wake`/`missed_wake`). This is the handoff format for training sets.
   - Run it with `uv run --no-project --with faster-whisper --with numpy python scripts/wake_mine_check.py …`
     (Mac: model already cached). Installing into the VM venv needs the
     user's OK first.
3. `scripts/wake_mine_report.py`: default `--phrase Atlas` and fix the
   docstring. The stale living2 model defaults don't matter because they're
   only used with `--replay`.
4. `scripts/test_wake_mine_check.py` (new): classification and phrase
   matching (no whisper), including "at last" ≠ "atlas". Phrase-only-in-post
   is covered by the smoke test, since it depends on ASR.
5. Docs: `training/wake/README.md` mining section.

## Out of scope

- **Shadow scoring.** An Atlas model could score and mine in parallel while
  SaySo stays live, but that isn't built here. Atlas records only appear
  once the Pi runs an Atlas model.
- Pi config/model switch (the user's call, after real Atlas takes exist).
- Automatic pull/ack over the network. rsync commands are documented instead.

## Verification

- `python3 -m pytest scripts/test_wake_mine_check.py satellite/sayso` green.
- End-to-end smoke: synthesize a fake spool with Piper TTS "Atlas" and
  "at last" windows, run the check, and confirm the classes.
