# Plan: Koda TV-on wakes from synthetic voices over TV (v10r, v10s, v10t)

**Date:** 2026-11-04
**Goal (user):** keep synthetic data as the bulk of training, but put it in the
right domain so it helps the TV-on wakes. No extra speaker (no replay, no
measured IRs).
**Runs:** `/srv/llm/data/wake/runs/koda-phonics-v6-20261018/` on `llm`
(`run_v10r.sh`, `run_v10s.sh`, `run_v10t.sh`, `run_v10t2.sh`).

## Findings

- **The eMeet's processing is always on.** The M0 Plus exposes only a mic
  switch over USB. In the Pi TV hours, 12-18% of samples are exact zeros (a
  gate). It also cuts ~20 dB at 6.5-7.5 kHz relative to TTS. The training TV
  hours were recorded on the Pi, so they already include the room and the eMeet.
- **v9b only fires on end-aligned Kodas:** ~0.9 when the word ends at the window
  end, ~0.02 mid-window.
- **v10a/b staging was flawed:** random placement, voice scaled to the TV's local
  power, and no near-miss negatives over TV. v9b scores 0% of v10b's TV
  positives.
- **Prefix negatives (v10r)** stop prefix fires but cost ~10% synthetic recall.
  In the ensemble they are a wash. Not shipped.
- **Near-miss negatives over TV (v10s) can't be fit.** A frozen-embedding probe
  separates TTS Koda from near-misses over TV at AUC 0.78 (0.96 clean), so the
  model rejected every TV positive.
- **Distant speech over TV (v10t) works.** It pairs the TV positives with TTS
  phrases more than 2 phone edits from K OW D AH, taken from the train TV
  transcripts. Two seeds are in the README table. The ensemble 10t2+10k+10l
  raises real room voice x TV from 102/86/57 to 108/100/65 and TV-on from 3.62
  to 4.25, with 0 false wakes at 0.65 and 0.55.

## Scope

1. `scripts/wake_stage_prefix.py` + `scripts/test_wake_stage_prefix.py`: stage
   end-aligned TTS prefix negatives (the v10r record).
2. `scripts/wake_stage_tts_tv.py` + `scripts/test_wake_stage_tts_tv.py`: stage
   end-aligned TTS over real Pi TV with independent levels, plus an optional
   eMeet spectrum-matching FIR.
3. `scripts/wake_tts_distant.py` + `scripts/test_wake_tts_distant.py`: Piper
   TTS of TV-transcript chunks far from Koda.
4. `training/wake/README.md`: "Prefix negatives" and "TV-on: synthetic voices
   over TV (v10t)" sections with the result tables.
5. This plan.

Out of scope: the VM run scripts (they stay on the VM, like the other `run_v10*.sh`),
threshold changes, satellite runtime code.

## Deploy (done 2026-11-04 6:17 PM CT, with the user's OK)

The Pi runs `/opt/sayso-satellite/models/koda-phonics-ens-10t2-10k-10l-20261104.onnx`
(sha256 `9263f7eb…`) at threshold 0.65. Config backup:
`/etc/sayso-satellite/config.yaml.bak-pre-ens-10t2-20261104`. To roll back,
point `wake_word.model` at `koda-phonics-ens-9b-10k-10l-20261104.onnx`.

## Verification

- `pytest -q scripts/test_wake_stage_prefix.py scripts/test_wake_stage_tts_tv.py scripts/test_wake_tts_distant.py`
- CI gate: `ruff check .`, `ruff format --check .`,
  `pytest tests evals/tests --ignore=tests/test_realistic_v3.py`, `pytest -q satellite/sayso`.
- Live Pi test (user): TV-on wakes, quiet-room wakes, and an hour of TV with
  nobody speaking (0 false wakes).
