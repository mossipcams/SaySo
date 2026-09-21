# Satellite wake-data collection

How the living-room Pi **actually** gathers audio today. This is **not** an
ideal collection pipeline. Training-side layout:
`docs/WAKE_TRAINING_DATA_ARCHITECTURE.md`. Live model: `docs/HANDOFF_WAKE.md`.

Satellite is a thin LVA overlay. It does not do STT, NLU, or actions. It
**does** own the mic, the 2 s wake window, and the mining spool.

## Device

| | |
| --- | --- |
| Host | `192.168.1.54`, user `pi`, SSH **2222** |
| Mic | Blue Snowball USB (`hw:1,0` / Pulse `alsa_input.usb-BLUE_MICROPHONE_Blue_Snowball_…`) |
| Native capture | **48 kHz** mono; pipeline resamples **once** to **16 kHz** |
| Config gain | `audio.mic_gain_db: 10.0` on the satellite path |
| Unit | `sayso-satellite.service` (override User=`pi`, `PULSE_SERVER` in `/run/user/1001`) |

Pi has **no `sftp-server`**. Copy with `tar` over SSH. Wavs stay off git.

## Three capture paths (current)

They do not share a format, a clock, or a label.

### 1. Live inference + miner (the only production path)

Config: `wake_word.mine_dir: /var/lib/sayso-satellite/wake-mining`.

`HardNegativeMiner` publishes the **exact 2 s / 16 kHz window** the
classifier scored (`satellite/sayso/wake/mining.py`), plus optional
`pre.wav` / `post.wav` from the capture ring. Sampling classes:

- `detection` — score ≥ threshold (0.28)
- `near_threshold` — `mine_threshold` (0.25) ≤ score < threshold
- `below_threshold` — sparse sample of the rest

Each capture is a directory:

```text
/var/lib/sayso-satellite/wake-mining/records/<capture_id>/
  window.wav
  record.json
  pre.wav / post.wav   # optional
```

Host ingest: `scripts/wake_mine_report.py` (`--label`, `--inventory`). The
satellite deletes **only** host-acked records. Caps: `mine_max_records`
(default 2000).

This is the path that matches inference. Labelling is still manual and
easy to skip. Unlabelled spool must **not** go into the classifier.

### 2. Prompted session (2026-09-21) — ad hoc, not wired into the unit

Satellite **stopped** so ALSA could take the Snowball. PipeWire source was
`SUSPENDED`; `parecord` wrote empty 44-byte wavs. Capture was:

```text
timeout 1.2 arecord -D hw:1,0 -f S16_LE -r 48000 -c 1
```

Beep on the speaker, ~1.2 s record, 200 takes. Output on the Pi:

```text
/var/lib/sayso-satellite/recordings/positive_sayso_20260921/
  sayso_NNN.wav          # ~1.13 s, 48 kHz, no 10 dB gain
  padded_16k_2s/         # 16 kHz, 2.0 s, silence pad (center)
```

Host copy: `sayso-wake-data/data/200-positive/` (padded) and `raw/`.

Measured vs living2’s 50 / holdout SaySo:

| | living2 50 / holdout | 200 session |
| --- | ---: | ---: |
| Duration | 2.05 s native | 1.13 s chop |
| Active speech | ~1.2 s | ~0.56 s |
| RMS | ~−38 dBFS | ~−48 dBFS raw / −50 padded |
| Peak | ~3500 | ~900 |

`timeout` also left **24** wav headers broken (fixed from file size after
the fact). First countdown was **silent** (log only); a later run used
audible 3-2-1.

This path is how we got 200 files. It is **not** how the satellite hears
SaySo. Retrains that treated these as a replacement for the 50, or that
overlaid them on `room_snowball`, did not beat living2 on wake gates.

### 3. Operator mic check

`sayso-satellite test-mic` → 5 s `parecord` at native rate into
`/var/tmp/sayso-satellite/mic-check.wav`, then a 16 kHz processed copy
**with** `mic_gain_db`. Useful as a level probe. Not a corpus.

STT tap (`stt_capture_dir`) is post-wake command audio, not the wake
window.

## What never gets collected today

- A **2.05 s native** prompted take at living2 level (the 50/holdout object)
- Automatic **gain** on ALSA prompted sessions (10 dB is satellite-path only)
- Forced **word position** in the 2 s frame (streaming wake finishes near
  the end of the window; beep-loop chops are wherever the speaker landed)
- Sidecar **labels** on prompted wavs (positive is implied by folder)
- A holdout split at capture time (easy to train on the gate clips)
- `satellite/eval/audio/` fixtures — still empty; `wake_train.py --schedule`
  stays off

## Transfer and stop/start

Prompted ALSA needs the unit **stopped** or the Snowball stays with
PipeWire. After a session, start `sayso-satellite.service` again. Miner
collection needs the unit **running**. Those two modes fight.

Do not stop LFM2 on `192.168.1.140` when copying wavs there.

## Why this collection setup is not ideal

1. **Miner windows (right) and prompted chops (wrong length/level) both
   get called “positives.”** Folder names collapse that distinction.
2. **Stopping the unit to record** means you cannot collect from the same
   resampler/gain/AEC path the ONNX uses.
3. **Beep-loop 200** maximizes count, minimizes diversity (same distance,
   same rhythm, same truncation). living2 already AND-fires ~147/200 of
   the padded copies — extra easy positives, not 05-like hard ones.
4. **No intake checklist** (duration, RMS, peak, clip, holdout exclusion)
   before a set is named `200-positive` and treated as train-ready.
5. **Eval/train contamination risk** is operational: holdout lives only on
   the host; nothing on the Pi marks “do not record this script again into
   train.”

## Collection that would match the model (not implemented)

Keep the miner for hard negatives (human `--label`). For new true wakes:
same Snowball, **unit running or an equivalent 16 kHz +10 dB 2 s path**,
native **~2 s**, level like the 50, **not** the holdout scripts, keep the
50 in train. Do not record more 1.2 s beep takes unless the goal is
session-matched data, not living2 replacement.
