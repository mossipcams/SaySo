# 0008. Satellite deploy and audio

## Status

Accepted.

## Context

The satellite (`satellite/`) is a thin overlay on linux-voice-assistant (LVA). It runs on a Raspberry Pi as the `pi` user, from `/opt/sayso-satellite`, which is not a git checkout. Four problems came up:

- LVA upgrades changed behaviour the overlay replaces. v1.1.18 calls `_tts_finished(pipeline_ended=...)`, requires the output device to be known to mpv at startup, and its mic refresh replaces the recorder wrapper the overlay depends on.
- The deploy procedure was not documented. Plain `uv sync` replaces the patched LVA install with the unpatched tag from `uv.lock`, and the satellite stops starting.
- Home Assistant's voice activity detection (VAD) opened about 1.05 s into the audio stream, so the first part of a command was clipped.
- The wake-hook mining spool dropped new hard-negative records when it was full, so the spool kept old records and counted only the drops.

## Decision

- **LVA pin.** The pin moves from `99ee08f` to the v1.1.18 tag (`b55d36a`). Patches 0001 and 0002 are regenerated against that tag, so plain sequential `git apply` works on a shallow checkout, as CI does. Patch 0002 adds a `linux_voice_assistant/events.py` shim that re-exports `LVAEvent`. A repo rebuild fails without it.
- **Overlay fixes.** `_tts_finished` forwards `*args, **kwargs`, because an explicit `pipeline_ended=` breaks fakes that take no kwargs. The output device is passed as `pulse/<sink>`, and v1.1.18 exits at startup otherwise. The mic refresh keeps the `_ResamplingRecorder` wrapper, so gain, `raw_sink` and the clip tally survive.
- **Adopted from upstream.** The wake phrase is set on follow-ups. A `duplicate_wake_up_detected` error gets no failure chime, but the turn still finishes its trace and capture and the wake hook still re-arms.
- **Not adopted.** The pipeline-ended sound is disabled with `--pipeline-ended-sound ""`, because it conflicts with SaySo's chimes. `--mac-address` is not adopted, because it needs new config and fixes no current problem. `install_playback_recovery` stays, because removing it would delete existing test assertions.
- **Deploy procedure.** Pull `main` into a clone on the Pi. Copy `satellite/sayso/`, `satellite/patches/`, `pyproject.toml` and `uv.lock` into `/opt/sayso-satellite/`, and leave `models/` alone. Rebuild the LVA tree beside the live one at the pinned tag, apply the patches, then swap it in. Run `uv sync --frozen --inexact --no-install-package linux-voice-assistant`. Restart with `sudo systemctl restart sayso-satellite`. The steps are in `satellite/README.md`.
- **VAD warm-up.** The satellite sends 1.2 s of silent PCM before the buffered microphone audio, so HA's VAD advances before the audio is flushed. The buffered wake-ring audio is flushed right after the primer, with the token-based cancellation guard and underflow handling kept. The earlier 1.2 s wall-clock hold was removed. It duplicated the primer, which already advances HA's VAD by the observed ~1.05 s of processed audio. The Phase 1 latency work that removed the hold is referenced in the old plan, but its plan file is not in the repo.
- **Mining spool.** When a class is at its cap, the oldest record of that class is evicted. When the spool is at its record or byte limit, the oldest record overall is evicted. Eviction removes `records/<id>/` and `outcomes/<id>.*.json`, as `drain_acks()` does. The counters stay consistent, the record being published is never evicted, and evictions are counted as `evicted` in the losses stats. An ack for an evicted ID is harmless. The `ingest_record()`, `write_ack()` and `drain_acks()` contract with the host is unchanged.

Rejected:

- Keeping the wall-clock hold on top of the primer. It was a conservative duplicate of the primer.
- Dropping new records when the spool is full. It kept stale records and hid the loss.

## Consequences

- The satellite runs LVA v1.1.18 with two patches, and the repo rebuild needs the `events.py` shim.
- A plain `uv sync` in `/opt/sayso-satellite` is a known failure. Use the documented command.
- The VAD primer is 1.2 s of silence on every command. If a live wake-to-STT A/B test shows the command onset clipped or missing, Home Assistant is dropping pre-VAD-ready audio, and the wall-clock hold must be restored.
- Retuning `post_tts_cooldown_ms` for v1.1.18 needs the live device and is not done.
- The mining spool keeps the newest records. Old negatives are evicted under pressure, so a record set is not a complete history of misses.

## Shipped in

- `9843212` fix(satellite): upgrade linux-voice-assistant to v1.1.18
- `e842d00` docs(satellite): document the Pi deploy and safe uv sync
- `f0786be` feat(wake): evict oldest mining records when the spool is full
- `cd833f8` fix(satellite): warm up HA VAD before command audio
- `526c6a2` fix(satellite): advance HA VAD with silence pre-roll
- `c0cd6ec` perf(satellite): flush wake ring immediately after VAD primer
