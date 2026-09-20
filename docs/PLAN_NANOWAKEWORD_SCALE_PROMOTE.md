# Plan: Promote scale Nano `0a3c0d64` (do not flip LiveKit)

Status: done (2026-09-20). Pi staged `0a3c0d64`; LiveKit still live. Do not flip the live satellite off LiveKit. Do not
train. Do not commit wavs, npy corpora, or `context.json`. Do not put
passwords in the repo. Do not open a PR unless asked.

## Outcome

Stage scale NanoWakeWord ONNX `0a3c0d64` on the Pi as
`/opt/sayso-satellite/models/sayso-nanowakeword.onnx` (keep ACAV and 80-pos
backups) and replace the shipped repo ONNX plus docs. `wake_word.provider`
stays `livekit`.

## Why this model

Living-room hop-feed at 0.50 vs ACAV `83d9a507`: SaySo 8/8, talk 0/8,
overlapping 0/89 (max 0.006), miner 0/74 (max 0.007), command **3/4** vs 1/4.
Frozen-21 SaySo **10/12** vs ACAV 6/12. Beats official on overlapping/miner.
Gates in `docs/PLAN_NANOWAKEWORD_SCALE.md` hold.

## Scope

1. Pi (ops): copy `sayso-official-scale-0a3c0d64.onnx` over
   `sayso-nanowakeword.onnx`. Keep `*.bak-80pos` and `*.bak-acav-83d9a507`.
   Hash-named backup `*.bak-scale-0a3c0d64`. Do not edit
   `/etc/sayso-satellite/config.yaml`. Do not restart unless the copy fails.
2. Repo on `ajax/nanowakeword` (no commit unless asked):
   - `satellite/models/sayso-nanowakeword.onnx` (md5 `0a3c0d645c82adbb8c1d33f39cf81017`)
   - `satellite/models/README.md` (shipped hash + scores; yaml already scale recipe)
   - `docs/HANDOFF_WAKE.md` (live staged model, scale result)
   - `docs/PLAN_NANOWAKEWORD_SCALE.md` (done; eval table)
   - this plan
3. Out of repo: host wavs/npy, eval JSON, passwords.

## Verification

- Pi: `md5sum` of `/opt/sayso-satellite/models/sayso-nanowakeword.onnx` is
  `0a3c0d645c82adbb8c1d33f39cf81017`
- Pi: `wake_word.provider` is still `livekit`; LiveKit `sayso.onnx` hash
  unchanged (`03e612d8…`)
- Repo ONNX same md5; README/HANDOFF list scale as staged, not live
