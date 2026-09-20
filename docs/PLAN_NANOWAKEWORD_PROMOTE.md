# Plan: Promote ACAV Nano `83d9a507` and open a PR

Status: implement. Do not flip the live satellite off LiveKit. Do not
train. Do not commit wavs, npy corpora, or `context.json`. Do not put
passwords in the repo.

## Outcome

Stage the promoted NanoWakeWord ONNX on the Pi as
`/opt/sayso-satellite/models/sayso-nanowakeword.onnx` (backup 80-pos) and
open a PR that ships the same ONNX plus the ACAV recipe and wake eval
harness. `wake_word.provider` stays `livekit`.

## Why this model

Living-room hop-feed at 0.50 (promotion gates): SaySo 8/8, isolated talk
0/8, overlapping talk 0/89, miner 0/74. Beats 80-pos `2fe297e0` and
official `32eaa92e` on overlapping talk and miner. Frozen-21 (old room)
is not the promotion suite.

## Scope

1. Pi (ops): copy `sayso-official-acav-83d9a507.onnx` over
   `sayso-nanowakeword.onnx`. Keep `*.bak-80pos`. Hash-named backup
   `*.bak-acav-83d9a507`. Do not edit `/etc/sayso-satellite/config.yaml`.
   Do not restart unless the copy fails.
2. Repo PR on `ajax/nanowakeword` (merge/rebase `origin/main` if needed):
   - `satellite/models/sayso-nanowakeword.onnx` (md5 `83d9a507…`)
   - `satellite/models/sayso-nanowakeword.yaml` (ACAV recipe already in tree)
   - `satellite/models/README.md` (Nano is no longer a smoke/`-G` config)
   - `satellite/sayso/wake/nanowakeword.py` + `test_nanowakeword.py`
   - `satellite/eval/compare_providers.py` + `test_compare_providers.py`
   - `docs/HANDOFF_WAKE.md` updated with promotion result
   - this plan
3. Out of PR: scratch `docs/PLAN_NANOWAKEWORD_*.md` except ACAV + this
   file, wavs, npy, host-only eval JSON.

## Verification

- Pi: `md5sum` of live Nano path is `83d9a507e35530adffaf55465a3b1478`;
  LiveKit `sayso.onnx` still `03e612d8…`; `provider: livekit`; service
  active.
- `python3 -m pytest satellite/sayso/wake/test_nanowakeword.py satellite/eval/test_compare_providers.py -q`
- PR URL returned. Default provider in docs/config remains LiveKit.
