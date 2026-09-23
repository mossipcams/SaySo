# Log letter-spaced TV as a model defect

Scope: record the STT `TV` → `T V` mismatch as a tracked model defect, the same way #94 / #97 are tracked. No generator change, no corpus rebuild, no training launch, no eval/runtime edits, no HA alias change.

## Defect

Spoken “turn on the TV” can reach SaySo as `turn on the T V`. The live entity is named `TV` with alias `TV`. Home Assistant’s matcher looks that string up; `T V` is not it, so the call raises `MatchFailedError`.

This is adjacent to #94, not the same bug:

| | #94 | This defect |
|---|---|---|
| Bad `name` | `Living room TV` | `T V` |
| Cause | corpus taught `f"{area} {role}"` | STT/model copies letter-spaced acronym; gold never shows the split |
| Gold | `name="TV"` (bare) is already the right label | still `name="TV"`; the *utterance* is what must vary |

## Evidence already in this checkout

- `training/fixtures/real_home.json` → `media_player.living_room_tv` name `TV`, aliases `["TV"]`, area Living Room.
- Gold copies `entity["name"]`; `refers_to` is `\bTV\b`, which does not match `T V`.
- `training/generators/stt_noise.py` splits `outlet` → `out let` and has no `TV` / `T.V.` / `teevee` variant.
- Shipped v3 corpora and the 120-row v4 smoke set have no `T V` labels.
- Live HA traces on 192.168.1.35 showed `HassTurnOff` then `MatchFailedError`. The tool `name` argument was not captured here; confirm `T V` vs `Living room TV` from the conversation trace before closing.

## Files to touch

- `docs/SAYSO_LFM_TRAINING_PLAN.md` — one row in “Defects that determine the data”, linked to the new GitHub issue.
- GitHub issue on `mossipcams/SaySo` (create unless an existing issue already covers letter-spaced `TV`).

Out of scope: `stt_noise.py`, recipes, evals, HA aliases, training.

## Future data fix (do not implement now)

Corrupt the utterance to `T V` / `T.V.` / `teevee`; keep gold `name="TV"`. Same pattern as `out let`. Optional runtime bandage: alias `T V` on the media player.

## GitHub issue

https://github.com/mossipcams/SaySo/issues/102

## Verification

- `gh issue view 102` shows a model-defect issue describing letter-spaced TV and distinguishing #94.
- Training-plan table row links that issue.
- `git diff --stat` is docs only; no generator/eval/runtime files.
