# Plan: ACAV-scale negatives, no “say so” clones, room-talk val

Status: implement. LiveKit stays live. Do not record more. Do not train on
`holdout_*` or miner fires. Do not stop LFM2. Do not put the 89 into
`feature_manifest.negatives` (train). Do not ship `0554f165`.

## Outcome

Retrain official embedding DNN with the published OpenWakeWord/NanoWakeWord
negative mass and without the custom `/seI soU/` clone list. Early-stop on
this-room overlapping talk (val only), not RACON miss-rate. Compare to
`32eaa92e`. Do not deploy.

## Why

Upstream ([NWW #16](https://github.com/arcosoph/nanowakeword/issues/16),
[OWW custom_model.yml](https://github.com/dscripka/openWakeWord/blob/main/examples/custom_model.yml)):

1. Do not use custom negatives that are the wake word with a spelling change.
   “say so” vs **SaySo** is that. Disable the `from_list` confusable task and
   delete existing `neg_custom*` wavs from `data/negative` so `n_features`
   is not still trained on them.
2. Bulk negatives are `openwakeword_features_ACAV100M_2000_hrs_16bit.npy`
   (~17 GB, 5.6M windows). NWW example key `oww` at batch **1000**. AE29H
   100 + RACON 90 stay as extra, not the mass.
3. Val on deployment-like speech. `negatives_val` gets features from the 89
   overlapping talk clips (**val only**). Keep Piper `targets_val`. Raise
   `val_fp_weight` (4) and drop `val_miss_weight` (2) so early-stop is not
   4× biased to recall. RACON can remain a second val key (`bv`).

Remove `hn_fp` (two-clip Whac-A-Mole). Keep `hn_iso` (23 isolated sentences).
Keep phoneme + auto_adversarial. `generate_clips: false` (no `-G`).

## Scope

1. Host download to `corpora/openwakeword_features_ACAV100M_2000_hrs_16bit.npy`.
   Need ~18 GB free. Do not convert to float32.
2. Host: `rm data/negative/neg_custom*.wav`. Confirm `n` no longer contains
   those files.
3. Host: feature job `val_talk_features` from `data/negative_living_talk`
   (89) → `val_talk_features.npy`. Manifest under `negatives_val` only.
4. `satellite/models/sayso-nanowakeword.yaml`: `oww` in negatives +
   `batch_composition.oww: 1000`; no `hn_fp`; confusable task `enabled:
   false`; `generate_clips: false`; val keys as above; `val_fp_weight: 4.0`,
   `val_miss_weight: 2.0`.
5. Command (keep `32eaa92e` backup):

   ```text
   nanowakeword -c satellite/models/sayso-nanowakeword.yaml -t -T --overwrite
   ```

   Overwrite is required so `n_features` drops the custom “say so” clips.
   Do not pass `-G` or `-d`.

6. Score hop-feed on `holdout_living`, miner 74, living_talk 89 vs `32eaa92e`.
   Must keep talk 0/8 and SaySo ≥ 6/8. Target: overlapping talk and miner
   ≤ official 2/89 and 1/74. Do not copy to Pi unless it wins.

## Files

- `docs/PLAN_NANOWAKEWORD_ACAV.md` (this file)
- `satellite/models/sayso-nanowakeword.yaml`
- Host-only: 17 GB npy, regenerated features (not committed)

## Out of scope

- Recording, miner-in-train, 89-in-train, `0554f165`, switching LiveKit
- Changing the wake phrase

## Handoff continuation (2026-09-20 UTC)

Scope: finish the existing ACAV run and evaluate it against the preserved
`32eaa92e` model. Reuse the host scorer and current recipe; no new mixtures.

Files to touch: this plan and `docs/HANDOFF_WAKE.md` for verified results;
host-only evaluation reports and a hash-named backup of the ACAV model.
Repair the existing host waiter only if it fails. No runtime code changes.

Verification:

1. Confirm the host recipe matches the local YAML, prohibited training sources
   remain excluded, the official backup is intact, and ACAV mmap loads with
   shape `(N, 16, 96)` before training starts.
2. Reproduce official scores on all 23 living-room holdouts, 89 overlapping
   talk clips, and 74 miner clips using the existing hop scorer with 1 s pad.
3. Check transform/training completion and score the new, hash-named ONNX
   identically at threshold 0.50. Require SaySo >= 6/8 and talk 0/8; compare
   overlapping talk and miner against official 2/89 and 1/74.
4. Record results and any remaining blocker in the handoff. Do not switch the
   satellite provider; if the new model wins, only stage a backed-up copy as
   allowed by the handoff. Live microphone verification needs the user present.
