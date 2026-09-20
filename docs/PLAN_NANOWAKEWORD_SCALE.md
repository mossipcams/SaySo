# Plan: Scale NanoWakeWord to official TTS + DNN capacity

Status: done (2026-09-20). LiveKit stays live. Pi staging is a separate promote
plan (`docs/PLAN_NANOWAKEWORD_SCALE_PROMOTE.md`). Do not record more. Do not train on
`holdout_*` or miner fires. Do not stop LFM2. Do not put the 89 into
`feature_manifest.negatives` (train). Do not restore `from_list` “say so”
clones. Do not pass `-d`. Do not empty `CUDA_VISIBLE_DEVICES`.

## Outcome

Retrain official embedding DNN on host `192.168.1.140` with the NanoWakeWord
published data scale: `-G` TTS of **10,000+** SaySo clips from several voices,
matching adversarial volume, and the configuration-guide default DNN width
(`layer_size: 128`). Keep ACAV100M `oww` batch **1000**. Compare to
`83d9a507`. Do not deploy. Do not flip `wake_word.provider`.

## Why

ACAV `83d9a507` passed living-room promotion (SaySo 8/8, talk 0/8, overlapping
0/89, miner 0/74) but the recipe is still a **small-data** official run:

| Knob | ACAV `83d9a507` | NWW docs / example |
| --- | --- | --- |
| `generate_clips` / `-G` | false, no `-G` | true, `-G` |
| Train positives | 3700 wavs, 3 Piper voices | **10,000+** from several voices |
| TTS voices | lessac / amy / ryan | default Piper is **libritts_r-medium** (900+ speakers); guide example uses it |
| `layer_size` | 32 | guide default **128** (ICE floor 64) |
| `n_blocks` | 3 | keep 3 (ICE ~3–4 on this volume) |
| Adversarial / phoneme | 5000 / 3000 | scale with positives (~3× duration) |
| Bulk negatives | ACAV `oww: 1000` | keep (already correct) |
| `from_list` clones | disabled | keep disabled ([NWW #16](https://github.com/arcosoph/nanowakeword/issues/16)) |

Remaining live gaps (command 1/4, frozen-21 SaySo 6/12, “say so” 2/3) are
recall / collision, not missing ACAV. More voices and a real DNN are the
documented lever. Do **not** mix overlapping talk or miner windows into train.

Val weights stay `val_miss_weight: 2.0` / `val_fp_weight: 4.0` (ACAV). The
guide example is 4.0 / 1.0; flipping that now would spend the overlap win
before we know whether scale alone lifts command / frozen-21. Documented
deviation.

Stay `model_type: dnn`. The guide “complete example” is a Conformer; that is a
separate experiment on an 8 GB GTX 1070 with `oww` batch 1000.

## Scope

1. `satellite/models/sayso-nanowakeword.yaml`:
   - `generate_clips: true`
   - Comment: run with `-G -t -T --overwrite` (overwrite is **features**, not
     wavs; NWW names clips with timestamp + random so existing 3700 pos /
     6500 neg stay and new clips are added)
   - `layer_size: 128` (keep `n_blocks: 3`, `embedding_dim: 128`)
   - `tts_settings.models` add `en_US-libritts_r-medium` (NWW default
     multi-speaker). Omit `speaker_ids` so libritts samples randomly across
     speakers. Keep lessac / amy / ryan. Keep length/noise scales.
   - Positives: train `num_samples: 10000`, val `num_samples: 4000`
   - Adversarial: `15000`. Phoneme: `5000`. Confusable task stays `enabled:
     false`
2. Host (ops, not git):
   - Keep `sayso-official-acav-83d9a507.onnx` and `sayso-official-32eaa92e.onnx`
   - Copy yaml into `/home/ubuntu/sayso-nanowakeword/`
   - Do not delete `data/positive_recorded`, `data/negative_living_iso`,
     `data/negative_living_talk`, holdout, or miner dirs
   - First `-G` run downloads libritts_r into `NwwResourcesModel/tts_models`
     if missing (~60 MB)
   - Command:

     ```text
     nanowakeword -c satellite/models/sayso-nanowakeword.yaml -G -t -T --overwrite
     ```

     If GTX 1070 OOMs at `layer_size: 128` + `oww: 1000`, drop `layer_size` to
     **64** (ICE floor) and rerun `-t -T` only. Do not drop `oww`.
3. Score hop-feed at 0.50 vs `83d9a507` on living holdout, overlapping 89,
   miner 74, and frozen-21. Must keep talk **0/8** and SaySo **≥ 6/8**. Must
   not exceed official `32eaa92e` on overlapping (2/89) or miner (1/74).
   Target: beat ACAV on command and/or frozen-21 SaySo without losing the
   overlap/miner zeros. Do not copy to Pi unless it wins those gates.

## Files

- `docs/PLAN_NANOWAKEWORD_SCALE.md` (this file)
- `satellite/models/sayso-nanowakeword.yaml`
- `satellite/models/README.md` (recipe command includes `-G`; still “do not
  flip live provider”)
- Host-only: new wavs, regenerated features, hash-named ONNX backup (not
  committed)

## Out of scope

- Recording, miner-in-train, 89-in-train, restoring `from_list`, Conformer,
  SonicWeave swap, switching LiveKit, Pi provider flip, `-d` distill

## Result

Exported ONNX: `sayso-official-scale-0a3c0d64.onnx` (md5
`0a3c0d645c82adbb8c1d33f39cf81017`).

Hop-feed + 1 s pad at 0.50 vs ACAV `83d9a507`:

| Set | Scale `0a3c0d64` | ACAV `83d9a507` | official `32eaa92e` | 80-pos `2fe297e0` |
| --- | ---: | ---: | ---: | ---: |
| SaySo | **8/8** | 8/8 | 8/8 | 6/8 |
| SaySo+command | **3/4** | 1/4 | — | — |
| Isolated talk | **0/8** | 0/8 | 0/8 | 0/8 |
| Overlapping 89 | **0/89** (max 0.006) | 0/89 | 2/89 | 50/89 |
| Miner 74 | **0/74** (max 0.007) | 0/74 | 1/74 | 42/74 |
| Frozen-21 SaySo | **10/12** | 6/12 | — | — |

Gates hold: talk 0/8, SaySo ≥ 6/8, overlapping/miner do not exceed official.
Beats ACAV on command and frozen-21 without losing overlap/miner zeros.

## Verification

- Repo yaml: `generate_clips: true`, `layer_size: 128`, libritts in
  `tts_settings.models`, 10000 / 4000 / 15000 / 5000 counts, confusable
  `enabled: false`, `oww: 1000`, talk features under `negatives_val` only
- Host: train log reached fitting; exported ONNX md5 `0a3c0d64…` ≠ `83d9a507`
- Eval: `/tmp/evalnano/score.py` hop-feed vs ACAV backup on the four sets
  above
- `wake_word.provider` on the Pi is still `livekit`
