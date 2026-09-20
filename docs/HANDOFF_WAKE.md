# Handoff: living-room wake word (2026-09-19 evening)

For whoever picks this up. Wavs are off git. Do not train on holdout dirs
or miner fires. Do not stop LFM2 on the train host. Do not put passwords
in the repo. Phrase is **SaySo**, not “say so”.

Satellite is a thin LVA overlay. It does not do STT, NLU, or actions.

## 1. Live system

| Machine | Role |
| --- | --- |
| Satellite `192.168.1.54` (`pi`, SSH port **2222**) | Capture, live wake, miner |
| Train host `192.168.1.140` (`ubuntu`) | `/home/ubuntu/sayso-nanowakeword/` data + GTX 1070 |
| Worktree | `ajax-nanowakeword` (branch `ajax/nanowakeword`) |

**Last live configuration** (Pi is up; ACAV Nano staged on disk, LiveKit still
the loaded provider):

- `provider: livekit` (unchanged)
- LiveKit model: `/opt/sayso-satellite/models/sayso.onnx` (`03e612d8671df941bd63c5a982d37ad4`)
- threshold `0.50`, refractory `2.0 s`, `mine_threshold: 0.45`
- miner: `/var/lib/sayso-satellite/wake-mining`
- Nano ACAV staged, **not loaded**:
  `/opt/sayso-satellite/models/sayso-nanowakeword.onnx` (`83d9a507e35530adffaf55465a3b1478`)
- Nano 80-pos backup: `sayso-nanowakeword.onnx.bak-80pos` (`2fe297e0b06e4660115d929cbca790a7`)
- Hash-named ACAV backup: `sayso-nanowakeword.onnx.bak-acav-83d9a507`
- Nano-era config backup: `/etc/sayso-satellite/config.yaml.bak-nano-80pos`

ACAV `83d9a507` beats 80-pos and official `32eaa92e` on overlapping talk and
miner at 0.50 (SaySo 8/8, talk 0/8, overlapping 0/89, miner 0/74). Copying an
ONNX is not the same as flipping `provider` — leave LiveKit live until someone
explicitly switches.

## 2. What we learned

Nano 80-pos fires constantly on overlapping living-room speech at **0.90–0.99**.
Raising threshold does nothing. Isolated `live_talk_*` 0/8 hid that.

Kitchen-sink mixes (89 overlapping slices and/or miner windows as a large
fraction of the batch) cut miner FPs and **destroyed** isolated-sentence
rejection (talk 4–6/8). Do not do that again.

The published NanoWakeWord / OpenWakeWord method is not “this room’s wavs”:

- 10,000+ wake clips from several voices (we have 50 Snowball + Piper).
- Bulk negatives: **~2,000 h ACAV** embeddings, ~1000 per batch
  ([openwakeword_features](https://huggingface.co/datasets/davidscripka/openwakeword_features)).
  AE29H (~29 h) is a small extra, not the mass.
- RACON (~11 h) is mainly **validation**, not a substitute for ACAV.
- Maintainer ([NWW #16](https://github.com/arcosoph/nanowakeword/issues/16)):
  do **not** use custom negatives that are the wake word with a spelling
  change. Our `from_list` of “say so” / “says so” / … is that. Deleted
  `neg_custom*.wav` from `data/negative` (0 left).
- FP mining is hours of audio into a large `.npy`, not two 2 s clips.

**SaySo** is two syllables and English “say so”. Industry guides and NWW
#16 all say that collision may not be separable. Isolated
`live_neg_sayso_03` at ~0.99 on the official model is that, not a YAML typo.

## 3. Models on the train host

All under `/home/ubuntu/sayso-nanowakeword/output/sayso/model/`.

| File | md5 prefix | Notes |
| --- | --- | --- |
| `sayso-official-32eaa92e.onnx` | `32eaa92e` | **Best Nano so far.** Official recipe (AE29H+RACON+Piper). Keep. |
| `sayso-official-hn-0554f165.onnx` | `0554f165` | Official + 2 FP windows (`neg_talk_007/035`). **Worse. Do not ship.** Also currently copied to `sayso.onnx`. |
| `sayso-like80.onnx` | `df0dfdd3` | Last 80-pos-style mix on this host. Ignore. |
| Pi 80-pos | `2fe297e0` | On the satellite only. |

`sayso.onnx` on the host **is 0554f165** right now. Do not deploy it. The
ACAV waiter will overwrite `sayso.onnx` again when train starts; `32eaa92e`
backup must stay.

### Official `32eaa92e` vs hn `0554f165` vs 80-pos

Hop-feed + 1 s pad, fire if max ≥ 0.50. Scorer:
`/tmp/evalnano/score.py` on the host (no LiveKit there).

| Set | 80-pos `2fe297e0` | official `32eaa92e` | hn `0554f165` |
| --- | ---: | ---: | ---: |
| SaySo (`live_sayso_*`) | 6/8 | **8/8** | 8/8 |
| SaySo+command | 1/4 | 2/4 | 3/4 |
| Isolated talk | 0/8 | **0/8** (max ~0.003) | 0/8 |
| “say so” | 0/3 | 1/3 | **2/3** |
| Miner party 74 | 42/74 | **1/74** | **6/74** |
| Overlapping 89 | 50/89 | **2/89** | **4/89** |

80-pos and official rerun on all three sets at 2026-09-20 00:50 UTC;
holdout results reproduced. Raw per-clip scores and summaries are saved on
the host as `acav-eval-2fe297e0.json` and `acav-eval-32eaa92e.json` in
`/home/ubuntu/sayso-nanowakeword/`. The 80-pos scorer copy came from local
`/tmp/nano-80pos.onnx`, verified as `2fe297e0b06e4660115d929cbca790a7`.

hn_fp memorized 007/035 (~0.003) and lit 016/026/038/049 plus extra miner
windows at 0.73–0.99. Whac-A-Mole. Isolated talk 0/8 is not the complaint.

LiveKit on holdout: SaySo **0/8**, talk 0/8, quiet on miner sample. Live
recall in this room since the switch is untested.

## 4. ACAV retrain (done)

Plan: `docs/PLAN_NANOWAKEWORD_ACAV.md`. Promotion plan:
`docs/PLAN_NANOWAKEWORD_PROMOTE.md`.

Recipe: `satellite/models/sayso-nanowakeword.yaml` — `oww` ACAV100M batch
**1000**, no `-G`, no `from_list` “say so” clones, overlapping talk val-only.

**Result:** `sayso-official-acav-83d9a507.onnx` (md5 `83d9a507…`).

Hop-feed + 1 s pad at 0.50 vs prior models:

| Set | ACAV `83d9a507` | official `32eaa92e` | 80-pos `2fe297e0` |
| --- | ---: | ---: | ---: |
| SaySo | **8/8** | 8/8 | 6/8 |
| Isolated talk | **0/8** | 0/8 | 0/8 |
| Overlapping 89 | **0/89** | 2/89 | 50/89 |
| Miner 74 | **0/74** | 1/74 | 42/74 |

Staged on Pi at `/opt/sayso-satellite/models/sayso-nanowakeword.onnx`.
80-pos backed up as `*.bak-80pos`. Shipped in repo on branch
`ajax/nanowakeword`. **`provider: livekit` unchanged.**

## 5. Eval how-to

Host (Nano only):

```bash
cd /tmp/evalnano
. /home/ubuntu/sayso-nanowakeword/.venv/bin/activate
python score.py --model /path/to.onnx \
  --audio-dir /home/ubuntu/sayso-nanowakeword/data/holdout_living
```

Same scorer was wrapped to summarize miner 74 + living_talk 89. Isolated
`live_talk_*` is **not** the FP test that matches the live complaint.
Overlapping 89 and miner 74 are.

Pi (both engines), if it is up:

```bash
PYTHONPATH=/tmp/evalwrap \
  /opt/sayso-satellite/.venv/bin/python -m satellite.eval.compare_providers \
  --audio-dir DIR --livekit PATH.onnx --nano PATH.onnx
```

`satellite/eval/compare_providers.py`: LiveKit 2 s windows; Nano hop-feed + 1 s pad.
Tests: `satellite/eval/test_compare_providers.py`.

## 6. Data (host `/home/ubuntu/sayso-nanowakeword/data/`)

**Never train on**

| Set | n | Path | What |
| --- | ---: | --- | --- |
| `holdout_living` | 23 | `data/holdout_living/` | `live_sayso_*` 8, `live_cmd_*` 4, `live_talk_*` 8, `live_neg_sayso_*` 3 |
| `holdout_eval` | 21 | `data/holdout_eval/` | Old room. Wrong after the move. |
| Miner party | 74 | `data/negative_miner_party/` | Eval only. Exact fire windows. |
| Unsure miner | — | Pi spool | Do not train. |

**Val only (ACAV run), not train negatives**

| Set | n | Path |
| --- | ---: | --- |
| Friends talking, 3 min, 2 s slices | 89 | `data/negative_living_talk/` |

**Train (current YAML)**

| Set | n | Notes |
| --- | ---: | --- |
| `data/positive` | ~3700 | Piper SaySo (incl. older 1200) |
| `data/positive_val` | 2000 | Piper val |
| `data/positive_recorded` | 50 | This-room wakes |
| `data/negative` | 6500 | Adversarial TTS; custom “say so” **removed** |
| `data/negative_phoneme` | 3000 | Phoneme adversarial |
| `data/negative_living_iso` | 23 | Isolated prompted sentences |
| `data/negative_fp_talk/` | 2 | 007+035 copies. **Not in current YAML.** Do not add back. |

Noise for aug: `/home/ubuntu/sayso-wakeword/data/backgrounds/noise/{free-sound,sound-bible}` (774 files). Point at those subdirs; the parent dir has no top-level wavs.

Corpora: `AE29H_float32.npy`, `RACON_11h_v1.npy`, ACAV download in progress.

NWW venv: `2.1.3`, torch `2.4.1+cu118` (needed `add_safe_globals` for generate). GTX 1070.

## 7. Do not

- Record more (user said no).
- Train on holdout / miner / unsure miner.
- Put the 89 in `feature_manifest.negatives` (train). Val only.
- Oversample 2 FP windows (`hn_fp` / `0554f165`).
- Restore `from_list` “say so” clones.
- Stop LFM2. Empty `CUDA_VISIBLE_DEVICES`. Pass `-d` (we ship the teacher).
- Commit wavs or `context.json`.
- Flip the satellite off LiveKit until eval says so.
- Treat prompted talk 0/8 as “FPs are fixed.”

## 8. Next session checklist

1. Pi back up? Confirm `sayso-nanowakeword.onnx` md5 `83d9a507…` and LiveKit
   `sayso.onnx` still `03e612d8…`; `provider: livekit`.
2. Optional: `compare_providers` on holdout / 89 / miner 74 with both ONNX paths.
3. Live recall on LiveKit still untested — say **SaySo** at the Snowball and
   read the journal.
4. To try Nano live: flip `wake_word.provider: nanowakeword` explicitly; do
   not assume staging the ONNX switched the provider.
