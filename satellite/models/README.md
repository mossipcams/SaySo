What ships on the satellite. Training (LiveKit, host, data, recipes):
[training/wake/README.md](../../training/wake/README.md). Design:
[docs/SAYSO_WAKE_WORD_TRAINING_PLAN.md](../../docs/SAYSO_WAKE_WORD_TRAINING_PLAN.md).

Place this LiveKit-exported Sayso classifier on the satellite:

  /opt/sayso-satellite/models/sayso.onnx

It detects the spoken phrase "Sayso" only. The generate-first model runs
**single-stage LiveKit** by default — omit `wake_word.verifier`. Do not use the
generic trainer threshold (~0.05). Do not substitute hey_livekit, hey_jarvis,
or another model.

## SaySo Voicev1 champion (pass 2)

- Artifact: `/opt/sayso-satellite/models/sayso-voicev1.onnx`
- SHA-256: `f1313f031ae5c05867f4d716c9e47a0c459e0c6b586400335cce5ae8e72d2199`
- Selected threshold: **0.22** (pass-2 LiveKit validation `optimal_threshold`)
- Pi eval: **24/30** silent30 positive cases detected at 0.22; six missed.

Voicev1 is active on the Pi at threshold `0.22`; `sayso-satellite.service` was
verified active after the config switch and restart. The previous model and
config are backed up at `/home/pi/sayso-voicev1-rollback-20260926/`.

Previously shipped on the Pi: living2 (`b840f51f312abcd5b205e1fc1e32b2ed`) at
threshold **0.28** with legacy `sayso-verifier.npz`
(`0c632e778ca263e51c92d9ca95f451af`). That mel artifact is **not** a phrase
check for the generate-first export; the satellite ignores it and stays
single-stage when it is configured.

## Optional embedding verifier (second stage)

The generate-first model already classifies from (16, 96) speech embeddings.
A compatible second stage must score those same embeddings — not a parallel
collapsed-mel pass.

Legacy `sayso-verifier.npz` (`feature_kind` absent, treated as `mel_union`) is
a this-room recording-envelope logistic on mel mean+std, fit on 50 Snowball
clips vs 19 `verifier_live_fp` windows. It is **not** a phrase check for the
generate-first export and does **not** AND-gate LiveKit when configured; the
satellite logs once and runs single-stage.

A future compatible artifact sets `feature_kind=speech_embedding` in the npz and
carries logistic weights for concat(mean, std) over the last 16 speech
embeddings (192-d). None is shipped yet — do not invent one in this tree.

When `wake_word.verifier` points at a compatible npz, LiveKit remains the
primary scorer and the embedding verifier must also pass before the satellite
fires. Mine on the LiveKit score as today — the veto runs after mining, not
before. Omit `verifier` for single-stage LiveKit (generate-first default). If
`verifier` is set but the file is missing, wake detection fails closed.

### living2 + mel verifier (historical Pi operating point)

Frozen Google speech embeddings separate this-mic SaySo from overlapping talk
(AUROC 0.97) but not from the 19 live false wakes. The mel logistic fit on
50 vs 19 achieved mel AUROC 1.0 on that narrow set. Do not dump those FPs
into the classifier.

Host AND-gate (hop-scan at living2 **0.50** / mel verifier **0.445**):

| Set | Result |
| --- | ---: |
| living-room SaySo | **6/8** |
| isolated talk | **0/8** |
| overlapping talk (89) | **0/89** |
| miner party (74) | **0/74** |
| verifier_live_fp (19) | **0/19** |

Pi operating point (living2 **0.28** / mel verifier **0.445**):

| Set | Result |
| --- | ---: |
| living-room SaySo | **7/8** |

Miss on Pi: `live_sayso_05` = 0.179 collides with `live_talk_02` = 0.178
(verifier blesses both). 8/8 is blocked by that collision.

The 19 are verifier-train, not an unbiased FP set. Best unbiased-FP backup
on the Pi is `sayso.onnx.bak-03e612d8`.

## DMA-KWS second stage (optional, Koda)

Set `wake_word.dma_kws_model` to run `DmaKwsVerifier` (`sayso/wake/dma_kws_verifier.py`) after
LiveKit fires. It crops 0.8 s ending 0.2 s after the last loud frame of the fired window, computes
an 80-bin log-mel filterbank (numpy, matches torchaudio), and scores it against the keyword
phonemes with the DMA-KWS Stage II text-to-audio matcher (4.1M params, ONNX, 1 thread). The score is
the best over `wake_word.dma_kws_phonemes`; the wake passes at `>= dma_kws_threshold`. Mutually
exclusive with `wake_word.verifier`. Load failure fails closed.

```yaml
wake_word:
  dma_kws_model: /opt/sayso-satellite/models/dma-kws-stage2.onnx
  dma_kws_phonemes: ["K OW1 D AH0", "K OW2 D AH0", "K OW1 D AA0"]   # exactly 4 ARPAbet phonemes each
  dma_kws_threshold: 0.98
```

**Provisioning.** The service never downloads a model: the file must hash to the SHA-256 pinned in
`dma_kws_verifier.py`. Build it on a dev machine with `scripts/export_dma_kws_onnx.py` (source:
github.com/aizhiqi-work/DMA-KWS at commit 207d056, checkpoint `155k-v2-ft.ckpt`, torch 2.14.1 /
onnx 1.23.2 give a reproducible file) and copy it to the satellite. The upstream repo has no
LICENSE file, so the model is not committed here. The keyword length (4 phonemes) and crop (78
frames) are baked into the export; another phrase needs a re-export.

**Offline numbers** (851 Pi windows, 2026-10-06; 27 command-followed real Koda wakes, 808 other
windows incl. 57 TV false fires; labels unverified, thresholds tuned on the same data):

| threshold | real wakes | other windows rejected | TV false fires rejected |
| ---: | ---: | ---: | ---: |
| 0.90 | 22/27 | 94.2% | 84.2% |
| 0.95 | 20/27 | 96.0% | 91.2% |
| 0.98 (default) | 18/27 | 97.8% | 94.7% |
| 0.99 | 14/27 | 98.6% | 94.7% |
| 0.995 | 12/27 | 99.3% | 96.5% |

AUC 0.956. Scores saturate near 1.0, so recall moves fast between 0.97 and 0.99: tune on verified
data before relying on a threshold. About 54 ms per check on a Pi 4 (1 thread; fbank 4 ms, model
48 ms), run in the inference worker thread. A veto no longer starts the refractory (see
`livekit.py`). Transcript-free: only the score is logged, at DEBUG.
