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

## Moonshine second stage (optional, Koda)

Set `wake_word.moonshine_verifier: true` to run `MoonshineVerifier`
(`sayso/wake/moonshine_verifier.py`) after LiveKit fires. It transcribes the
last 1.2 s of the fired window (+0.5 s zero pad) with Moonshine v2
tiny-streaming, key-term biased toward the phrase, and passes only if the text
contains the phrase or one of `wake_word.moonshine_accept` (the phrase is always
accepted; for Koda use `[koda, kota, coda, kohda, cota, koder]`). Mutually exclusive
with `wake_word.verifier`. `moonshine_boost` (default 3.0) is the bias strength;
5+ hallucinates the phrase. Load failure fails closed. The service never downloads
the model: `wake_word.moonshine_cache_dir` (required) must hold a model provisioned
once with network, and its files must hash to the SHA-256 pinned in
`moonshine_verifier.py` or the verifier refuses to load:
`python -m moonshine_voice.download --stt --language en --model-arch 2 --root <cache_dir>`.
Transcripts are logged at DEBUG only.

Offline numbers (Pi windows, 2026-10-06, boost 3, `/tmp` scripts since removed):
27 real Koda wakes -> 17-19 pass (~65-70%); 57 TV false fires -> 2-3 pass;
~98% of non-wake windows rejected. About 0.7 s per check on the Pi 4, run in
the inference worker thread. Real-wake labels were unverified, so treat recall
as provisional. A veto no longer starts the refractory (see `livekit.py`).
