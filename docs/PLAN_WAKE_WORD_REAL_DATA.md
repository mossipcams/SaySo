# Wake word: what the real satellite data says, and how to improve

Status: analysis complete, both blocking decisions made. Phrase decided as
`/seɪ soʊ/` — **"SaySo" or "Say So"** (both spellings are the same utterance;
`config.py` pins `wake_word.phrase` to the string `SaySo`, and the acoustics do
not depend on the spelling). Wake rule decided: **isolation** — if another word
is in front of the wake word it is not a wake (item 2b, resolved below). Config
and eval changes are required to match those decisions before any training run.

## This revision

Scope: encode the isolation rule in the places that decide what the model is
taught and what it is judged on. No runtime wiring changes — the classifier is
the gate, so the rule is a labelling rule, not a new code path in the satellite.

Files touched:

- `docs/PLAN_WAKE_WORD_REAL_DATA.md` — this revision section, item 2, item 2b,
  item 4, verification, open questions.
- `docs/WAKE_WORD_DATA.md` — wake rule in the decision record; correct the
  "no real wake audio exists" framing now that the spool exists.
- `satellite/models/sayso-training.yaml` — both realizations as targets; bare
  phrase removed from negatives; carriers kept as negatives.
- `satellite/eval/cases.json`, `satellite/sayso/wake/eval.py`,
  `satellite/eval/README.md` — `negative_natural_say_so` becomes the carrier
  category `negative_say_so_carrier`; add the two-word positive and a carrier
  case.
- `satellite/models/README.md` — the two stale references above.
- `satellite/sayso/wake/mining.py` — "mined means negative" framing.
- `scripts/wake_label_spool.py` (+ colocated test) — apply the isolation rule to
  a spool + transcripts and emit a review ledger. This is item 3's first step
  (label the spool) with the rule the labels must follow; it writes no sidecar
  labels, because labelling is a listening decision.
- `scripts/wake_mine_report.py` (+ new colocated test) — read that ledger so the
  review loop is one pass instead of two tools, and `--label-cluster` so a
  verdict lands on the whole utterance rather than one 160 ms window.
- `satellite/models/wake-spool-labels-20260914.json` — ledger for the 152 real
  windows.
- `scripts/wake_prepare_real_data.py` (+ colocated test) — turn the ledger into
  the two corpora: an eval root outside the repo, and seeded
  `output/sayso/{positive,negative}_train` for the retrain. Written around three
  upstream behaviours that silently drop real audio; see
  `satellite/models/README.md`.
- `satellite/models/wake-held-out-20260914.json` — the clusters the eval corpus
  reserves, excluded from training and from the `*_test` threshold-selection
  splits.
- `.gitignore` — recorded audio and livekit-wakeword run artifacts stay out of
  the repo (this one is public), plus /data/ and /output/.

Verification:

- `python3 -m pytest satellite/sayso/wake/test_eval.py satellite/eval/test_run.py scripts/test_wake_label_spool.py -q`
- `python3 scripts/wake_label_spool.py --spool <spool> --transcripts <json>`
  reproduces the ledger committed above from the real spool.
- `python3 scripts/wake_prepare_real_data.py eval-corpus ...` populates 6 of 8
  cases and reports the 2 it cannot; `training-data --dry-run` reports 38 real
  positive and 23 negative utterances with 5 clusters held out.
- `python3 scripts/wake_bench.py` stays green (unchanged path).

Outcome (2026-09-14): done as scoped. The corpus now has a carrier category and
two per-realization positive cases, the training config validates against the
upstream `WakeWordConfig` schema with no target/negative overlap, and the spool
ledger is committed with 16 of 66 clusters flagged for listening. Still open in
this plan: `satellite/eval/audio/` is empty, so every eval case skips, and the
real clips are not yet in `data/positive_train`/`data/negative_train`.

Baseline measured with a corpus built by `wake_prepare_real_data.py eval-corpus`
and scored with the shipped `sayso.onnx` at threshold 0.5:

| category | cases | detection_rate |
|---|---|---|
| `negative_say_so_carrier` | 2 | 0.00 |
| `negative_tv_conversation` | 1 | 0.00 |
| `negative_distance_noise` | 1 | 0.00 |
| `positive_say_so_two_word` | 1 | 1.00 |
| `continuous_command` | 1 | 0.00 |
| `positive_sayso` (fused) | 2 | skipped — no capture |

Read it as the plan already reads the real data: **the negatives are clean and
recall is the defect.** Both carriers stay under threshold, so the isolation rule
is not what is broken today; the loudest isolated capture (0.744) fires, while a
genuine wake-plus-command at 0.4825 does not. A retrain has to lift that without
moving the four negatives.

### Retrain run, 2026-09-14 (in progress)

Run on the Mac with the config committed above, seeded from the ledger. Two
decisions were forced by measurement rather than preference:

- **`tts_backend` VoxCPM → `piper_vits`.** VoxCPM2 measures **184 s per clip** on
  an M1 (CPU-bound; 136 diffusion steps), so the 20000-clip set is ~42 days.
  Piper measures **0.21 s per clip**, ~1.2 h for the same set. The cost is
  speaker diversity, which the config previously bought deliberately; the real
  positives are one speaker either way, and the pipeline can be re-run with
  VoxCPM on a CUDA host from this same config.
- **ACAV100M is included.** `--skip-acav` was only in play while the host showed
  20 GB free against a 17.5 GB features file; once space was free the pool was
  downloaded, so the fpph anchor is the reviewed 2000 hours. Nothing else about
  the negative mix changed.

Two further host findings, both measured, are recorded in
`satellite/models/README.md` rather than here: `tts_batch_size` had to drop from
50 to 8 (the VITS decoder pads every sequence in a batch to the longest, so
batching a two-word phrase costs 30x), and the host's memory pressure made Metal
fail allocation and wedge `generate`, so the run is CPU-only.

### Training for the live channel, not for clean TTS

The first seeding attempt was 91% synthetic positives, which trains the wrong
channel: the deployed failure is real speech through the satellite's own
processing, not Piper's output. `wake_prepare_real_data.py` now seeds for the
live path, and the evals above are what it will be judged on.

- **Every scored window, not one per utterance** (`--all-windows`): 82 positive
  and 55 negative windows, at the alignment the production pipeline actually
  scored them at.
- **Real captures at ~38% of the positive pool**: `--repeat 60` gives 4920 real
  positive clips against 8000 synthetic. Repeats are not padding — each copy
  gets its own augmentation round, so repetition buys channel variation, not
  duplicates.
- **The live room is an augmentation background** (`--live-backgrounds`): 51
  phrase-free real windows go to `data/live-backgrounds`, which the config lists
  beside MUSAN, so positives are mixed under the actual microphone, reverb and
  household clutter rather than generic noise alone. Carrier and undecidable
  clusters are excluded — mixing a carrier under a positive would teach the
  phrase next to itself — as are the held-out eval clusters.

Still synthetic-only: the *isolated fused* realization (no real capture exists)
and the carrier negatives (TTS "if you say so" and friends). Both are recorded
as gaps in the corpus rather than papered over.

Source data: `/var/lib/sayso-satellite/wake-mining` on the living-room satellite
(Pi 4, `192.168.1.54`), drained 2026-09-14. 152 windows / 66 utterance clusters
over ~21 h, plus 102 `stt_capture` command takes. Model under test is
`satellite/models/sayso.onnx`, md5 `1e26512018007484a917b1e2449405d7` — byte
identical to the deployed `/opt/sayso-satellite/models/sayso.onnx`.

## Verdict

**The corpus and the eval contradict the phrase. The phrase is right; the labels
are wrong.**

The phrase is `/seɪ soʊ/` — "SaySo" or "Say So", the same utterance either way.
The project currently trains and evaluates against the opposite label as well:

- `satellite/models/sayso-training.yaml` → `target_phrases: ["Sayso"]`, then
  `custom_negative_phrases` opens with `"say so"`, `"saying so"`,
  `"if you say so"`, `"just say so"`, `"you don't say so"` — literally the
  positive phrase and supersets of it.
- `satellite/eval/cases.json` → `positive_sayso` (`expect_detection: true`)
  **and** `negative_natural_say_so` ("Natural speech containing say so without an
  intentional wake", `expect_detection: false`).

Those two labels describe the same acoustic event. Given the phrase decision, the
second one is simply wrong and must go: under "the phrase is SaySo / Say So",
natural "say so" **is** a wake.

So the model is not under-trained and the corpus is not too small. It was asked
to separate a sound from itself, and it responded correctly to an impossible
brief by parking the whole /seɪ soʊ/ family on top of the decision boundary.
Nothing about volume, threshold or model size can fix a contradictory label;
fixing the label is what unblocks everything else.

### Proof 1 — the target scores below its own posted negatives

Clean TTS rendered through the model, best alignment over all window offsets
(threshold in production is **0.5**). "Role today" is what the config says;
"role under the decided phrase" is what it should say, since both realizations of
`/seɪ soʊ/` are wakes:

| TTS phrase | score | role today | role under decided phrase |
|---|---|---|---|
| "stay so" | **0.792** | NEGATIVE | NEGATIVE (near-miss) |
| "say so" | **0.715** | NEGATIVE | **POSITIVE** (2-word realization) |
| "hey so" | **0.690** | NEGATIVE | NEGATIVE (near-miss) |
| **"Sayso"** | **0.647** | POSITIVE | **POSITIVE** (1-word realization) |
| "said so" | 0.500 | NEGATIVE | NEGATIVE (near-miss) |
| "says so" | 0.473 | NEGATIVE | NEGATIVE (near-miss) |
| "so so" | 0.458 | NEGATIVE | NEGATIVE (near-miss) |
| "say something" | 0.393 | NEGATIVE | NEGATIVE (unrelated) |
| "see you soon" | 0.313 | — | NEGATIVE (measured FP) |
| "if you say so" | 0.141 | NEGATIVE | **open — item 2b** |
| "lasso" | 0.127 | NEGATIVE | NEGATIVE (unrelated) |
| "turn on the TV" | 0.005 | — | NEGATIVE (unrelated) |

The positive class is therefore 0.647–0.715, and **the top-scoring event of the
entire set is a near-miss negative: "stay so" at 0.792**. The threshold 0.5
slices straight through the positive class it is supposed to accept. Note
"if you say so" at 0.141 scores like an unrelated word, not like a carrier — the
model has already, accidentally, taken the isolation reading for that phrase,
which is direct evidence for item 2b.

Under the current labels the model cannot tell which side of the line is which,
because it was told both sides are the same sound.

Caveat on method: the shipped model (2026-09-03) predates `sayso-training.yaml`
(2026-09-10), so it was trained with `custom_negative_phrases` empty and the
upstream default adversarial fillers (`hello, okay, hey, stop, go, yes, no`). The
collision therefore exists **without** the "say so" negatives — which means the
retrain config would make it strictly worse, not better. Removing the
contradiction is therefore necessary but not sufficient: the shipped model's
weak discrimination inside the /seɪ soʊ/ ~ /steɪ soʊ/ ~ /seɪ suːn/ neighbourhood
has to be retrained away, which is what the real data is for.

### Proof 2 — the real satellite data agrees, on every axis

Utterance-level analysis of all 66 clusters (windows grouped by ≤3 s gaps,
transcribed with Whisper large-v3-turbo, labels from transcript):

| | genuine "SaySo" | real negatives |
|---|---|---|
| clusters | 44 | 22 |
| fire at t=0.5 | 28 (64%) | **0** |
| best-window max | 0.744 | 0.423 |
| best-window p90 | 0.684 | 0.292 |
| best-window mean | 0.511 | 0.202 |

Read that table carefully, because it inverts the premise in `WAKE_WORD_DATA.md`.
One caveat before quoting it: those two columns come from a regex over "say so"
*anywhere* in the window, which is the containment reading. Under the isolation
rule 4 of the 44 genuine clusters move to the negative side (carriers and windows
the transcripts cannot settle), so the rule-consistent split is 40 / 25 / 1 — see
`satellite/models/wake-spool-labels-20260914.json`. The table's conclusion is
unchanged either way: false positives are not the production problem, false
rejects are.

- **False positives are not the real production problem.** In ~21 h of living
  room audio the worst negative cluster reached 0.423 — under the threshold. At
  t=0.5 the FP rate is zero.
- **False rejects are.** 16 of 44 genuine wake utterances (36%) never produce a
  single window above 0.5. The user says "SaySo" and nothing happens.

Reproduction of the deployed scores is exact: re-scoring the 152 mined WAVs
through `livekit.wakeword` 0.2.1 gives `max|diff| = 1e-6` against the recorded
sidecar scores, so these numbers describe the shipped model, not a proxy.

### Proof 3 — the model gets one decision per utterance, never two

Per-utterance window score profiles (windows ≥ 0.5):

| chances per genuine utterance | 0 | 1 | 2+ |
|---|---|---|---|
| utterances | 16 | 28 | **0** |

Not one genuine utterance in 21 h ever produced two above-threshold windows.
A sliding-window detector is supposed to answer the same event several times in
a row; that redundancy is the entire reason for the 160 ms hop. Here the
response is a single spike, so there is exactly one chance to fire and a dropped
or unlucky window is an unconditionally missed wake.

Measured on clean TTS the activation plateau is **400–480 ms wide above 0.5**
(e.g. 0.680 at offset 0 ms, 0.604 at 440 ms, 0.515 at 480 ms, 0.088 at 560 ms).
At a 160 ms hop that is ~3 chances in-domain and ~1 on real speech. Real speech
degrades both the peak and the width, because the model has no confident
direction to move in for /seɪ soʊ/.

### Proof 4 — threshold and gain are not the levers

Threshold, real data, zero false positives required:

| threshold | recall | FP |
|---|---|---|
| 0.424 (= max negative) | 31/44 (70%) | 0 |
| **0.50 (deployed)** | 28/44 (64%) | 0 |

Dropping to the zero-FP threshold buys 3 utterances out of 44. The deployed 0.5
is within 7 points of optimal. There is nothing to win here.

Gain, real data, at utterance level. Earlier window-level sweeps suggested
boosting level helps; that was an artifact of scoring near-miss windows of the
same utterance. Measured per utterance, it does the opposite:

| additional gain | max genuine | max negative | recall @ 0 FP |
|---|---|---|---|
| −12 dB | 0.728 | 0.378 | 32/44 (73%) |
| **0 dB (deployed)** | 0.744 | 0.423 | 31/44 (70%) |
| +18 dB | 0.746 | 0.480 | 26/44 (59%) |
| +30 dB | 0.841 | 0.706 | 9/44 (20%) |

Amplifying lifts positives and negatives together and saturation destroys the
margin. **Level is not the bug.** (It is still true that production audio is
quiet — median −39.0 dBFS RMS, peak amplitude 0.067 FS, versus −17.2 dBFS for
the TTS that built the positives, a 22 dB gap — and that is worth covering in
training augmentation. It is just not what is failing today.)

Two other hypotheses tested and rejected:

- **Silent lead-in.** `augment.py::align_clip_to_end` places round-0 positives
  at the window end over digital zeros, and with `rounds: 1` (the shipped
  model's setting) background mixing happens before that padding, so positives
  really did train as `silence → phrase`. Zeroing the lead-in of real positives
  does **not** raise their scores (max falls 0.646 → 0.500 across 0–1200 ms).
  Full-window `LatestWindowQueue` behaviour is not implicated either.
- **Dropped windows.** Inter-window gaps inside a cluster are ~0.11 s and
  ~0.20 s, consistent with the 160 ms hop plus write-time jitter, so the mining
  spool is not showing a window-starvation signature. The single-spike profile
  is the classifier's own response shape.

### Proof 5 — the spool is mostly positives, and is labelled as a negative corpus

`satellite/sayso/wake/mining.py` used to open with "Capture high-scoring wake
windows from production audio **as hard negatives**" (corrected), and
`WAKE_WORD_DATA.md` described the workflow as "identify false positives →
cluster failure modes". The actual spool is the opposite: **40 of 66 clusters
are isolated genuine "SaySo"** — the deliberate test session between
2026-09-13 19:48 and 20:47 UTC alone accounts for most of it — against 25
negatives and 1 the transcripts cannot settle.

Every sidecar in the spool is still `"label": null`. Anything that labels these
by score, or treats them as the negative set the docstring promises, will train
the model to reject the wake word. This is the single most dangerous
interpretation of the existing tooling and must be corrected before use.

## Improvement plan

Ordered by impact per unit effort. Items 1 and 2b contain the only remaining
decisions; 2, 3 and 4 are now unblocked work.

### 1. The phrase is fixed, and it has two realizations (decided)

The phrase is `/seɪ soʊ/` in **two acceptable realizations**: the fused one-word
brand name **"SaySo"**, and the two-word **"say so"**. Both are wakes. This
settles the spec: the bare phrase is not a negative, it is the target, so
`negative_natural_say_so` in `cases.json` cannot simply be retuned — it must be
deleted or narrowed (item 2b).

It also exposes a coverage gap that no amount of positive volume would have
fixed. `target_phrases: ["Sayso"]` supplies only the fused TTS token. The
two-word realization is a different acoustic event — two stress groups with a
possible juncture, longer — and cannot be derived from the fused token by
augmentation. The real data shows how far off this is: of 152 windows, **78 carry
a two-word "say so" transcript (63 exactly "Say so") and 0 contain the fused
string "Sayso"**. The model was trained on a rendering the user does not appear
to produce. (Caveat: Whisper's spelling preference is not proof of the speaker's
articulation — see Open questions.)

Worth noting for the retrain: the 400–480 ms plateau in Proof 3 was measured on
the **two-word** rendering, which is the one that matters here. The fused
token's plateau has not been measured at all, precisely because nothing in the
corpus is fused.

Required: list both realizations as targets.

```yaml
target_phrases:
  - "Sayso"
  - "say so"
```

Keep `config.py`'s `wake_word.phrase == "SaySo"` check as-is; it is an operator
label, not a pronunciation. Do not add an onset word (`"Hey Sayso"`) — that is a
phrase change and has been ruled out.

### 2. Stop training against the wake word

`custom_negative_phrases` in `satellite/models/sayso-training.yaml` currently
labels the target as a negative. Split it:

- **Delete — the bare phrase itself**: `"say so"`.
- **Pending item 2b — supersets of the phrase**: `"if you say so"`,
  `"just say so"`, `"you don't say so"`, `"they say so"`. Delete under
  containment, keep under isolation. Decided: keep (item 2b).
- **Delete — `"saying so"`** is `/ˈseɪɪŋ soʊ/`; the "-ing" makes it a different
  word, but it is close enough that listing it invites the same collision. Verify
  empirically before re-adding.
- **Keep — genuinely distinct near-misses**: `"says so"`, `"said so"`,
  `"so so"`, `"stay so"`, `"same so"`, `"safe so"`, `"say slow"`,
  `"say show"`, `"say sorry"`, `"say something"`, `"essay so"`, `"hey so"`,
  `"lasso"`.

That last group is no longer guesswork. The shipped model's ranking inside the
neighbourhood is measurably wrong — `"stay so"` scores **0.792** and `"say so"`
0.715, i.e. a near-miss negative outranks the target — and the real spool
contains "Stay soon" (0.549, 0.652) and "See you soon" (0.643) actually firing.
Those are the measured negatives the retrain must push down, and they become
learnable only once the positive is not defined as its own opposite.

### 2b. Supersets of the phrase — DECIDED: isolation

**The rule: if another word is in front of the wake word, it is not a wake.**
Only an isolated `/seɪ soʊ/` wakes. A word *after* the phrase does not disqualify
it, so "SaySo turn on the TV" is still a wake — that is the continuous-command
case, and it is the shape the whole product depends on.

The alternative (containment — any occurrence of the phrase wakes) was rejected.
It is easier to learn, but it wakes on TV dialogue, on quoted speech, and on the
assistant's own TTS the moment someone says "...if you say so" on the couch,
and there is no phrase-boundary signal in the pipeline to claw that back.

What the decision changes:

- `custom_negative_phrases` keeps `"if you say so"`, `"just say so"`,
  `"you don't say so"`, `"they say so"`. These are the *only* place the model
  sees the phrase preceded by speech, so they carry the rule.
- `"say so"` moves out of the negative list and into `target_phrases` (item 1).
- The `negative_natural_say_so` eval case is wrong under this rule — "natural
  speech containing say so" is exactly a wake when the phrase is first, and
  exactly a negative when a word precedes it. It is replaced by
  `negative_say_so_carrier` (item 4).
- Mining labels follow the rule too. A window that scores high and contains the
  phrase preceded by speech is a **negative**, and it is the most valuable kind:
  it is the measured form of the confusion. Item 3's labelling pass applies the
  rule, not the score.

Cost accepted: recall now depends on where the phrase starts, and the STT
handoff at `wake_skip_ms: 120` cuts before the phrase anyway, so a mis-fired
carrier leaves no transcript evidence. The classifier's 2 s window does contain
`if you` at ~0.4 s before `say so`, so the rule is learnable from the audio the
model already sees — no new runtime signal is required.

### 3. Put the real recordings into training

This is the first time real wake audio has existed. Use it.

- Label the spool before anything else:
  `python scripts/wake_mine_report.py <spool> --unreviewed --play`, then
  `--label <stem> positive|negative`. Do not label from the score: apply the
  isolation rule, which `scripts/wake_label_spool.py` does for a spool plus a
  Whisper transcript dump, and which is what
  `satellite/models/wake-spool-labels-20260914.json` records. The first pass gives
  40 isolated positives, 25 negatives and 1 undecidable across 66 clusters, and
  flags 16 for a listen — 13 because the windows of one utterance disagree about
  what was said, 3 because the phrase was heard as a near-miss ("Stay soon",
  "So so."). Correct the framing in `mining.py` and `WAKE_WORD_DATA.md` to
  "scored windows, mostly genuine wakes, unlabelled". Review with
  `python scripts/wake_mine_report.py <spool> --ledger <ledger> --review-only
  --play`, then `--label-cluster <stem> positive|negative`, which labels every
  window of that utterance at once.
- **Positives.** Seed `positive_train` with the genuine windows, keeping the two
  realizations distinguishable in the manifest.
  `scripts/wake_prepare_real_data.py training-data` does the copy: one window per
  utterance, `--repeat` for the oversampling, held-out clusters excluded, and the
  windows named `clip_9XXXXX_r0.wav` because feature extraction only reads
  `clip_\d{6}_r\d+.wav` and round-0 augmentation would truncate their tails. 40
  clips against `n_samples: 8000` synthetic is 0.5% and will be ignored — repeat
  ×20–50 so real speech is 10–20% of the positive batch. Real positives are the
  only source of the actual channel: same mic, same room, same speaker, same
  casual articulation. They are also the only place the two-word realization is
  attested — 78 of 152 windows — so they are the evidence that `target_phrases`
  needs both spellings, not a convenience.
- **Negatives.** Replace the phonetic guess list with the measured negatives:
  the ledger's 23 `negative_*` clusters (dog, TV, "good boy", "new players
  available", the carrier utterance) plus the `stt_capture` takes. These are true
  hard negatives at the true SNR, and `training-data` seeds them the same way.
- **Collect more, deliberately.** The current spool is one speaker, one room,
  and is dominated by a single test session. A recording protocol of 3–5
  speakers × 3 distances (0.5/1.5/3 m) × 3 conditions (quiet, TV on, kitchen)
  × ~20 reps gives 300–900 real positives with actual speaker diversity. Keep
  session-level train/eval splitting — never window-level.
- Add the `stt_capture` handoff misalignment as a *known* limitation: with
  `wake_skip_ms: 120` the STT audio starts after the phrase, which is why all
  51 command takes score ~0.004 and why the transcript describes what was said
  *after* the trigger. Do not mine positives from `stt_capture`.

### 4. Make the eval real, and report the right metric

- `satellite/eval/audio/` is still `.gitkeep` only, so all five cases in
  `satellite/eval/cases.json` skip silently. Populate it from the labelled
  spool; that was the documented blocker and it is now clearable. Under the
  isolation rule the old `negative_natural_say_so` case becomes
  `negative_say_so_carrier` — same category slot, narrowed to the phrase
  preceded by a word ("if you say so", "assistant tools say so"), which is a
  negative under the decided rule and a wake under the rejected one.
- **Decided: the recordings are not committed.** This repository is public and
  the corpus is household speech, so `satellite/eval/audio/` is gitignored and
  the corpus is built outside the repo by `scripts/wake_prepare_real_data.py`
  and passed with `--eval-root`. `cases.json` stays the tracked definition, so
  the cases and their expectations are still reviewable in a PR.
- Report **recall at zero false positives on held-out real audio**, not
  synthetic-only recall. The shipped model's own `sayso_eval.json` claims
  `recall 0.598 @ t=0.5, fpph 0.0` on 28.38 synthetic validation hours; the real
  measurement is 64% with 0 FP in 21 h. The synthetic number was honest and
  simply not good enough — the eval needs the real corpus so it stops hiding
  that.
- Keep `wake_word.threshold` at 0.5 until a retrained model's metrics JSON
  gives a new operating point. 0.424 is the zero-FP threshold today and buying
  3 utterances is not worth an unmeasured FP rate on 22 negative clusters.

### 5. Training-config fixes independent of the phrase decision

- **Cover both realizations.** `target_phrases: ["Sayso", "say so"]` is a data
  change, not just a label change: the two-word form has two stress groups and a
  possible juncture, so it must be generated and augmented as its own set. Do
  not assume the fused token's 400–480 ms plateau transfers.
- **Add level augmentation.** `AudioAugmentor` applies EQ (p=0.25), tanh
  distortion (p=0.25), RIR (p=0.5) and background mixing at 5–15 dB SNR, but no
  random gain. Production sits 22 dB below the synthetic positives and the mel
  frontend is `power_to_db/10 + 2` with no normalisation, so it is
  level-sensitive by construction. Sweep roughly −20…+6 dB.
- **`model_size`.** The README says the retrain config "moves small → medium";
  the YAML says `model_size: small`. Decide explicitly and state the latency
  budget — `wake_bench.py` currently holds p50 ≈ 48 ms against a 160 ms hop, so
  there is headroom.
- **`rounds: 2`.** Compound augmentation on a small corpus stacks distortion on
  distortion. Verify a round-1-only run does not train better.
- **`target_fp_per_hour: 0.02` + `max_negative_weight: 3000`.** This is what
  produced the `find_best_threshold()` fallback last time. Confirm the trainer
  actually selects a threshold this run instead of falling through to
  max-balanced-accuracy, and read the operating point off the metrics JSON.
- **Broaden positive alignment jitter** in `align_clip_to_end` (currently
  0–200 ms) once the phrase is fixed, so the activation plateau covers more
  than one hop. Measure the plateau before and after.

### 6. Do not train on this spool until it is labelled

Restating because it is the easy mistake: the filenames carry `s0000` scores and
the module is called `HardNegativeMiner`, but 67% of the content is the wake
word. Any bulk ingest that assumes "mined means negative" makes recall worse.

## Verification before shipping any retrain

1. Re-score the labelled real positives with the new model; require
   `recall @ 0 real FP` materially above the current 64% (31/44 = 70% at 0.424).
2. Require ≥3 consecutive above-threshold windows on held-out genuine utterances
   where today the maximum is 1, i.e. the plateau must actually widen.
3. Score the near-miss set ("stay so", "hey so", "says so", "said so",
   "so so", "see you soon") against the new model; every one must sit below the
   positive floor, and today the top of the set ("stay so" 0.792) sits above the
   positive ceiling (0.715). "see you soon" must stop firing — it fired on real
   audio at 0.643.
4. Both realizations must clear threshold independently: the fused "Sayso"
   **and** the two-word "say so", each tested as its own TTS set, since
   `target_phrases` now lists both and the fused token cannot stand in for the
   two-word form.
5. The isolation rule must hold on held-out audio: every carrier ("if you say
   so", "just say so", "they say so", "assistant tools say so") stays below the
   positive floor, and the same phrase with nothing in front of it still fires.
   Both halves are required — a model that rejects carriers by rejecting the
   phrase has not learned the rule.
5. Run `satellite/eval/run.py` against a populated `satellite/eval/audio/` so no
   case skips.
6. `scripts/wake_bench.py` must stay green and p50 must stay well under the
   160 ms hop.
7. Keep threshold selection on held-out real audio at session granularity.

## Open questions

- **Item 2b — containment or isolation? RESOLVED: isolation.** "If you say so"
  does not wake the assistant; any word in front of the phrase disqualifies it.
  Note the evidence that motivated the question is now the thing to distrust:
  the current model already scores "if you say so" at 0.141, but that was never
  trained deliberately, so it is coincidence, not the rule, and a retrain can
  lose it. The rule is now in `target_phrases`/`custom_negative_phrases` and in
  `negative_say_so_carrier`, so it is trained rather than assumed.
- **Do both realizations actually occur in the wild?** The user reports "SaySo"
  (1 word) or "say so" (2 words); the real corpus only attests the two-word
  rendering, at least as Whisper spells it. A short deliberate recording of both
  forms, same speaker, same session, would settle whether this is two acoustic
  classes or one plus a spelling convention — and that in turn decides whether
  `target_phrases` needs one entry or two.
- **Is `sayso.onnx` reproducible?** No training log or artifact for the shipped
  2026-09-03 model is in this repo, and its config is unrecoverable from the
  defaults plus this YAML. A retrain baseline should be re-established rather
  than compared against `sayso_eval.json`.
- **Single-speaker bias.** Every real positive is one voice. Real gains here
  may not transfer to other household members until the protocol in item 3 is
  run.
- **Whisper labelling noise.** Transcript-based labels are reliable for
  "Say so"/"Seizo"/"ZZO" but ambiguous for "Stay soon"/"See you soon"/"So",
  which may be either mis-transcribed genuine wakes or real false activations.
  Human listening review is required before these become training labels; the
  counts above should be treated as ±3 utterances.
