# 10% log-realistic STT corruptions

Scope: ~10% of training rows get utterance noise copied from live faster-whisper
on the SaySo pipeline, not invented generic typos. Gold stays the canonical
entity/tool. No eval edits, no training launch, no 40k generate.

## Evidence

SaySo pipeline `stt.faster_whisper` debug runs on 192.168.1.35:

- `Turn off librarian TV.` ← spoken living-room TV (same class as
  `living room light` → `ribbon room light`)
- `Yeah.` ← non-command (junk family, not this 10%)

## Files

- `training/generators/stt_noise.py` — log pool, `fold_stt_text` so
  librarian/ribbon still refer to living; `apply_log_stt_noise`
- `training/generators/config.py` — `stt_log_rate`
- `training/generators/pipeline.py` / `row_generation.py` / `stats.py` — fill
  the 10% after phrasing so dropped articles are not re-inserted
- `training/configs/generation/full_sft_v4.yaml` — `stt_log_rate: 0.10`
- `training/tests/test_stt_noise.py`
- `docs/PLAN_AGGRESSIVE_SFT_CORPUS.md` — 10% log slice

## Transforms (meaning-preserving; gold unchanged)

- Phrase: `living room` → `ribbon room` / `librarian`
- Drop fillers/articles: the, a, uh, um, please, just
- Homophones / minor name: lite, van, lok, blends, T V
- Punct/casing: trailing `.` as in the live log; mixed case

## Verification

```bash
cd training && .venv/bin/python -m pytest tests/test_stt_noise.py -q
```
