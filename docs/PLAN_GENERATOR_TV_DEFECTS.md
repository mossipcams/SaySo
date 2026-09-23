# Generator: #94 bare names and #102 letter-spaced TV

Scope: make the v4 generator emit rows that teach the two TV `MatchFailedError`s. Full fine-tune still starts from `LFM2.5-230M-Base` later; this change is data only. No training launch, no eval/runtime/HA edits, no 40k generate, no shrinking family shares.

## Defects

**#94** — utterance “the living room TV”, registry name `TV`. Gold must be `name="TV"`, never `Living room TV`. `bare_name_rate` already names a share of fixtures without the area prefix. Confirm utterances can mention the room while gold stays the canonical name, and that validation does not reject those rows.

**#102** — STT writes `T V` / `T.V.` / `teevee`. Gold stays `name="TV"`. `stt_noise.py` already splits `outlet` → `out let` and has no TV rule. Gold is assigned before STT, so the label is right, but `validate_row` requires the literal target string in the utterance (`missing_expected_target`) and will drop `T V` rows.

Do not put recoverable TV commands in `junk`. Do not add alias `T V` on the fixture (that would teach emitting a name HA does not hold).

#39 / #97 floors and the junk family already exist in this worktree; leave them unless a TV row path is currently mis-labeled.

## Files

- `training/generators/stt_noise.py` — meaning-preserving TV/acronym splits, same pool as `out let`.
- `training/generators/validation.py` — `missing_expected_target` treats letter-spaced / punctuated acronyms as the canonical name.
- `training/generators/gold.py` / `utterances.py` / `homes.py` — only if investigation shows #94 still concatenates labels or cannot mention area with a bare name.
- `training/tests/test_stt_noise.py`, `training/tests/test_defect_v4_corpus.py` — failing checks first: STT `TV`→`T V` keeps gold `name="TV"`; validator accepts it; bare `TV` in Living Room plus “living room TV” utterance still labels `name="TV"`.

Out of scope: `evals/`, recipes’ allocation shares, `full_sft_v4` 40k build, training scripts, `real_home.json` aliases.

## Verification

- `python -m pytest training/tests/test_stt_noise.py training/tests/test_defect_v4_corpus.py training/tests/test_canonical_generator.py -q`
- `python -m pytest training/tests -q` stays green (pre-existing failures already noted in this worktree are not this task unless the new code causes them).
- No eval case IDs/utterances copied into training labels.
