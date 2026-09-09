# English intent grammar source

`ohf_en.json` is adapted from [OHF-Voice/intents](https://github.com/OHF-Voice/intents),
by the Open Home Foundation and its contributors, at revision
`c805585d4604c2819ce618f034758cda3bf92fbb`.
The upstream material is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Attribution, source paths, source-file SHA256 hashes and the revision are retained
in the JSON. Upstream tests are not included as training examples.

The extraction selects 21 English intent families and shared English expansion
rules, preserving sentence definitions and their slot/domain restrictions.
To reproduce the JSON, check out that exact upstream revision and run:

```sh
python training/generators/ohf_extract.py /path/to/intents
```

SaySo uses Hassil 3.12.0 to parse the sentence syntax. It samples a bounded path
through the grammar, binds names/aliases as literal values, and accepts a result
only when all supplied semantic slots are present. It filters shorthand that
cannot be framed as a complete request, normalizes duration agreement, and makes
the light domain explicit in scoped brightness requests. These are adaptations
for generation; this is not a replacement for Home Assistant intent recognition.

Calls without a compatible grammar retain SaySo's explicit renderer. In
particular, GetLiveContext state questions, per-script tools, vacuum calls and
combined settings can use the fallback. The imported inventory does not expand
the offered Home Assistant tool contract. Positive v3 rows record the selected
upstream source/block/template/revision or `sayso_fallback` in `metadata.linguistics`.
Style and STT transforms follow rendering; that metadata identifies the base
grammar, not a claim that noisy final text matches the original sentence exactly.
