# Home-LLM English piles (vendored)

CSV/text piles from [Home-LLM](https://github.com/acon96/home-llm) @ `50cf35c9`
(`training/upstream.lock.json`). MIT-licensed; see `LICENSE`.

Used offline by `training/generators/` for SaySo v1 SFT example diversity.
Only rows mappable to `schemas/sayso-tool-schema-v1.json` are emitted; the rest
are dropped with counted reasons.
