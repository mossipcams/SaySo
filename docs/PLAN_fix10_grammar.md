# Plan: Fix 10, grammar-constrained decoding

Parent: `docs/PLAN_review_fixes.md` Fix 10 (Batch 3, after Fix 11 and Fix 12).
Branch: `ajax/review-batch3`, cut from `e81caea`.
Scope of this plan: no code changes. Line numbers are against the current tree.

## Verdict

- The grammar must encode the LFM2 Pythonic call syntax that
  `core/lfm_parse.py` accepts, not JSON. A JSON grammar would produce text the
  parser rejects.
- The embedded and external paths emit different shapes, and only one of them
  is parsed from content. The external path currently never parses content
  (see Finding 3), so grammar-on external output would reach the user as
  speech unless llama-server converts the calls itself. This is unverified and
  is the first open check.
- Numbers, object arguments, and interior quotes are not representable in the
  parser's grammar. The grammar must mirror `lfm_parse.py` exactly; changing
  the parser is a separate decision because the training scorer uses it
  (`ARCHITECTURE.md:137-143`).
- The offline stub eval cannot measure decoding. The real measurement is a
  VM run (v11, smoke and promotion, grammar off vs on), but that needs
  `--grammar` and per-request latency, neither of which exists in the harness.
- The VM was not reachable from this session (see Open checks). Nothing here
  depends on a VM result.

## Evidence: what the model emits on each path

### Embedded (llama-cpp-python 0.3.33, default)

- Tools are not passed to llama-cpp. `_complete` calls
  `create_chat_completion(messages=...)` with no `tools` or `tool_choice`
  (`custom_components/sayso/inference.py:351-357`; asserted by
  `tests/test_inference.py:731-732`).
- The catalog is rendered into the system message as
  `List of tools: [<json>, ...]`, before the area block
  (`core/area_context.py:142-156`, applied at `inference.py:352-354`).
- The model's output is text. Calls are LFM2 native syntax:
  `<|tool_call_start|>[Name(key='value', n=1)]<|tool_call_end|>`
  (`core/completion.py:134-135`, `tests/test_inference.py:45`).
- Parsing is the integration's own: `extract_tool_calls`
  (`core/completion.py:147-178`) then `parse_lfm_python_tool_calls`
  (`core/lfm_parse.py:141-176`). `parse_completion_result` runs it when
  `message.tool_calls` is empty (`core/completion.py:187-193`).
- ARCHITECTURE states this explicitly: `libllama` in llama-cpp-python ships
  without `common/chat.cpp`, so llama.cpp's tool-call parser is not available
  in-process (`ARCHITECTURE.md:137-143`).

### External, production (`ExternalEngine` to llama-server)

- `ExternalEngine.async_chat_completion` sends the messages as-is and the
  catalog as the `tools` field (`inference.py:264-278`). It does not call
  `tools_before_area_messages`, so the catalog is not in the system prompt on
  this path. This matches `ARCHITECTURE.md:105`.
- `client.py` sends `tools` (`client.py:64-65`, payload at `:50-66`).
- `LlamaCppClient.chat_completion` parses only `message.tool_calls`
  (`client.py:186-202`). It never calls `extract_tool_calls`. A content-only
  response with a `<|tool_call_start|>...` string comes back as
  `ChatCompletionResult(content=<raw text>, tool_calls=[])`.
- `ExternalEngine` returns that result directly (`inference.py:272-278`), so
  post-model treats the raw text as speech (`core/post_model.py:208-212`).
- Therefore the external path works today only if llama-server's own
  `--jinja` parser converts the LFM2 text into `tool_calls`. That is not
  verified (open check 2).

### Eval endpoint (`evals/adapters/endpoint.py`)

- `render_case` always produces an area block (`evals/runner.py:198-216`,
  through `render_system_prompt`), so `tools_before_area_messages` injects the
  catalog (`evals/runner.py:407-410`).
- `sent_tools` is `[]` whenever injection happens (`evals/runner.py:410`).
  `EndpointAdapter` sends no `tools` field (`endpoint.py:39-40`).
- So the eval exercises the content-text path (same shape as embedded), and
  it does not exercise the external production layout (`tools` field, no
  injection). Any grammar measurement from the eval covers the embedded shape
  only. It says nothing about external production.
- Eval parsing uses `completion.parse_completion_result`
  (`evals/runner.py:42`, `:428`), which includes the content extraction that
  `ExternalEngine` skips.

### Parser contract the grammar must match

From `core/lfm_parse.py` (probed against the real module in this session):

- Values: single- or double-quoted strings (`:15-50`), integers only
  (`_parse_value` consumes `-0123456789` and calls `int`, `:72-76`), and
  `True`/`False`/`None` (`:59`, `:77-79`), and lists (`:83-101`). No floats,
  no dicts.
- Arguments: keyword-only `key=value`, key `[A-Za-z_][A-Za-z0-9_]*`
  (`:104-124`, key regex at `:110`), any order, repeats allowed, `Name()`
  allowed.
- A string closes at a quote followed by `,`, `)`, `]`, `}`, or end
  (`_CLOSER_AFTER_QUOTE`, `:12`, `:27`, `:46`). An interior quote followed by
  anything else is literal.
- Calls: one bracketed list `[a(...), b(...)]` (`:143-144`), or a bare call
  (`core/completion.py:144-161`). Multiple bracketed groups are rejected (probe,
  below).
- Tool name: `[A-Za-z][A-Za-z0-9_]*` (`lfm_parse.py:153`). The envelope allows
  `[a-zA-Z0-9_-]{1,64}` (`core/tool_schema.py:12`), so hyphenated names exist in
  the envelope but cannot be expressed in the call syntax.

Probe results from this session (read-only, `extract_tool_calls` on the
current tree):

| input | result |
|---|---|
| `<|tool_call_start|>[HassTurnOn(name='lamp')]<|tool_call_end|>` | parsed |
| `[HassTurnOn(name='lamp')]` (no markers) | parsed |
| `[HassTurnOn(name='lamp'), HassTurnOff(name='fan')]` | parsed (2 calls) |
| `[HassTurnOn(name='lamp')] [HassTurnOff(name='fan')]` | **rejected**: "expected ',' between tool calls" |
| `[SetTemp(level=21)]` | parsed |
| `[SetTemp(level=21.5)]` | rejected (no floats) |
| `[Set(on=True)]` | parsed |
| `[HassTurnOn({"name": "lamp"})]` | rejected (no dicts) |
| `[SetTemp(level=-)]` | **raises raw `ValueError`** (`int('-')`), not `SaySoError` |

The `ValueError` case matters. `parse_completion_result` runs outside the
`try` in `EmbeddedEngine.async_chat_completion` (`inference.py:399`).
`_Turn.model_failure` re-raises anything that is not a `SaySoError`
(`conversation.py:194-195`). So a bare `-` can crash the turn.

### Catalog shape (checked-in fixtures)

- `schemas/ha-2026.11.1-tools.json`: 40 tools, all names are identifiers, all
  argument keys are identifiers, top-level `object` only. Types: `string`
  133, `integer` 34, `array` 29, `boolean` 2, `number` 1.
- `schemas/sayso-tool-schema-v2.json`: 27 tools, same pattern.
- The only `number` is `HassClimateSetTemperature.temperature`
  (`ha-2026.11.1-tools.json`, `climate__HassClimateSetTemperature/.../temperature`).
  The parser cannot express floats, so this argument is unrepresentable. Under
  the current parser a float already fails (probe above), so the grammar
  costs nothing it had.
- Sanitizer fallback: unknown/unsupported nodes become `{"type": "string"}`
  (`schema.py:43`, `:158-171`). Live HA catalogs may carry `anyOf` or
  `nullable`; those are not in the fixtures and must fail open (see Design).

## Corrections to the Fix 10 text

1. "fingerprint already exists (`schema.py:373-376`)". The file is 354 lines.
   The fingerprint is defined at `core/tool_schema.py:97-99` and used at
   `schema.py:353`.
2. "`root ::= call | text`, typed per tool". The review does not say what
   `call` is. It must be the LFM2 Pythonic form, not JSON (see Evidence). A
   JSON-shaped grammar would make every call the parser rejects. The
   grammar must be generated from the parser contract, with a test that
   checks agreement with `parse_lfm_python_tool_calls`.
3. "Thread an optional grammar ... and the `grammar` field for llama-server".
   This assumes the external response is parsed from content. It is not
   (`client.py:186-202`). Either llama-server converts the calls itself (open
   check 2), or `ExternalEngine` must run content extraction. That second
   option is a bug fix independent of grammar (see Slice 0) and needs a user
   decision.
4. "Adopt it only if the Fix 13 evals show ... acceptable p50 latency". The
   harness records no latency. `grep` for latency, elapsed, and timing in
   `evals/*.py` finds nothing; `runner.py` uses `time.monotonic` only for the
   held-call TTL (`runner.py:476`). The adoption rule cannot be checked until
   Slice 5 lands.
5. "Fix 13 evals" as the decoding measurement. Those evals send no `tools`
   field (see Evidence), and the stub adapter renders calls itself and never
   decodes. Only the endpoint adapter against a live server measures decoding,
   and only on the embedded-shaped prompt.

## Open checks (not verified here)

1. **Special-token matching.** Does llama.cpp's grammar sampler match the
   literal text `<|tool_call_start|>` when that token is a control token? The
   grammar must include the marker if the model emits it. Check: in a
   throwaway env with `llama-cpp-python==0.3.33`, build
   `LlamaGrammar.from_string('root ::= "<|tool_call_start|>" "x"')` and generate
   with a model that emits the marker; or inspect llama.cpp
   `src/llama-grammar.cpp` at the vendored commit.
2. **llama-server with `tools` plus `grammar`.** Does the server honor a
   request `grammar` when `tools` is also present, and does it parse LFM2
   pythonic output into `tool_calls`? Check (one short request, read-only, on
   the VM): run `curl -s http://<vm>:8080/v1/chat/completions` with a `tools`
   array and a trivial `grammar` (for example `root ::= "A" "B"`, `max_tokens`
   8). If `content` is `AB`, the grammar is honored. Also check `/props` for
   `build_info`, the llama.cpp build.
3. **Real output shape.** Does v11 emit markers, bracketed lists, multi-call
   groups, and argument order as in `evals/cases`? Check: one raw completion
   per smoke case via `EndpointAdapter` on the VM, saving the `raw` field.
4. **Dead-end and truncation.** What does llama.cpp do when the grammar has no
   legal continuation, and when `max_tokens` (160, `const.py:28`) stops a call
   midway? Check: a unit case with a forced dead-end grammar, and an
   `EndpointAdapter` run with `max_tokens=8`.
5. **Live HA catalogs.** Do real HA 2026.11 catalogs contain `anyOf`,
   `nullable`, or nested objects? Check: dump the `_compile_catalog` output
   from a live HA instance (read-only) and grep for them. Fixtures say no.

VM status from this session (read-only): `ssh llm` does not resolve
(`Could not resolve hostname llm`). `ssh matt@192.168.1.76` was refused with
`Permission denied (publickey,password,keyboard-interactive)`. Nothing was run
on the VM. The user must run the status check:

    ssh <llm-host> 'docker ps --format "{{.Names}} {{.Status}}"; curl -s -m 5 http://127.0.0.1:8080/health; echo; curl -s -m 5 http://127.0.0.1:8080/props | python3 -m json.tool | head -60'

## Design

### `core/grammar.py` (HA-free)

- Input: the compiled tools exactly as sent to the model. That is
  `_tools_of(turn.active_schema)` (`conversation.py:473-475`), not
  `complete_schema`. A grammar that allows a tool outside the prompt would
  expand what the model may call, which the boundary rules forbid
  (`ARCHITECTURE.md:143-160`).
- Output: a GBNF string, or a typed failure (`GrammarUnsupported` with a
  reason code).
- Cache: keyed by `schema_fingerprint` (`core/tool_schema.py:97-99`), following
  the `lru_cache` pattern at `schema.py:233-234`. The cache key is the canonical
  JSON string, which is what the fingerprint hashes, so it is hashable.
- Imports: `json`, `re`, and `core.tool_schema` only. No `homeassistant`
  (`pyproject.toml:68-72` contract).

### Grammar shape (sketch, not final)

    root        ::= call-block | text
    call-block  ::= "<|tool_call_start|>" ws "[" ws call (ws "," ws call)* ws "]" ws "<|tool_call_end|>"
    call        ::= call-<tool> | ...          # one alternative per tool
    call-<tool> ::= "<tool>(" ws (kv (ws "," ws kv)*)? ws ")"
    kv          ::= kv-<tool>-<arg> | ...      # arg keys from that tool's properties
    kv-<tool>-<arg> ::= "<arg>=" <value per type>
    text        ::= <any text that does not begin with the call start marker>

- The marker form is included only if open check 3 shows the model emits it.
  Otherwise use the bare-bracket form. The parser accepts both
  (`completion.py:152-161`), so either is parse-compatible.
- Argument keys: any order, repeats allowed, no required-check in the grammar.
  Required and schema checks stay in the existing validation barrier
  (`tool_contract.py`, `post_model.py:234-256`). The grammar only shapes syntax.
  A fixed key order would force an order the parser does not require and the
  model may not have been trained on.
- Values, per schema type:
  - `string`: `'` body `'` where body is `([^'] | "'" [^,)\]}'])*`. The
    interior quote is allowed only when the next char is not a closer and not
    a quote. This is the exact language the parser accepts, and it avoids the
    `''` pair ambiguity. `Mom's` is representable; `Kids'` is not. Double
    quotes use the same rule with `"`. Enum string values with a quote are
    unrepresentable.
  - `integer`: `"-"? [0-9]+`. Requires at least one digit, so bare `-` is
    impossible (the `ValueError` above).
  - `number`: integer only, because the parser has no floats. This makes the
    `temperature` argument integer-only under grammar. Floats stay
    unrepresentable until a parser change is decided separately.
  - `boolean`: `"True" | "False"`.
  - `array`: `"[" ws (item (ws "," ws item)*)? ws "]"`, with item typed by
    `items`.
  - `enum`: alternatives of the typed literals.
- Fail-open rules (grammar not built, request sent without grammar, reason
  recorded): an unsupported `type` or missing `type` (including `anyOf`,
  `nullable`, `null`, nested `object`), a tool name not matching
  `[A-Za-z][A-Za-z0-9_]*`, an argument key not matching
  `[A-Za-z_][A-Za-z0-9_]*`, an enum value containing a quote, or any
  `GrammarUnsupported`. Fail open is chosen over excluding the tool, because
  excluding a tool would make it uncallable under grammar, a regression.

### Threading

- `SaySoInferenceEngine.async_chat_completion` gets a keyword-only
  `grammar: str | None = None` (`inference.py:237-245`). Same in
  `ExternalEngine` (`:264-278`) and `EmbeddedEngine` (`:368-399`).
- External: `LlamaCppClient.chat_completion` gains `grammar` and writes
  `payload["grammar"]` only when it is not `None` (same pattern as `tools` at
  `client.py:64-65`).
- Embedded: `_complete` (`inference.py:343-366`) passes
  `grammar=LlamaGrammar.from_string(grammar)` only when set. The conversion is
  behind a seam (for example `_make_grammar`) so tests can monkeypatch it, the
  way `_load` is patched (`tests/test_inference.py:713-730`). `llama_cpp` is not
  installed in `.venv`, so tests cannot import it.
- The existing assertion that `tools` and `tool_choice` are absent from the
  embedded kwargs (`tests/test_inference.py:731-732`) keeps holding, because
  `grammar` is added only when on.
- `prefix_hash` (`core/completion.py:122-131`) must not include the grammar.
  The grammar does not change prompt tokens, so prefix checkpoints
  (`inference.py:83-213`) are unaffected.
- Grammar is built in `_Turn.complete` (`conversation.py:209-241`) from the
  same `tools` it passes to the engine, and only when tools are present and the
  option is on.

### Option

- New `CONF_GRAMMAR_DECODING = "grammar_decoding"` and
  `DEFAULT_GRAMMAR_DECODING = False` in `const.py` (next to the other `CONF_`
  keys, `:84-107`), added to `OPTION_KEYS` (`const.py:111-126`).
  `OPTION_KEYS` is defined but not referenced anywhere else in the tree
  (`grep` found only the definition), so it is documentation for now.
- `config_flow.py` `_OPTIONS` (`:109-140`): a `BooleanSelector` row, same
  pattern as `CONF_TRACE_STORE_UTTERANCES` (`:134-138`). Defaults come from
  `_OPTION_DEFAULTS` (`:142`).
- Labels in `strings.json` and `translations/en.json` (options step data at
  `:61-62`, descriptions at `:66`).
- Read in `__init__.py` (`:286` and `:301` are the pattern) and carried on
  `SaySoRuntimeData` (`__init__.py:96-108`) as a bool.
- Default off. Adoption, if measurement supports it, is a separate change that
  flips the default. Not in this batch.
- Batch 4 (Fix 8, subentries) also touches options. Coordinate before merging.

### Failure behavior

- **Construction fails** (`GrammarUnsupported` or any exception in
  `core/grammar.py`). Send without grammar, set span metadata
  `grammar: "skipped"` and `grammar_skip_reason: <code>` on the INFERENCE span
  (`conversation.py:217-239`). Never fail the turn for this.
- **Grammar on, model stops mid-call** (max tokens, or a dead end). The
  content is an incomplete call. `extract_tool_calls` raises
  `SaySoInvalidToolCallsError` (`completion.py:163-168`; pinned by
  `tests/test_inference.py:110`), which `model_failure` maps to
  `ERROR_NOT_UNDERSTOOD` (`conversation.py:197-200`). This is the same
  fail-closed result as today. No automatic retry is added; correction is for
  validation failures only (`post_model.py:234-256`).
- **Record `finish_reason`.** It is never read (`grep` finds no `finish_reason`
  in `custom_components`, `evals`, or `satellite`). Add it to the result
  (`ChatCompletionResult`, `completion.py:24-33`) and to the INFERENCE span,
  so a `length` stop is visible in traces.
- **External content with markers and no `tool_calls`** (open check 2 fails).
  Either `ExternalEngine` runs the same content extraction as embedded (Slice
  0), or the external grammar path is disabled. Do not ship grammar on the
  external path until one of these is true.

## Evals

### Offline stub (`evals/results/fix13_stub_eval.py`)

- It renders calls as `[name(k=v)]` text and returns them as content
  (`fix13_stub_eval.py:16-35`). It never decodes and never sees a grammar. It
  cannot measure grammar effects.
- It can test the grammar's compatibility with scoring. Sample strings from
  the grammar (via a regex translation of the same language, or a bounded
  generator), parse them with `parse_completion_result`, and check they
  validate. This is a unit test (Slice 1), not an eval result.
- Known stub artifact: its multi-call rendering (`[a()] [b()]`, from
  `render_calls`) is rejected by the parser (probe above). The stub's
  `multi_action` category is 0.0 in `fix13_before_smoke.json`
  (`category_pass_rate`), which is consistent with this. The stub's outcome
  counts, including `malformed_output`, reflect the stub, not v11. Do not use
  them to bound the grammar gain.

### Real measurement (requires user approval and the VM)

Adapter changes (Slice 5): `EndpointAdapter` gets an optional grammar
per request and records wall time per request, and `evals/cli.py` gets a
`--grammar` flag. The grammar is built per case from `rendered["tools"]`,
the same list used for injection (`runner.py:407-410`). `ModelAdapter.complete`
gains a `grammar` argument, so `InMemoryAdapter` and the stub need the new
argument too.

Commands (run from the Mac against the VM server; the server URL is a
placeholder, the integration default is `127.0.0.1:8080`, `const.py:26`):

    export PATH=/Users/matt/Projects/SaySo__worktrees/ajax-codebase/.venv/bin:$PATH
    S=http://<vm-host>:8080
    python -m evals.cli run --suite smoke     --adapter endpoint --server $S --model sayso
    python -m evals.cli run --suite smoke     --adapter endpoint --server $S --model sayso --grammar
    python -m evals.cli run --suite promotion --adapter endpoint --server $S --model sayso
    python -m evals.cli run --suite promotion --adapter endpoint --server $S --model sayso --grammar

Also run `--suite promotion` (off) twice, to measure run-to-run noise. Results
land in `evals/results/<run_id>/` (`runner.py:46`, `write_run`). `PRODUCTION_TEMPERATURE`
is 0 (`evals/outcomes.py:10`), so any off-vs-on difference comes from grammar
rejecting tokens the model would otherwise pick. Run-to-run noise from the
server bounds the reading.

Compare, per suite, off vs on:

- `passed` and `total`, `overall_pass_rate`, `action_execution_accuracy`,
  `false_action_rate`, `malformed_output_rate`, `invalid_argument_rate`,
  `category_pass_rate`, and any `flags` (zero-tolerance per `gates.yaml`).
- `first_pass` (raw first completion) as well as final outcome.
- Latency: p50 and p95 of per-request wall time, and per-turn totals.

### Adoption rule

Adopt only if all hold on promotion and smoke, relative to grammar off on the
same server and build:

- Accuracy: `passed` on is at least `passed` on off; `action_execution_accuracy`
  not lower.
- Safety: `false_action_rate` not higher; no zero-tolerance flag appears that
  was absent when off (`gates.yaml` `zero_tolerance_flags`).
- Format: `malformed_output_rate` not higher.
- Latency (proposal, to confirm with the user): p50 per request at most 1.10x
  off, and p95 at most 1.25x off. Set an absolute p50 ceiling after the off
  baseline is measured.
- Noise: an accuracy difference counts only if it exceeds the off-vs-off
  run-to-run spread.

Adoption flips nothing by itself. The default stays off, and flipping it is a
separate change.

Embedded adoption needs its own check. The VM runs llama-server's build; the
embedded path runs llama-cpp-python 0.3.33's `libllama` (`const.py:42`). The
two builds can differ, so VM results do not transfer. There is no embedded
measurement harness today. Ask the user before building one.

## Parent decisions (binding for the slices)

1. **Slice 0 ships.** `ExternalEngine` runs the same content extraction as
   embedded when `message.tool_calls` is empty. Tests go in the new
   `tests/test_grammar_external.py`; `tests/test_inference.py` is not edited.
2. **The bare-minus crash is fixed in this batch.** `lfm_parse._parse_value`
   raises `LfmPythonParseError` instead of leaking `ValueError`. The accepted
   language is unchanged, so training-scorer parity holds. Ship it with Slice 0,
   with its own test file.
3. **Marker form is a builder parameter.** Open check 1 needs a real model, so
   `build_grammar(..., markers: bool)` emits either the
   `<|tool_call_start|>`-wrapped block or the bare-bracket block. Both
   round-trip through the parser. The eval `--grammar` flag picks one
   (`--grammar markers|bare`). The production option uses `markers` until the
   measurement says otherwise.
4. **`text` must not swallow calls.** If the text branch accepts any string, the
   grammar constrains nothing. `text` must be non-empty and its first character
   must not be `[` or `<`. A test asserts that a malformed call such as
   `[HassTurnOn(name=)]` is outside the whole grammar language, not just outside
   the call branch.
5. **Slice 7 (VM measurement) runs only after the user approves.** The llm VM
   GPU and server are shared with production HA. Slices 0-6 ship with the
   option off by default.
6. Latency thresholds (p50 at most 1.10x, p95 at most 1.25x) stay a proposal
   until the adoption change.

## Slices

Each slice is at most four files. Each has a verify command set. Run every
python command with `.venv/bin` first on `PATH`.

Global verification (run at the end of each slice that touches code):

    ruff check . && ruff format --check . && pytest tests evals/tests --ignore=tests/test_realistic_v3.py -q && ~/.cache/sayso-venvs/satellite/bin/python -m pytest -q satellite/sayso && lint-imports && python -m evals.generate_realistic --check

The satellite command is the AGENTS.md gate (`pytest -q satellite/sayso`), run in
the satellite venv. Confirm the exact CI invocation before the first code slice.

- **Slice 0 (decision required, may ship first).** Fix `ExternalEngine` so a
  content-only response with calls is parsed the same way as embedded. Files:
  `custom_components/sayso/inference.py`, `tests/test_inference.py` (edit,
  needs explicit approval per the global test rule), `tests/test_grammar_external.py`
  (new). Verify: `pytest tests/test_inference.py tests/test_grammar_external.py -q`.
  Without a decision, skip Slice 0 and keep grammar off on the external path.
- **Slice 1: builder.** Files: `custom_components/sayso/core/grammar.py` (new),
  `tests/test_grammar.py` (new). Contents: builder, representability rules,
  fail-open reasons, fingerprint cache. Tests: grammar output for fixture
  catalogs; round-trip (strings sampled from the value languages parse to the
  same values through `parse_lfm_python_tool_calls`); unsupported inputs raise
  `GrammarUnsupported` with the right reason; bare `-` is not in the language.
  Verify: `pytest tests/test_grammar.py -q`; `lint-imports` (core has no HA import).
- **Slice 2: threading.** Files: `custom_components/sayso/inference.py`,
  `custom_components/sayso/client.py`, `tests/test_grammar_threading.py` (new).
  Tests: external payload carries `grammar` only when set; embedded passes a
  grammar only when set, via the seam; existing assertions unchanged. Verify:
  `pytest tests/test_inference.py tests/test_client.py tests/test_grammar_threading.py -q`.
- **Slice 3: option.** Files: `custom_components/sayso/const.py`,
  `custom_components/sayso/config_flow.py`, `custom_components/sayso/strings.json`,
  `custom_components/sayso/translations/en.json`. Verify: `pytest tests -q -k config`
  (the options-flow tests), plus ruff.
- **Slice 4: wiring and trace.** Files: `custom_components/sayso/conversation.py`
  (build grammar in `_Turn.complete`, span metadata `grammar`,
  `grammar_skip_reason`, `finish_reason`), `custom_components/sayso/__init__.py`
  (read option, runtime field), `custom_components/sayso/core/completion.py`
  (`finish_reason` on `ChatCompletionResult`), `tests/test_grammar_wiring.py`
  (new). Verify: `pytest tests/test_grammar_wiring.py tests/test_tracing.py tests/test_inference.py -q`.
  Field names are provisional: Fix 17 renames trace fields
  (`PLAN_review_fixes.md`, Fix 17), so the span keys here must be coordinated
  with that batch.
- **Slice 5: eval adapter.** Files: `evals/adapters/endpoint.py`,
  `evals/runner.py` (grammar per case, `ModelAdapter.complete` argument),
  `evals/cli.py` (`--grammar`), `evals/tests/test_endpoint_grammar.py` (new).
  Verify: `pytest evals/tests -q`; `python -m evals.cli run --help` shows
  `--grammar`. Do not run a suite without approval.
- **Slice 6: latency.** Files: `evals/scorer.py` (p50/p95 in `summarize`),
  `evals/runner.py` (per-request timing on `CaseResult`), `evals/tests/test_scorer_latency.py`
  (new). Verify: `pytest evals/tests -q`. Existing scorer assertions unchanged.
- **Slice 7: measurement.** No code. Run the VM commands (Evals section) after
  the user approves, and write the comparison to a new file in `docs/` or
  `evals/results/`. Verify by reading the two summary JSONs.

## Risks

1. **Parser and grammar drift.** Mitigated by round-trip tests in Slice 1. The
   grammar must change with `lfm_parse.py`. Add a test that fails when they
   disagree on the sampled languages.
2. **External path returns speech.** Grammar on external may return raw
   markers as the reply (Evidence, external). Blocked by Slice 0 or by open
   check 2.
3. **Eval layout is not production.** The endpoint eval injects the catalog
   and sends no `tools`. External production sends `tools` and injects
   nothing. A grammar result on the eval says nothing about external
   production. State this in any result report.
4. **Number argument loss.** `HassClimateSetTemperature.temperature` is
   integer-only under grammar. Floats already fail in the parser, so no
   regression, but it is a capability gap against HA. Fixing it means changing
   `lfm_parse.py`, which changes training-scorer parity (`ARCHITECTURE.md:137-143`).
5. **Special-token matching.** If llama.cpp cannot match control-token text,
   the marker form is impossible, and the bare-bracket form must be used
   (open check 1).
6. **Truncation at 160 tokens.** A grammar-constrained multi-call can run past
   `DEFAULT_MAX_OUTPUT_TOKENS=160` (`const.py:28`, `evals/outcomes.py:11`). The
   effect is a fail-closed parse error. Measure in the VM run, and count
   `finish_reason == "length"`.
7. **Stub and promotion outcomes are not model outcomes.** The Fix 13 stub
   numbers (for example `malformed_output: 46` in `fix13_after_promotion.json`)
   come from the stub. Do not cite them as v11 failure rates.
8. **Dead-end behavior.** Unknown in llama.cpp and llama-cpp-python 0.3.33
   (open check 4). A dead end must not hang the turn. The embedded path has a
   timeout (`inference.py:381-393`); the external path has the aiohttp timeout
   (`client.py:141`).
9. **Greedy decoding bounds the effect.** At temperature 0 (`outcomes.py:10`),
   a grammar changes output only where the unconstrained argmax is
   ungrammatical. Expect small deltas. A null result is possible and is a valid
   outcome.
10. **Bare-minus crash exists without grammar.** The `ValueError` leak
    (`lfm_parse.py:72-76`, `inference.py:399`, `conversation.py:194-195`)
    affects the unconstrained path today. Grammar blocks `-` alone, but the
    parser bug stays. Fix it in its own change (wrap `int()` in the parser to
    raise `LfmPythonParseError`), and note it parser-wide.
11. **Test rule.** The global rule forbids editing existing `tests/` files
    without an explicit ask. New files in `tests/` are allowed by this task.
    Slice 0 edits `tests/test_inference.py` and needs explicit approval.
12. **Option and Batch 4 overlap.** Batch 4 (Fix 8) changes options and config
    entries. Coordinate the option names and `OPTION_KEYS`.
13. **Prefix cache.** Must not change. The grammar is excluded from
    `prefix_hash`. Verify with the existing prefix tests
    (`tests/test_inference.py:750-777`, the `TestPrefixHash` class).
14. **Embedded versus VM.** Different libllama builds. A VM win does not
    justify embedded adoption.

## Verification of this plan

- `test -s docs/PLAN_fix10_grammar.md` (non-empty).
- `git branch --show-current` returns `ajax/review-batch3`.
- No code files changed, no commits, no push.
