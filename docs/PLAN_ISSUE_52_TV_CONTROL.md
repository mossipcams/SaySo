# PLAN: Issue #52 — TV control produces no tool call

Status: measured, then implemented in this branch.
Author: agent, 2026-09-14.
Scope: `custom_components/sayso/schema.py`, `custom_components/sayso/manifest.json`,
`training/generators/`, `training/evals/grounding_eval.py`, docs.

## 1. What the evidence actually says

Issue #52 attributes the failure to Run 008 and offers three root-cause
candidates. The first is wrong, the second is real but secondary, and the
actual dominant cause was not listed.

### 1.1 The deployed model is not Run 008

Verified on the VM rather than inferred from a display name:

```
/etc/default/llama-server  MODEL=/srv/models/SaySo-LFM2.5-230M-40k-gauntlet-step2500-Q8_0.gguf
                           ALIAS=v3-40k   EXTRA_ARGS=--jinja   N_GPU_LAYERS=0
sha256  229c805d85e7ef807bf895d91bf653079ec1eff7ee8d18618baefd6cb4e536f1
/props  model_path = the same file, ftype Q8_0, n_params 229693184, n_ctx 16384
```

That artifact is **Run 013 step 2500**, promoted 2026-09-13 13:52 — about ten
hours before issue #52 was filed (2026-09-13T23:44:07Z). Three separate labels
all still say something else:

| Label | Claims | Reality |
|---|---|---|
| `--alias v3-40k` | v3-40k | Run 013 step-2500 |
| HA entity friendly name | `/srv/models/LFM2.5-230M-Q8_0.gguf` | same |
| `/srv/models/CHAMPION.txt` | `v3-semantic-early-20260908` epoch 2 | same — champion by eval, but **not** the served model since 2026-09-13 |

So "Run 008, 9/38 recipe-lock" describes a model that was not serving the
traffic in the issue. **Candidate 1 is not established.**

### 1.2 The dominant cause: every tool reaches the model with no parameters

Captured at the llama.cpp boundary (temporary `-v --log-file`, since reverted).
The rendered prompt contains all 23 tool definitions. Every one of them:

```json
{"type":"function","function":{"name":"intent__HassTurnOn",
 "description":"...","parameters":{"properties":{},"type":"object"}}}
```

and the derived grammar therefore admits no arguments at all:

```
tool-intent-HassTurnOn ::= ("intent__HassTurnOn" "(") space space ")"
```

`HassTurnOn(name="TV", domain=["media_player"])` is **not expressible** in the
contract the model is given. A perfect model could not pass this. Offered a
tool set in which nothing can carry a target, the model declines and speaks
prose — which is exactly the reported behaviour.

Root cause, reproduced offline against `homeassistant==2026.9.2`:

- HA 2026.9 replaced voluptuous with **`probatio`**, installed under the
  `voluptuous` name via `install_as_voluptuous()`. `vol.Schema` is
  `probatio.schema.Schema`.
- SaySo pins `voluptuous-openapi==0.4.1` in its manifest. That version's
  `convert()` does not understand probatio schemas and returns
  `probatio.codecs._shared._Unsupported`.
- `schema._is_unsupported_node()` duck-types on `type(node).__name__ ==
  "_Unsupported"`, so it matches, and `_sanitize_convert_output()` returns
  `_PARAMETERS_ROOT_FALLBACK = {"type": "object", "properties": {}}`.
- That fallback fires for **every** tool, silently.

HA 2026.9's own integrations use `from probatio import to_openapi`. Against the
same schema it returns the real properties (`name`, `area`, `floor`, `domain`,
`device_class`).

This is a fail-open path in a component whose stated contract
(`ARCHITECTURE.md`) is that ambiguous or failed validation fails closed. It
does not fail — it quietly ships a capability-stripped tool.

### 1.3 Tool namespacing is real, and secondary

HA 2026.9 wraps tools as `NamespacedTool`, giving `intent__HassTurnOn`,
`media_player__HassSetVolume`, `homeassistant__GetLiveContext`. Training used
plain `HassTurnOn`. **Candidate 2 is confirmed**, and measured below to degrade
tool *selection* but not to be the blocker.

### 1.4 Controlled replay — one factor at a time

Same captured system prompt (which does contain `TV` / `media_player` /
`Living Room`), same utterances, same server, `temperature=0`, no execution:

| Variant | `Turn on the TV.` | `Turn off the living room TV.` | `Turn on the TV in the living room.` |
|---|---|---|---|
| **A** production as-is (namespaced, no params) | no call, "I can't do that…" | no call, "The living room has no media players available." | no call, "I can't do that…" |
| **B** namespaced + real params | `media_player__HassSetVolume(name=TV)` | `intent__HassTurnOff(name=Living Room Media Player)` | `homeassistant__GetLiveContext(name=TV)` |
| **C** plain names + real params | **`HassTurnOn(name=TV, domain=[media_player])`** | `HassTurnOff(name=**Living Room Media Player**)` | **`HassTurnOn(name=TV, domain=[media_player])`** |
| **D** plain names, no params | `HassTurnOn({})` | no call | `HassTurnOn({})` |

Reading:

1. **A reproduces issue #52 exactly**, including both spoken strings.
2. **Parameters are the blocker.** Restoring them takes 0/3 tool calls to 3/3.
3. **Namespacing costs tool selection.** B vs C is one changed factor: with
   namespaced names the model picks the wrong tool on 2 of 3.
4. **One residual is genuine training debt.** In C, `Turn off the living room
   TV` resolves `Living Room Media Player` instead of `TV` — the corpus's
   documented name-echo weakness, where a distractor whose *name* contains the
   room beats the entity whose *area* is the room.

So: two production defects and one data defect, in that order of impact.

## 2. Scope

### In scope

1. `schema.py` — convert through `probatio.to_openapi` when present, keep
   `voluptuous_openapi.convert` for older HA, and **fail closed** when neither
   yields a usable schema instead of shipping empty parameters.
2. `manifest.json` — drop the `voluptuous-openapi==0.4.1` pin that forces the
   broken resolution on HA 2026.9.
3. Training data — teach the namespaced contract and the TV reasoning the model
   actually needs (§3 of the task brief), on a production-sized schema.
4. Regression coverage for every historical failure not already protected,
   including the request/label consistency rules in §2 of the brief.
5. Held-out eval rows for the three exact issue utterances.

### Out of scope

- Retraining. It is a separate, expensive decision and the corpus gate in
  `docs/HANDOFF_gauntlet_v2.md` is still unmet.
- Any TV-specific shortcut, regex absence repair, or target resolver in SaySo.
  Home Assistant stays authoritative for exposure and execution.
- `ARCHITECTURE.md` — no ownership or trust-boundary change.
- Promoting or unpromoting a checkpoint.

## 3. Files to touch

| File | Change |
|---|---|
| `custom_components/sayso/schema.py` | `probatio.to_openapi` path; fail closed on unusable conversion |
| `custom_components/sayso/manifest.json` | remove the `voluptuous-openapi` pin |
| `custom_components/sayso/test_schema_*.py` | regression: probatio schemas keep their parameters; empty-parameter fallback is refused |
| `training/generators/tools.py` | emit the namespaced production contract |
| `training/generators/grounding.py` | TV families: name vs area, alias, room-qualified, ambiguity, absence, speaker distractor |
| `training/generators/pipeline.py` | route the new families; enforce coverage after validation/dedup |
| `training/generators/validate.py` | label/evidence consistency on the rendered row |
| `training/evals/grounding_eval.py` | the three issue utterances, held out |
| `docs/PLAN_ISSUE_52_TV_CONTROL.md` | this file |

## 4. Verification

1. Offline repro of the conversion defect (`probatio` vs `voluptuous-openapi`) — **done**, §1.2.
2. Controlled boundary replay isolating each factor — **done**, §1.4.
3. `probatio.to_openapi` restores real parameters — **done**, §1.2.
4. Unit tests: `pytest custom_components/sayso training/generators training/tests -q`.
   Each new test must fail against the code it guards.
5. Deterministic small corpus; audit the serialized rows, not the config.
6. Coverage enforced on accepted rows after validation and dedup.
7. Held-out eval rows do not leak into training.
8. Live re-verification of the three utterances once the integration fix is
   deployed, scored on resolved tool + target, not on spoken text.

## 4b. What was implemented and measured

### Integration

- `schema.py`: one shared `_convert_parameters()` behind both call sites. Tries
  `voluptuous_openapi.convert`, falls back to `probatio.to_openapi`, and raises
  `SaySoInvalidToolEnvelopeError` when neither yields a usable schema. The
  silent `{"type":"object","properties":{}}` fallback is gone. Both imports are
  optional, so HA 2026.8 and 2026.9 both load.
- `manifest.json` / `pyproject.toml`: the `voluptuous-openapi==0.4.1` pin is
  removed. That pin is what installed the stale converter on HA 2026.9, which
  does not ship the package at all. It stays in the `test` extra so the legacy
  path keeps its coverage.
- Verified against `homeassistant==2026.9.2`: a `HassTurnOn`-shaped tool now
  compiles to `{area, device_class, domain, floor, name}`, and in the exact
  deployed library state (probatio **and** voluptuous-openapi 0.4.1 present)
  it produces the same. With the probatio path removed it raises instead of
  shipping empty parameters.

### Training data

- `grounding.named_device_family` — one fixed request, seven homes: `bare`,
  `room_named`, `aliased`, `speaker`, `absent`, `ambiguous`, `irrelevant`.
  Instantiated canonically (required) plus one wording per generated site,
  rotated over bare / room-qualified / "in the <room>" with the satellite
  elsewhere.
- `gold` gained a `request_intent` (device noun + explicit room). The label is
  still derived from the graph, but the graph is now filtered to what the
  request can actually refer to. Without it the generator produced exactly the
  defect it was meant to teach against: the `speaker` home labelled as ambiguous,
  the `absent` home labelled as "turn on the speaker", and every
  "in the living room" row answered from the satellite's room.
- New `device_absent` response: "There is no TV in the living room." A room
  holding a speaker but no TV must not be labelled "no media players available",
  which is the fabricated absence in the issue.
- Production contract: `tools.HA_TOOL_NAMESPACES` (verified against HA 2026.9.2
  source, `f"{DOMAIN}__{intent_type}"`), applied to the offered schema, the
  rendered label, and the tool names quoted in the system prompt, from one flag.
- `--full-catalog-rate` offers the whole catalogue instead of a subset built
  around the expected answer.
- Both rates default to `0.0` (opt-in). Flipping either default silently changes
  what every existing recipe generates; two suites caught exactly that.
- `coverage.bare_tool_name`: `intent__HassTurnOn` and `HassTurnOn` are one tool.
  Comparing raw names made every namespaced row fail its positive quota, so the
  namespaced share shipped at 11.5% against a 35% request with no error.
- Rate gates now cover both new shares, asserted on accepted rows after
  validation, dedup and the quota.
- `evals/grounding_eval.py` holds the three issue utterances verbatim. The
  previous "exact production regression" anchor actually said "turn on the living
  room media player", which is neither what a user says nor what was reported.
- Bare `TV` is reserved out of the training site vocabulary the way Living Room
  already is: the bare wording carries no area, so a site drawing "TV" reproduced
  the frozen eval prompt verbatim. It did.

### Measured

| Check | Result |
|---|---|
| integration suite | 260 passed, 6 pre-existing failures (unchanged from baseline) |
| training suite | 391 passed, 0 failed |
| recipe corpus, 4,000 rows | namespaced 34.5%, full catalogue 35.9%, exit 0 |
| calls not present in their row's own offered schema | 0 |
| prompt/schema tool-name disagreement | 0 |
| absence rows naming an area absent from context | 0 |
| all seven named contrasts on accepted rows | yes (32 rows, every label ≥ 2) |
| held-out eval prompts appearing in training | 0 |
| determinism, two 4,000-row builds at one seed | byte-identical |

Each new gate was checked against the code it guards: reverting
`bare_tool_name` fails the build with
`namespaced_tools rate shortfall: requested 0.3500 ... achieved 0.1150`.

### Not done

- **No retraining.** The corpus is built and audited; spending GPU time is a
  separate decision, and `docs/HANDOFF_gauntlet_v2.md`'s pre-flight gate is still
  unmet.
- **The integration fix is not deployed.** Home Assistant runs on a separate host
  reachable only over its REST API, so the new `schema.py` could not be copied
  there. Live re-verification of the three utterances is therefore outstanding.
- The 6 pre-existing integration failures are themselves HA 2026.9 namespacing
  fallout (`assert 'intent__HassTurnOn' == 'HassTurnOn'`). They live in `tests/`
  and were failing before this work; they need their expectations updated.

## 5. Risks

- The `probatio` import must stay optional: older HA has no `probatio`, and
  SaySo should not hard-fail there.
- Failing closed on conversion is a behaviour change: a tool that cannot be
  compiled is withheld rather than offered crippled. That is the correct
  direction per `ARCHITECTURE.md`, but it means a broken environment now
  surfaces as an error instead of as silent uselessness.
- Namespaced training data must not displace plain-name coverage; the pinned
  contract and the recipe-lock gold still use plain names.
