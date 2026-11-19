# Plan: batch 4, config entry v2 (Fix 5, Fix 8, Fix 7b)

Scope: plan only. No code, test, or manifest changes. Source of the fixes:
`docs/PLAN_review_fixes.md` (Fix 5 at line 187, Fix 8 at line 196, Fix 7b at
line 221, batch order at line 414). Branch: `ajax/review-batch4`, cut from
`411eeb7`.

Baseline facts (all in HEAD):
- Fix 6 has landed: `51c7c6b` (`async_migrate_entry`, VERSION 1,
  MINOR_VERSION 2) is an ancestor of HEAD. `custom_components/sayso/config_flow.py:275-276`,
  `custom_components/sayso/__init__.py:196-202`.
- Fix 7a has landed: `f1137f2`. `model_store.py:47-54` builds a copied env
  and no longer writes `os.environ`.
- The `.venv` runs HA 2026.8.3 (`pyproject.toml:11,19` declares
  `homeassistant>=2026.8.3`). `~/.cache/sayso-venvs/ha-2026.11` runs HA
  2026.11.1. Both are checked below.

## 0. Decisions needed from the user before slice 1

| ID | Question | Recommendation |
|----|----------|----------------|
| D1 | Version bump. Fix 5 says "bump to 1.3" and Fix 8 says "next minor version". A merge that deletes entries is a structural change. A minor bump lets a 1.20.x build load a main entry with no conversation entity, so the agent silently disappears. | Major bump: `VERSION = 2`, `MINOR_VERSION = 1`. HA refuses to load a higher major version on older code (`config_entries.py:1156` in 2026.8.3, `:1310` in 2026.11), so a downgrade fails closed. |
| D2 | Two external llama.cpp URLs or models today mean two entries with different engines (`config_flow.py:365` unique id `f"{base_url}|{model}"`). The plan says one engine per process. | External: the main owns URL + API key. The model moves to each conversation subentry and is passed per request, so one URL means one main. Embedded: always one main (unique id `sayso_embedded`, `config_flow.py:322`). Two different URLs stay two mains, and that is called out in the release notes. |
| D3 | Downgrade. After migration the legacy entries are gone, so 1.20.x cannot restore them. | Write a one-file snapshot of the pre-migration entries (data, options, entity and device ids) with `homeassistant.helpers.storage.Store` under key `sayso.v1_backup`. The snapshot is never read at runtime. |
| D4 | VAD select, when the platform is re-keyed. Fix 5 says to remove the old entity. | Re-key the registry entry (`new_unique_id`, `config_entry_id`) and keep its `entity_id`, user customizations and restored state. Deleting it forces a new `entity_id`. |
| D5 | Test edits and new test files. Global `CLAUDE.md` says never modify `tests/` unless asked, and the review plan approves `tests/` edits only for Fix 1, 16, 18 (`PLAN_review_fixes.md:11-13`). New files under `tests/` count as modifications. | Approve the list in section 8 explicitly, file by file. |
| D6 | Manifest. Add `"ai_task"` to `dependencies` (`manifest.json`). Optionally add `"single_config_entry": true` (`loader.py:966-968` in 2026.8.3 reads it, `config_entries.py:1525-1531` blocks second flows). | Add `ai_task` to `dependencies`. Set `single_config_entry` only after D2 is decided. |
| D7 | External setup when llama.cpp is unreachable. Fix 7b wants `ConfigEntryNotReady`. Today the error is swallowed (`__init__.py:234-237`), and `ExternalEngine.async_start` is a no-op (`inference.py:267-268`). | Keep today's behavior (setup succeeds, the first request reports the error). Fix 7b's line is changed to say so. |
| D8 | AI Task structured output. | Text by default. When `task.structure` is set, ask for JSON, parse it, and validate it by calling `task.structure(parsed)`. No grammar, no tool-based structure. |
| D9 | Menu drift. `strings.json:4-11` defines a `user` menu (`local`, `external`) and steps `local`, `external`. `config_flow.py:312-343` shows a URL form, not a menu. The `local` download progress needs a home. | Keep the current form. Drop the dead `menu_options`/`local`/`external` strings, or add the menu deliberately. Either way, it is a visible UI change, so ask. |

### Parent decisions

- D1: major bump, `VERSION = 2`, `MINOR_VERSION = 1` (a downgrade fails closed).
- D2: external model moves to each conversation subentry and is passed per
  request; one URL means one main.
- D3: write the `sayso.v1_backup` snapshot before the migration changes anything.
- D4: re-key the VAD select and keep its `entity_id`.
- D5: new test files are fine. Editing the existing files in 8.1 waits for
  the user's approval.
- D6: add `ai_task` to `dependencies`; no `single_config_entry` (D2 allows two mains).
- D7: keep today's external setup behavior.
- D8: text by default; JSON parsed and validated by calling `task.structure`.
- D9: keep the current form; drop the unused `menu_options`/`local`/`external`
  strings (nothing visible changes, because the menu is never shown).
- One PR for the batch.

## 1. Contradictions between the fix text, the code and HA

1. Fix 5 says "remove it in the Fix 6 migration (bump to 1.3)". Fix 6 is already
   landed at minor 2 (`__init__.py:196-202`). There is no 1.2 -> 1.3 step, so
   batch 4 has to add one, and the text needs the D1 decision.
2. Fix 5 misses the teardown bug. `vad_sensitivity.py:129-134` removes the
   registry entry on every unload (`registry.async_remove`). Every options
   reload therefore deletes the user's VAD entity customizations. The teardown
   goes away with the manual platform.
3. Fix 5 cites `vad_sensitivity.py:80-93` for the manual platform. Correct:
   `:79-93`, with `platform.async_add_entities` at `:93`. The constant unique
   id is `const.py:137` (`VAD_SENSITIVITY_UNIQUE_ID`) and is used at
   `vad_sensitivity.py:46`.
4. Fix 6 says `_migrated_embedded_options` runs on every setup at
   `__init__.py:262-267`. That was the old state. Now setup calls only
   `_synced_embedded_options` (`__init__.py:274-276`).
5. Fix 8 says "duck volume becomes an entry option". It already is:
   `config_flow.py:111-148` (`CONF_DUCK_VOLUME` in `_OPTIONS`) and `const.py:129`
   (`OPTION_KEYS`). The defect is the global ducker, not the option location.
6. Fix 8 cites `__init__.py:246-251` and `inference.py:306-312` for a new
   `Llama` per entry. Now: `__init__.py:238` (external engine),
   `__init__.py:258-263` (embedded engine), `inference.py:323-329` (`_load`,
   where `Llama` is constructed).
7. Fix 8 says "keep entity unique ids so `conversation.sayso_*` keeps working".
   The entity has `_attr_name = None` with `has_entity_name` (`conversation.py:340-341`),
   so its `entity_id` comes from the device name, which is the entry title
   (`conversation.py:349-354`). The `entity_id` is stable only because the
   registry keeps it. Migration must never re-derive it.
8. Fix 8 misses the legacy agent registration. `conversation.py:368` calls
   `conversation.async_set_agent(hass, self._entry, self)`, which registers
   the agent under the entry id (`conversation/__init__.py:126-132` in
   2026.8.3, `:105-111` in 2026.11). Pipelines can reference that id, and it
   is not an entity id (`agent_manager.py:172-190`).
9. Fix 8 file list omits `diagnostics.py`, `strings.json`,
   `translations/en.json`, `manifest.json`, `vad_sensitivity.py`,
   `model_store.py` (for the progress flow) and `conversation.py:257-259`
   (`async_wait_for_restores`, which reads the global ducker).
10. Fix 7b cites `__init__.py:229,233`. Now the wheel install is at `:241` and
    the GGUF download at `:245`. `ConfigEntryNotReady` is at `:256` and `:269`,
    not `:243-244,255-257`.
11. Fix 7b cites `model_store.py:71-76` (wheel) and `:107-112` (checksum).
    Now `async_ensure_llama_cpp` raises at `model_store.py:92-98` after
    `_install_llama_cpp` returns False at `:73-79`. The checksum branch is
    `model_store.py:126-134`, and it already deletes the corrupt file before
    raising, so the next retry downloads again. The claim that it "retries
    forever" is wrong for checksum errors. The permanent cases that do retry
    forever are: missing wheel (`SaySoDependencyError`, `__init__.py:255-256`),
    missing non-catalog path (`__init__.py:252-254`), and a GGUF that fails to
    load (`EmbeddedEngine.async_start`, `__init__.py:265-269`).
12. Fix 7b's external-server case (see D7).
13. Fix 7b says the download moves to the config flow. The flow creates the
    entry before any download (`config_flow.py:322-327`), so a failed download
    today leaves an entry that retries forever. Moving the download into the
    progress step fixes this, but only if the entry is created after the step.
14. Fix 8 says "one ConfigSubentryFlow per agent". HA gives subentry types from
    `async_get_supported_subentry_types`, which is a classmethod on the config
    flow (`config_entries.py:3054` in 2026.8.3, `:3262` in 2026.11). The type
    string for AI Task is `ai_task_data` by HA convention (Anthropic:
    `anthropic/config_flow.py:216`, `anthropic/ai_task.py:31`).
15. The CI gate in `AGENTS.md` lists `lint-imports` and
    `python -m evals.generate_realistic --check`. Neither is in
    `.github/workflows/ci.yml` (the jobs run ruff, pytest, the satellite
    pytest, and the compat matrix). Both are run locally in this plan.
16. `Platform.AI_TASK` does not exist in `homeassistant/const.py` (2026.8.3 or
    2026.11). Use the string `"ai_task"` in `PLATFORMS`.

## 2. HA API verification

Every row lists the 2026.8.3 location (`.venv`) first and the 2026.11.1
location (`~/.cache/sayso-venvs/ha-2026.11`) second. Paths are under
`homeassistant/`.

| Item | 2026.8.3 | 2026.11.1 | Notes for SaySo |
|------|----------|-----------|-----------------|
| `ConfigSubentry` dataclass (`subentry_id` default ULID, `unique_id`, `data`, `title`, `subentry_type`) | `config_entries.py:371-378` | `config_entries.py:421-428` | `unique_id` is per entry. The migrated subentry sets it to the legacy entry id so the entity keeps its `unique_id`. |
| `ConfigFlow.async_get_supported_subentry_types(cls, config_entry)` | `config_entries.py:3054-3058` | `config_entries.py:3262-3266` | Same signature. Returns `{type: SubentryFlowClass}`. |
| `ConfigSubentryFlow` base, `async_create_entry`, `async_update_reload_and_abort` | `config_entries.py:3732` (base), `:3744` (create), `:3827` (update-reload-abort) | `:3958`, `:3970`, about `:4066` | `async_update_reload_and_abort` reloads the entry. Creating a subentry does not (see below). |
| `hass.config_entries.async_add_subentry` / `async_remove_subentry` | `config_entries.py:2676-2683` / `:2686-2700` | `:2890-2897` / `:2900-2914` | `async_remove_subentry` calls `dev_reg.async_clear_config_subentry` and `ent_reg.async_clear_config_subentry` (`:2693-2699`). Entities on that subentry are **removed**. Migration must re-point them first. |
| Creating a subentry does not reload the entry | `_async_update_entry` fires update listeners (`config_entries.py:2569`, docstring `:2588-2591`) | `:2773` | Register an update listener in `async_setup_entry` that reloads when the set of subentry ids differs from what the runtime loaded. |
| `ConfigFlow.async_create_entry(..., subentries=...)` | `config_entries.py:3426-3435` | `:3644-3653` | The first user flow can create main plus default conversation subentry in one step. |
| `ConfigEntry.async_migrate` version check and call | `config_entries.py:1136`; version compare `:1156`; `async_migrate_entry` call and bool check `:1175-1195` | `:1285`; `supports_migrate` `:1324-1340`; result handling `_async_process_entry_method_result` `:356-381` | A higher version than the handler returns False (2026.8.3). Returning `True` on success is accepted for custom integrations in 2026.11 (a bool is only reported as a usage error, not failed). Return `None` is the 2026.11 convention. Plan returns `True` to keep `test_migrate_entry.py` meaningful. |
| Setup returns False | `config_entries.py:885-904` (SETUP_ERROR state set, `_async_process_on_unload` runs) | `:1040-1044` | No retry. Use for permanent problems (Repairs). |
| `ConfigEntryNotReady` retry | raised from `async_setup_entry`; scheduled retry | same | Use only for transient failures. |
| `async_show_progress` / `async_show_progress_done` | `data_entry_flow.py:791` / `:850` | `:865` / `:926` | Fields `progress_action` (`:139`), `progress_task` (`:140`), FlowResultType `progress` / `progress_done` (`:27-36`). Result type names are the same in both versions. |
| `single_config_entry` manifest flag | `loader.py:966-968`; flow abort `config_entries.py:1525-1531` | `loader.py:1110-1112`; `:1659-1665` | Blocks new flows only, not migrated entries. |
| Entity add with subentry | `AddConfigEntryEntitiesCallback` protocol `helpers/entity_platform.py:139-152`; `async_add_entities(..., config_subentry_id=...)` `:744-749` | `:146-159` protocol; `:752-757` | `async_add_entities([...], config_subentry_id=sub.subentry_id)` from `async_setup_entry`. |
| Entity registry: `config_entry_id`, `config_subentry_id`, `new_unique_id` on update | `helpers/entity_registry.py:2021-2033` (`async_update_entity`) | `:2254-2266` | Relinks an entity without touching its `entity_id`. |
| Entity registry: clearing a subentry removes entities | `helpers/entity_registry.py:2433-2443` (`async_clear_config_subentry`) | `:2712-2722` | Relink first, then remove. |
| Device registry: `add_config_subentry_id`, `remove_config_entry_id`, `new_identifiers` | `helpers/device_registry.py:2490-2509` | `:3953-3972` | Relinks the migrated device to main and subentry. Keep the identifier `(DOMAIN, legacy_entry_id)` so the device is not duplicated. |
| `conversation.async_set_agent` | `components/conversation/__init__.py:126-132` (keyed `config_entry.entry_id`) | `:105-111` | Legacy, entry-keyed. Remove it for subentry agents. |
| Entity-based agent lookup by entity id | `components/conversation/agent_manager.py:69` (`get_entity` when the id contains `.`) | `:70` | Entity agents are addressed by `entity_id`, not entry id. |
| Assist pipeline store | `components/assist_pipeline/pipeline.py:330` (`async_get_pipelines`), `:337` (`async_update_pipeline`, takes `conversation_engine`) | `:271`, `:278` | Public via `assist_pipeline/__init__.py:38-40`. Use it to rewrite a pipeline that stored the legacy entry id. |
| `ai_task` exports | `components/ai_task/__init__.py` exports `AITaskEntity`, `AITaskEntityFeature`, `GenDataTask`, `GenDataTaskResult` | same | No `Platform.AI_TASK`; use `"ai_task"`. |
| `AITaskEntity._async_generate_data(task, chat_log)` | `components/ai_task/entity.py:108-115` | `:115-122` | 2026.11 also adds a `context` argument to the wrapper `internal_async_generate_data` (`:101`) and `_async_get_ai_task_chat_log` (`:62`). Plan overrides only `_async_generate_data`. |
| `AITaskEntityFeature.GENERATE_DATA` | `components/ai_task/const.py:37-41` | `:43-47` | Set `_attr_supported_features`. |
| `GenDataTask.structure` type | `components/ai_task/task.py:253` (`vol.Schema`) | `:283` (`probatio.Schema`) | Version difference. Call `task.structure(parsed)` to validate; do not introspect the schema. |
| `GenDataTaskResult(conversation_id, data)` | `components/ai_task/task.py:269` | `:299` | Return text or validated dict. |
| Reference: Anthropic AI Task | n/a | `components/anthropic/ai_task.py:31` (type check), `:50` (translation key), `:59` (`_async_handle_chat_log` with structure), about `:68-83` (JSON parse) | Plan follows this shape, minus tools. |
| `issue_registry.async_create_issue` | `helpers/issue_registry.py:339-351` (`is_fixable`, `severity`, `translation_key`, `data`) | `:594-607` | `is_persistent=False`. Use `data={"entry_id": ...}`. |
| `issue_registry.async_delete_issue` | `helpers/issue_registry.py:415` | `:674` | Call on successful fix. |
| `RepairsFlow` base and `issue_id` | `components/repairs/models.py:12-18`; `issue_id` is a plain attribute set by the manager (`issue_handler.py:75`) | `models.py:36-62`; `issue_id` is a property read from the flow context, with a setter; dispatch at `issue_handler.py:95` | Version difference. Do not rely on `issue_id`; pass `entry_id` through `issue.data`. |
| Repairs fix-flow protocol | `components/repairs/models.py:24` (`async_create_fix_flow`) | `models.py:148` | Repairs platform file: `repairs.py`, `async_create_fix_flow(hass, issue_id, data)`. |

Known gaps: 2026.11 adds `ai_task` evaluation (`entity.py:147-157`). SaySo
does not implement it, so nothing changes. The
`probatio` schema library is present in the 2026.11 venv. Plan does not depend on
it.

## 3. Current code facts (HEAD `411eeb7`)

- `__init__.py:82` `SaySoConfigEntry = ConfigEntry[SaySoRuntimeData]`.
  `SaySoRuntimeData` (`:97-111`) mixes engine, traces and tracer
  (process-level) with prompt, llm_api, temperature, max_output_tokens,
  max_tool_iterations (conversation-level).
- `__init__.py:273-337` (`async_setup_entry`): synced options (`:274-276`),
  engine build (`:278`, `:220-270`), trace store (`:280-293`), VAD (`:296`),
  ducking (`:301-306`), runtime_data (`:308-327`), forward `conversation`
  (`:331-336`). `async_unload_entry` (`:340-350`) tears down VAD (`:347-348`).
- `__init__.py:220-270` (`_async_build_engine`): external branch
  `:227-238` (validate error swallowed `:234-237`); embedded branch
  `:239-256` (`async_ensure_llama_cpp` `:241`, `async_ensure_model` `:245`,
  path check `:252-254`, NotReady `:256`); `engine.async_start` NotReady
  `:265-269`.
- `conversation.py:104-110` (`async_setup_entry`) creates one entity.
  `_Turn` (`:112-124`) carries `runtime`. `_Turn.complete` (`:210-249`) calls
  `runtime.engine.async_chat_completion` (`:221-226`), passing
  `temperature` and `max_output_tokens` from runtime. `_Turn.execute` waits for
  ducking at `:257-259`. Entity `:336-374`: `unique_id = entry.entry_id`
  (`:346`), device identifiers `(DOMAIN, entry_id)` (`:349-354`), legacy agent
  registration `:366-368`, unregistration `:371-374`.
- `conversation.py:635` `area_id_for` is imported by `ducking.py:44`.
- `ducking.py:49-50` (`DUCKER_KEY`, `ENTRY_COUNT_KEY`); `MediaDucker` `:65`;
  `async_setup_ducking` `:300-310` (global, ref-counted, last duck volume
  wins); `async_wait_for_restores` `:313-319`; `async_release_ducking`
  `:322-339`.
- `vad_sensitivity.py:40-56` entity with `_attr_unique_id` constant (`:46`);
  `async_setup_vad_sensitivity` `:79-126` (manual `EntityPlatform` `:83-93`);
  `async_teardown_vad_sensitivity` `:129-136` (registry remove `:131-134`);
  `patch_satellite_entities` `:66-76` (kept).
- `config_flow.py`: `_OPTIONS` `:111-148`; `_OPTION_DEFAULTS` `:150`;
  `async_step_user` `:312`; `async_step_model` `:345`;
  `async_get_options_flow` `:378`; `SaySoOptionsFlowHandler` `:386`;
  `async_step_init` `:421`. Embedded options flow writes
  `{**entry.options, **user_input}` (`:428`).
- `model_store.py`: `list_local_models` `:36-40`; `_install_llama_cpp`
  `:47-80`; `async_ensure_llama_cpp` `:83-102`; `async_ensure_model`
  `:113-170`.
- `inference.py`: `SaySoInferenceEngine` protocol `:233-255`; `ExternalEngine`
  `:257-296` (model fixed at construction `:259-261`); `EmbeddedEngine`
  `:298-` (`_load` `:323-329`, `async_chat_completion` `:398-`).
- `diagnostics.py:118-180`: reads `runtime_data` (`:128-160`) and dumps
  `entry.options` (`:175`).
- `strings.json`: `config.step.user.menu_options` (`:4-11`) and steps
  `local`, `external` (`:12-17`) with no matching code; `options.step.init`
  (`:44-`). No `config_subentries`, `issues`, or `progress` sections.
  `translations/en.json` has the same top-level keys and is kept in step.
- `const.py`: `OPTION_KEYS` `:114-130`; `CONF_DUCK_VOLUME` `:110-111`;
  `DEFAULT_TRACE_STORE_UTTERANCES` `:95`; `CONF_GRAMMAR_DECODING` `:97-98`;
  `VAD_SENSITIVITY_UNIQUE_ID` `:137`; `EMBEDDED_MODELS` `:61-77`.
- `manifest.json`: version `1.20.4`; `dependencies: [conversation, llm]`;
  no `single_config_entry`.
- `trace_store.py:32` `STORAGE_KEY = "sayso.traces"` (global, not per
  entry, so it survives the merge).

## 4. Target design

### 4.1 Entry model (after the migration)

Main entry (version 2, minor 1 under D1):
- `data`: unchanged (`backend`, `url`, `api_key`). Embedded: `{backend: embedded}`.
- `options` (engine and process level): `n_threads`, `n_ctx`, `timeout`,
  `grammar_decoding`, `duck_volume`, `trace_retention_days`,
  `trace_max_interactions`, `trace_store_utterances`, plus `model`
  (embedded always; external only if D2 keeps the model on main).
  `model_path` stays `""` (batch 1).
- `subentries`:
  - `conversation` (`subentry_type = "conversation"`): `llm_hass_api`,
    `prompt`, `temperature`, `max_output_tokens`, `max_tool_iterations`,
    and `model` if D2 moves it here (external).
  - `ai_task_data` (`subentry_type = "ai_task_data"`): no options in v1;
    `title` only.
- `runtime_data` (main, `SaySoRuntimeData`): `engine`, `client`, `model`,
  `traces`, `tracer`, `vad`, `grammar_decoding`, `ducker`. The
  conversation-level fields move out.

Per conversation subentry: `SaySoConversationEntity(entry, subentry)`. Its
per-turn runtime is built as
`dataclasses.replace(entry.runtime_data, llm_api=..., system_prompt=..., temperature=..., max_output_tokens=..., max_tool_iterations=...)`
(plus `model` if D2 moves it). `_Turn` and `_Turn.complete` keep their
shape, so `_Turn` stays the same type and the trace code is unchanged.

Unique ids and device identifiers:
- Conversation entity `unique_id = subentry.unique_id or subentry.subentry_id`.
  Migrated subentries set `unique_id` to the legacy entry id, so the registry
  row is matched without rewriting it.
- Device identifiers `(DOMAIN, subentry.unique_id or subentry.subentry_id)`.
  Migrated devices keep `(DOMAIN, legacy_entry_id)`.
- AI Task entity `unique_id = subentry.subentry_id`, its own device.
- VAD select `unique_id = f"{main.entry_id}-vad-sensitivity"` (Fix 5).

Per-process state:
- Ducker: one `MediaDucker` per main entry, created in `async_setup_entry`,
  stored on `runtime_data.ducker`, torn down with
  `entry.async_on_unload(ducker.async_teardown)`.
- `async_setup_ducking(hass, duck_volume) -> MediaDucker` no longer writes
  `hass.data`. It creates, starts and returns a ducker. This keeps the 3 call
  sites in `tests/test_volume_wait.py` working (shim).
- `async_wait_for_restores(hass)` is removed. `_Turn` gets an optional
  `ducker: MediaDucker | None = None` field (default None), passed from the
  entity. `_Turn.execute` awaits `self.ducker.async_wait_restores()` when set.
  Tests that build `_Turn` directly keep working.

### 4.2 Options split and flows

- Main options flow (`SaySoOptionsFlowHandler`): engine fields only. Embedded
  keeps the model picker, `n_threads`, `n_ctx`. External keeps `timeout` and
  the model picker if D2 keeps it here. Both keep the trace fields,
  `grammar_decoding`, `duck_volume`. Model changes that need a download re-enter
  the progress step (Fix 7b item 3).
- Conversation subentry flow (`SaySoConversationSubentryFlow`):
  `async_step_user` (title + llm_hass_api, prompt, temperature, max_output_tokens,
  max_tool_iterations, plus model if D2) and `async_step_reconfigure`
  (`async_update_reload_and_abort`). `supports_reconfigure` is automatic
  (`config_entries.py:632-633`).
- AI Task subentry flow (`SaySoAITaskSubentryFlow`): `async_step_user` (title
  only), `async_step_reconfigure` (title only).
- `SaySoConfigFlow.async_get_supported_subentry_types` returns
  `{"conversation": ..., "ai_task_data": ...}`.
- The user flow creates the main entry and a default conversation subentry in
  one `async_create_entry(..., subentries=[...])` call (`config_entries.py:3426-3435`).
  Embedded and external both do this, so a fresh install still gets a
  working agent.

### 4.3 Reuse, not duplication

- Conversation options, prompt building, tool execution, and the grammar path
  stay in `conversation.py`. The only change is where the per-turn runtime
  comes from.
- AI Task uses the same engine call, not a new one. The entity calls
  `entry.runtime_data.engine.async_chat_completion(messages, tools=None,
  temperature=DEFAULT_TEMPERATURE, max_tokens=DEFAULT_MAX_OUTPUT_TOKENS, grammar=None)`
  (`inference.py:245-254` protocol). Messages come from
  `transcript.chat_log_to_messages` (`transcript.py:15`) over the `ChatLog` HA
  builds. Tool calls are never requested, so `_turn_grammar` (`conversation.py:607`)
  is not needed.
- `ExternalEngine` takes `model` at construction (`inference.py:259-261`). If D2
  keeps the model on the subentry, the engine accepts a per-call `model`
  argument (default: its own), and the `ChatCompletionResult` contract stays the
  same. This is the only inference change in batch 4.

## 5. Migration (VERSION 1 -> 2, MINOR 1 under D1)

Runs once, from `async_migrate_entry`, for every entry with `version == 1`.
Version 2 entries are untouched. A version above 2 returns False.

### 5.1 Grouping

Legacy entries are grouped by engine identity:
- Embedded: one entry (unique id `sayso_embedded`). Group of one.
- External: key `(normalized url, api_key)` under D2. Model is per subentry
  and does not split groups. Without D2, key is `(url, api_key, model)`.

Within a group, `main` is the entry with the earliest `created_at`. Its
`entry_id` is kept, so references to the main entry keep working. Under D2
the main's model moves to its own conversation subentry.

### 5.2 Single-entry path (primary, most users)

Input: one entry E, options O, data D.
1. Build the conversation subentry:
   `subentry_type="conversation"`, `unique_id=E.entry_id`,
   `title=E.title`, `data={llm_hass_api, prompt, temperature, max_output_tokens, max_tool_iterations}`
   from O, falling back to the defaults in `config_flow.py:111-150`. Under D2,
   add `model` from O.
   Add with `hass.config_entries.async_add_subentry(E, subentry)`.
2. Relink the conversation entity: find the entity registry row with
   `platform=="sayso"`, `domain=="conversation"`, `unique_id==E.entry_id`. Call
   `er.async_update_entity(entity_id, config_subentry_id=sub.subentry_id)`.
   `entity_id` does not change.
3. Relink the device: find the device with identifier `(DOMAIN, E.entry_id)`.
   Call `dr.async_update_device(id, add_config_subentry_id=sub.subentry_id)`.
   Identifiers do not change.
4. Re-key the VAD select (D4): find the row with
   `unique_id == VAD_SENSITIVITY_UNIQUE_ID` on the `select` platform, then
   `er.async_update_entity(eid, new_unique_id=f"{E.entry_id}-vad-sensitivity", config_entry_id=E.entry_id)`.
5. Strip conversation keys from options (`CONF_PROMPT`, `CONF_LLM_HASS_API`,
   `CONF_TEMPERATURE`, `CONF_MAX_OUTPUT_TOKENS`, `CONF_MAX_TOOL_ITERATIONS`,
   and `CONF_MODEL` under D2). Set `version=2`, `minor_version=1` in the same
   `async_update_entry` call (`config_entries.py:2531-2545`).
6. Rewrite assist pipelines (section 5.5).
7. Return True.

### 5.3 Multi-entry path (rare)

For a group with entries E1..En and main M (earliest):
1. Build the conversation subentry for each Ei, `unique_id=Ei.entry_id`.
   Add all to M. Options from M win. Differences in `duck_volume`, timeout, or
   trace settings are logged and listed in the release notes.
2. Relink each Ei's entity and device to `(M, sub_i)`, as in 5.2 steps 2-3.
3. Re-key the single VAD row to M (step 4 once, not per entry).
4. Remove each Ei after its relink: `await hass.config_entries.async_remove(Ei.entry_id)`.
   Removal clears only entities and devices still pointing at Ei, and by then
   there are none.
5. Set M to version 2, minor 1, strip keys, rewrite pipelines.

Steps 1-3 run without `await` between reads and writes, so no other
migration task can interleave. Only the removals are awaited.

### 5.4 Idempotency and restarts

Every step is check-before-act:
- Subentry exists on M with `unique_id == Ei.entry_id` -> skip the add.
- Entity or device already points at `(M, sub_i)` -> skip the relink.
- Legacy entry still exists -> remove it. `UnknownEntry` -> ignore.
- Main still at version 1 is the only signal to run the migration. The
  version bump is the last write, so a crash before it reruns everything, and
  reruns are no-ops for completed steps.

Crash risk: HA saves config entries and the registries on separate delayed
writes. A crash can persist a relinked entity while the subentry add is lost.
The rerun matches by `unique_id` and re-adds the subentry before relinking, so
the entity is recovered. Verify this with a kill-during-migration test in
slice 5. Do not assume it.

Concurrency: HA can set up other legacy entries while M migrates. Each legacy
entry's `async_migrate_entry` runs the same idempotent group merge (keyed by M's
id), so whichever starts first does the work. A legacy entry that finds itself
gone returns True and never reaches setup.

### 5.5 Pipelines and automations

- Assist pipelines whose `conversation_engine` equals a legacy entry id are
  rewritten to the new entity id with `async_update_pipeline(hass, pipeline.id,
  conversation_engine=entity_id)` (`pipeline.py:337` / `:278`), after the entity
  exists. Entity-id pipelines (`pipeline.py:299`) need no change.
- Automations that call `conversation.process` with `agent_id=<entry id>` are
  not rewritten. They are listed in the release notes.

Slice 5 includes a test that imports a pipeline with the legacy id and checks
the rewrite.

### 5.6 Downgrade

- 1.20.x sees `version == 2` and refuses to migrate (`config_entries.py:1156`
  in 2026.8.3). The entry is not loaded, and the agent disappears until the
  user upgrades again. This is fail-closed. It is not silent.
- Without D3, the legacy entries cannot be restored, so downgrade is
  unsupported. With D3, the snapshot is kept and documented for a manual restore.
- The trace store (`sayso.traces`, `trace_store.py:32`) is global and survives
  both directions.

### 5.7 Tests the migration needs (new, under approval)

- single entry, embedded: subentry created, entity_id and registry row kept, VAD re-keyed, options stripped, version 2.1.
- single entry, external (D2 both branches).
- two entries, same URL: one main, two subentries, both entities relinked, second entry removed.
- two entries, different URLs: two mains, no data loss.
- entry already at version 2: untouched.
- future version 3: returns False.
- rerun after a simulated crash between subentry add and relink: same end state.
- pipeline with legacy `conversation_engine` is rewritten.

## 6. Fix 7b: progress flow, Repairs, setup outcomes

### 6.1 Flow (config flow and options flow)

- New step `async_step_download` (config flow, after the user step for embedded)
  and `async_step_download` (options flow, when the model changes to a catalog
  model or to a missing file). Both return
  `async_show_progress(step_id="download", progress_action="download", progress_task=task)`.
  `task` runs `async_ensure_llama_cpp(hass)` and then
  `async_ensure_model(hass, url=..., filename=..., sha256=...)` (`model_store.py:83-102`, `:113-170`).
- On success, `async_show_progress_done(next_step_id="finish")`. `finish`
  calls `async_create_entry` (config flow) or `async_create_entry(data=...)`
  (options flow).
- On `SaySoError`, `async_show_progress_done(next_step_id="download_failed")`.
  That step shows an error with a retry, which re-runs the progress step.
- Embedded local: the entry is created only after the progress step succeeds,
  so a failed download leaves no entry behind.
- External: no progress step, no wheel install (`async_ensure_llama_cpp` is only
  called for embedded, `__init__.py:241`).
- Model file in the progress step: `model_store.async_ensure_model` already
  checks the checksum before returning, so the progress step never accepts a
  corrupt file.

### 6.2 Setup outcomes

Setup only loads files that already exist. It never downloads.

| Condition | Outcome | Detail |
|-----------|---------|--------|
| External, server unreachable | Setup succeeds (today's behavior, D7) | `__init__.py:234-237`, `inference.py:267-268`. |
| Embedded, model file missing (catalog) | `async_setup_entry` returns False; Repairs issue `model_missing` (fixable) | No retry. Fix flow = the download progress step. |
| Embedded, checksum mismatch on an existing file | Repairs issue `model_corrupt` (fixable) | `model_store.py:126-134` unlinks the file. Setup detects the missing file and takes the row above. |
| Embedded, custom path missing (`__init__.py:252-254`) | Repairs issue `model_path_missing` (not fixable; fix = options) | Setup returns False. |
| Embedded, GGUF fails to load (`EmbeddedEngine.async_start` error, `__init__.py:265-269`) | Repairs issue `model_corrupt`; setup returns False | Permanent until the file changes. |
| Embedded, wheel not installed or no prebuilt wheel (`SaySoDependencyError`, `model_store.py:92-98`) | Repairs issue `llama_cpp_missing` (fixable) | Fix flow = progress step that runs `async_ensure_llama_cpp`. |
| Transient: file system or network error during a load | `ConfigEntryNotReady` | Only for `SaySoConnectionError`/`SaySoTimeoutError` during setup. |

Setup returning False maps to `SETUP_ERROR` (`config_entries.py:885-904`,
`:1040-1044`). Repairs issues are created with
`ir.async_create_issue(hass, DOMAIN, issue_id, is_fixable=..., severity=ERROR, translation_key=..., data={"entry_id": ...})`.
A successful fix deletes the issue with `ir.async_delete_issue` and reloads the
entry.

### 6.3 Repairs flow

- `repairs.py` with `async_create_fix_flow(hass, issue_id, data)`. Returns a
  `RepairsFlow` whose `async_step_init` shows the progress step (download or
  wheel install) and then `async_create_entry(data={})`.
- The flow reads `entry_id` from `data`, not from `self.issue_id` (version
  difference, section 2).
- Strings: `issues` section in `strings.json`, mirrored to `translations/en.json`.
  Batch 4 keeps the two files in step. Fix 9 (batch 5) only does the rest.

### 6.4 External mode

The fix flow, the progress step, and the wheel install are embedded-only.
`ExternalEngine` never touches `model_store`. The external path has no
`model_missing` or `llama_cpp_missing` issues.

## 7. AI Task

- `ai_task.py` (new): `async_setup_entry(hass, entry, async_add_entities)`
  adds one `SaySoAITaskEntity` per `ai_task_data` subentry with
  `config_subentry_id=sub.subentry_id`. `PLATFORMS` gains `"ai_task"`.
- `SaySoAITaskEntity(AITaskEntity)`:
  - `_attr_supported_features = AITaskEntityFeature.GENERATE_DATA`.
  - `_async_generate_data(task, chat_log)`:
    1. If `task.attachments` is set, raise `HomeAssistantError` (text only in v1).
    2. `messages = chat_log_to_messages(chat_log.content)`.
    3. If `task.structure` is set, append a system message asking for a JSON
       object only (D8).
    4. `result = await runtime.engine.async_chat_completion(messages, tools=None, temperature=DEFAULT_TEMPERATURE, max_tokens=DEFAULT_MAX_OUTPUT_TOKENS)`.
    5. With no structure, return `GenDataTaskResult(conversation_id=chat_log.conversation_id, data=result.content or "")`.
    6. With structure, `json.loads(result.content)`, then `data = task.structure(parsed)`.
       `vol.Invalid`/`probatio` invalid and `JSONDecodeError` raise
       `HomeAssistantError(translation_domain=DOMAIN, translation_key="ai_task_invalid")`.
- Not in v1: tools (`llm_api` ignored), evaluation, image generation, grammar,
  conversation memory across tasks (HA gives each task a fresh chat log).
- Trace: an AI Task call is not traced in v1. Adding a span is a later step.
- Concurrency: the same `EmbeddedEngine` executor serializes AI Task and
  conversation turns. That is intended.
- Feature flags: `_attr_supported_features` is read by HA for the service
  handler (`ai_task/__init__.py:149`). Verify on both versions in slice 6.

## 8. Test impact

### 8.1 Edits that cannot be avoided (approval per file)

| File | Why | Lines (current) |
|------|-----|-----------------|
| `tests/test_ducking.py` | Imports `DUCKER_KEY`/`ENTRY_COUNT_KEY` (`:29-33`). `test_two_entries_share_one_ducker` (`:557`) asserts the global ref count, which is the behavior removed. `test_wait_for_restores_is_noop_without_ducker` (`:861`) and the `hass.data` setup in `:865` depend on `async_wait_for_restores`. | `:29-33`, `:557-593`, `:861-869` |
| `tests/test_vad_sensitivity.py` | Imports and calls `async_setup_vad_sensitivity(hass)` (`:24-26`, `:100`, `:112`, `:124`, `:139`), which is replaced by the platform. Entity id assertion `select.sayso_vad_sensitivity` (`:36`) must still hold. | `:24-26`, `:100-139` |
| `tests/test_migrate_entry.py` | Asserts `minor_version == 2` after migration (`test_migrates_legacy_model_path_in_catalog`, `test_migrates_legacy_absolute_model_path`, `test_minor_version_two_options_untouched`). Under D1 the minor is 1 of version 2. `test_future_version_fails_migration` uses version 2 as "future". | whole file |
| `tests/test_model_catalog.py` | `fake_runtime` (`:164-200`) patches `async_ensure_llama_cpp`/`async_ensure_model` in `__init__` and expects downloads during setup (`test_catalog_model_is_downloaded_and_loaded`, `:203`). `test_missing_model_path_is_not_ready` (`:230`) expects NotReady; Fix 7b changes it to setup False. | `:164-241` (2-3 tests) |
| `tests/test_init.py` | `test_options_reload_updates_runtime_data` sets `temperature` on the main options and asserts `runtime_data.temperature` (`:67-89`). Temperature is now per subentry. | `:67-89` |
| `tests/test_config_flow.py` | Options flow sets `temperature` (`:256-257`); the field is removed from main options. | `:225-257` (approximate) |
| `tests/test_conversation.py` | `_converse` passes `agent_id=entry.entry_id` (`:303`), which is no longer the agent id. `:271` reads `runtime_data.llm_api`. `:1319-1324` sets `CONF_MAX_TOOL_ITERATIONS` on the main options. | 3 spots |
| `tests/test_schema_identity.py` | `entry.runtime_data.llm_api` (`:155`). | 1 line |
| `tests/test_diagnostics.py` | Sets `CONF_MAX_TOOL_ITERATIONS` on the main options (`:286`). | 1 spot |

### 8.2 Left untouched through compatibility shims

| File | Why it survives |
|------|-----------------|
| `tests/test_volume_wait.py` | Calls `async_setup_ducking(hass, 0.02)` (`:100`, `:144`; 3 calls per the grep count) and uses the returned ducker. The shim returns a standalone ducker with the same signature. |
| `tests/test_area_arg_drop.py`, `tests/test_failure_speech.py`, `tests/test_grounding_flow.py`, `tests/test_match_failure_speech.py`, `tests/test_tool_result_error.py`, `tests/test_supported_languages.py` | Build `SaySoConversationEntity(MockConfigEntry(...))` with one argument. The constructor takes `subentry=None` as the default. |
| `tests/test_history_question_gate.py` | Builds `_Turn(...)` with a `SimpleNamespace` runtime and no `ducker`. `ducker` defaults to None. |
| `tests/test_grammar_wiring.py`, `tests/test_grammar_threading.py`, `tests/test_grammar_fallback.py`, `tests/test_grammar_external.py` | Use `runtime_data.grammar_decoding`, which stays on main, and `_converse`, which is fixed once in `test_conversation.py`. |
| `tests/test_model_option_sync.py` | Imports `_synced_embedded_options` (kept by Fix 6). |
| `tests/test_model_store_install.py`, `tests/test_inference.py` (`async_ensure_model` block, `:520-575`) | `model_store` function names and signatures are unchanged. |
| `tests/test_tracing.py`, `tests/test_trace_utterance_default.py` | `runtime_data.traces` stays on main. |
| `tests/test_live_command_smoke.py`, `tests/test_live_latency.py` | Live-gated. Re-check in slice 7 only. |

### 8.3 New test files (also under approval, D5)

- `tests/test_migrate_v2.py`: migration cases in 5.7.
- `tests/test_subentry_flows.py`: conversation and AI Task subentry flows.
- `tests/test_repairs.py`: issues, fix flow, setup False.
- `tests/test_ai_task.py`: text and structured output, invalid JSON.

## 9. Slices

Each slice is independently green on `.venv` (HA 2026.8.3). Slice 7 is the
release gate. Each slice re-runs the gate for everything it touches. "Gate" is
the CI command list in section 10.

S1. VAD select platform (Fix 5) and migration skeleton.
- Files: `select.py` (new), `vad_sensitivity.py`, `__init__.py`,
  `const.py`, `tests/test_vad_sensitivity.py` (approval), `tests/test_migrate_entry.py` (approval, D1).
- Work: `select.py` adds the entity with `unique_id = f"{entry.entry_id}-vad-sensitivity"`
  and `device_info` (as `conversation.py:349-354`). Remove the manual platform
  and the teardown registry removal (`vad_sensitivity.py:83-93`, `:131-134`). Keep
  `patch_satellite_entities`. Add the VAD re-key (5.2 step 4) in the migration.
  Set `VERSION=2`, `MINOR_VERSION=1` (D1), and migrate v1 entries (step 5 only,
  no subentries yet).
- Verify: `pytest tests/test_vad_sensitivity.py tests/test_migrate_entry.py tests/test_init.py -q`; gate.

S2. Download progress flow and Repairs (Fix 7b).
- Files: `config_flow.py`, `options flow` (same file), `__init__.py`, `model_store.py`,
  `repairs.py` (new), `strings.json`, `translations/en.json`,
  `tests/test_model_catalog.py` (approval), `tests/test_config_flow.py` (approval), `tests/test_repairs.py` (new, approval).
- Work: 6.1-6.4. Setup stops downloading. Setup returns False for the
  permanent cases. The download moves to the progress step. Embedded local
  creates the entry only after success.
- Verify: `pytest tests/test_model_catalog.py tests/test_config_flow.py tests/test_repairs.py tests/test_inference.py tests/test_model_store_install.py -q`; gate.

S3. Per-main ducker and runtime_data (Fix 8, part 1).
- Files: `ducking.py`, `__init__.py`, `conversation.py` (`_Turn` field and
  `:257-259`), `tests/test_ducking.py` (approval), `tests/test_volume_wait.py` (no edit expected).
- Work: remove `DUCKER_KEY`/`ENTRY_COUNT_KEY`/`async_wait_for_restores`, shim
  `async_setup_ducking`, store the ducker on `runtime_data`, `async_on_unload`
  teardown.
- Verify: `pytest tests/test_ducking.py tests/test_volume_wait.py tests/test_volume_step.py -q`; gate.

S4. Subentries (Fix 8, part 2): entry model without migration.
- Files: `config_flow.py` (subentry flows, `async_get_supported_subentry_types`,
  user flow creates default subentry), `__init__.py` (options split, update listener),
  `conversation.py` (`SaySoConversationEntity(entry, subentry=None)`, per-turn
  runtime via `replace`, unique id and device from subentry, drop
  `async_set_agent` for subentries), `diagnostics.py`, `strings.json`,
  `translations/en.json`, `inference.py` (only under D2), tests: `test_conversation.py`,
  `test_init.py`, `test_config_flow.py`, `test_schema_identity.py`, `test_diagnostics.py`
  (approval), `tests/test_subentry_flows.py` (new, approval).
- Verify: `pytest tests/test_conversation.py tests/test_init.py tests/test_config_flow.py tests/test_subentry_flows.py tests/test_diagnostics.py -q`; gate.

S5. Migration of legacy entries (Fix 8, part 3).
- Files: `__init__.py` (`async_migrate_entry`, section 5), new helper module
  for the group merge if it stays out of `__init__.py`,
  `tests/test_migrate_v2.py` (new, approval), `tests/test_migrate_entry.py`
  (approval, the assertions move to version 2.1).
- Work: 5.1-5.7. D3 snapshot if approved.
- Verify: `pytest tests/test_migrate_v2.py tests/test_migrate_entry.py -q`; a kill-during-migration test (5.4); gate.

S6. AI Task (Fix 8, part 4).
- Files: `ai_task.py` (new), `__init__.py` (`PLATFORMS`), `manifest.json` (D6),
  `strings.json`, `translations/en.json`, `tests/test_ai_task.py` (new, approval).
- Verify: `pytest tests/test_ai_task.py -q`; gate.

S7. Release gate and live check.
- Run the CI gate on `.venv` (2026.8.3) and on `ha-2026.11` (add the venv to
  `PYTHONPATH` only if the harness supports it; otherwise run the HA-touching
  tests with the 2026.11 install). Live check: install on a test HA host
  (`ssh llm` per memory, ask first), run a migration with a copy of a real
  entry, check the pipeline and the VAD entity id.
- Verify: full gate; `lint-imports` (pyproject contracts); `python -m evals.generate_realistic --check`.

## 10. Verification commands

Run from `/Users/matt/Projects/SaySo__worktrees/ajax-codebase` with the venv
first on `PATH`:

```
export PATH=/Users/matt/Projects/SaySo__worktrees/ajax-codebase/.venv/bin:$PATH
ruff check .
ruff format --check .
pytest tests evals/tests --ignore=tests/test_realistic_v3.py
pytest -q satellite/sayso        # satellite venv: ~/.cache/sayso-venvs/satellite
lint-imports
python -m evals.generate_realistic --check
```

Baseline before any slice: run the gate on `ajax/review-batch4` at `411eeb7`
and record the count. Do not start S1 without a green baseline.

## 11. Risks

1. Data loss on migration: a merged entry removed before its entity is relinked
   deletes the entity registry row and its `entity_id` (`entity_registry.py:2433-2443`).
   Mitigation: relink first; test in S5; D3 snapshot.
2. Broken pipelines: pipelines that store the legacy entry id (`conversation.py:368`
   legacy registration) stop resolving. Mitigation: 5.5 rewrite; release-note
   list for automations.
3. HACS users on 2026.8.3: every API in section 2 exists in 2026.8.3. The
   Repairs fix flow must not read `issue_id` directly. `GenDataTask.structure`
   is a different schema type in 2026.11, so validation calls the schema.
4. HA crash between config and registry writes (5.4). Mitigation: check-before-act
   and a kill test.
5. Ducking regression. The global ref count is replaced by one ducker per main.
   With two mains (D2 different URLs) two duckers can race on the same media
   player. Mitigation: documented; a cross-main coordinator is a separate
   change.
6. Fix 6 migration interaction. `_migrated_embedded_options` (`__init__.py:205-217`)
   must still run before the new v2 step on any entry that is still v1.
   Mitigation: the v2 step reads options after the legacy rewrite. Test in S5.
7. `ExternalEngine` model per request (D2): changes the chat completion
   contract. Mitigation: default to the constructor model; covered by
   `tests/test_inference.py`.
8. Entity id stability. The `entity_id` comes from the device name at first
   registration (`conversation.py:349-354`). A new subentry on an existing main
   gets an entity id from its title. Mitigation: never re-derive; document.
9. The release is a major version bump (D1). HACS shows a major change, and
   users must confirm the migration. Mitigation: changelog text.
10. Test edits are large (section 8.1). Mitigation: explicit approval per file (D5).

## 12. Open items for the user

Answer D1-D9 in section 0. Then:
- Approve the test list in 8.1 and the new files in 8.3.
- Approve the manifest change (D6).
- Confirm the batch order: batch 4 stays one PR, or S1-S2 go first.
