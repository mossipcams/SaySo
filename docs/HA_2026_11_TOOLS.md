# Home Assistant 2026.11.1 Assist API tool list

Source: `home-assistant/core` at tag **2026.11.1** only. Sparse, shallow clone at `/tmp/ha-core-2026.11.1` (read-only, never executed). Paths below are relative to `homeassistant/`. Line numbers are from that tag.

Tool-list discovery: `gh api repos/home-assistant/core/git/trees/2026.11.1?recursive=1` lists 21 `components/*/llm.py` files. Those are the only places an integration can add tools to the Assist API (`components/llm/__init__.py:85-106` loops over the `llm` platform of every integration). All 21 were read or checked (§6).

Scope: this doc describes what HA 2026.11.1 *offers* to the model. It does not verify HA runtime behaviour (no HA instance was run).

## 0. Answers

- **The Assist API is not in `helpers/llm.py`.** It is `AssistAPI` in `components/llm/__init__.py:140-166`, registered at `:71`. `helpers/llm.py` only holds the base classes (`IntentTool` at `:364`, `APIInstance` at `:233`).
- **`HassMediaSearchAndPlay` is not offered as a tool in 2026.11.1.** Its voice handler still exists (`media_player/intent.py:130` registers `MediaSearchAndPlayHandler`, class at `:270-288`), but `media_player/llm.py:44-46` says: "HassMediaSearchAndPlay is not listed because the search media and play media tools cover it." The `LLM_INTENTS` dict at `:46-55` does not contain it.
- **Its replacements are two non-intent tools:** `media_player__search_media` (`media_player/llm.py:129-217`) and `media_player__play_media` (`:220-271`). They appear only if some *exposed* media player has both `SEARCH_MEDIA` and `PLAY_MEDIA` features (gate at `:304-309`, quoted in §2.B).
- **Todo tools:** `todo__get_items` (query) plus `todo__HassListAddItem`, `todo__HassListCompleteItem`, `todo__HassListRemoveItem`. All four appear only if at least one exposed todo entity has a name or alias (`todo/llm.py:124-133`).
- **Broadcast:** `assist_satellite__HassBroadcast`, parameter `message` (required string). The only list-time gate is that the `assist_satellite` intent platform was set up (`assist_satellite/llm.py:31-41`). There is **no** satellite or exposure check at list time.
- **Device/area dependence:** only timer tools depend on the requesting device, through `llm_context.device_id` (`intent/llm.py:91-97`). No tool list depends on area or floor. Area and floor only change the prompt text (`intent/llm.py:128-158`) and the match preferences that `IntentTool` injects at call time (`helpers/llm.py:403-421`).
- **Most feature gates are not checked when the list is built.** For example, `HassMediaPause` appears whenever any media player is exposed, even if no player supports pause. The `required_features` check only runs at call time, during matching. The media search and play tools are the exception: they check features at list time.
- **Production SaySo** takes the Assist API tool list unchanged (`conversation.py:345` compiles `chat_log.llm_api`; `conversation.py:363` makes it the active schema). Tool names are HA's own `IntentTool.name` / `Tool.name` values, not renamed. Full detail in §4.
- **The eval catalog is wrong in several places** (§5): it offers `HassMediaSearchAndPlay`, which HA does not, and it omits 14 fixed HA tools (plus dynamic script and intent-script tools). It also always offers the timer tools, names script tools without the `script__` prefix, and offers the timer tools even when the device is not timer-capable.
- **Caveat on the baseline:** `schemas/sayso-tool-schema-v2.json` is pinned to HA `2026.8.3` (`home_assistant_version`). Part of the diff below may be version drift rather than a bug. This doc does not check 2026.8.3.

## 1. How the list is built

1. `AssistAPI.async_get_api_instance` calls `async_get_tools(hass, llm_context, "assist")` (`components/llm/__init__.py:155-166`).
2. `async_get_tools` calls `platform.async_get_tools(hass, llm_context, api_id)` on every LLM platform, sorted by domain (`components/llm/__init__.py:94-103`). A platform returning `None` contributes nothing. Exceptions are logged and skipped (`:97-99`).
3. `intent.async_get(hass)` returns only handlers whose integration has been set up. The `intent` integration loads every integration's `intent.py` `async_setup_intents` (`components/intent/__init__.py:93-95`, `:627-630`). A handler registered by an integration that is not loaded is therefore not offered.
4. The tool name is `"<domain>__<intent_type>"` for `IntentTool`, or a hard-coded `"<domain>__<name>"` for custom tools (see §2.B for `media_player__search_media` and similar).

Exposure is `async_should_expose(hass, assistant, entity_id)` (`components/homeassistant/exposed_entities.py:247`). It uses the entity registry option `should_expose` for the assistant, and otherwise a default rule. The default rule excludes entities with `entity_category` or `hidden_by` (`:333-337`).

## 2. Tool inventory (Assist API, 2026.11.1)

Notation: "list-time" means the gate is checked when the tool list is built, which decides whether the model sees the tool. "call-time" means the gate is only checked when the tool runs, which can return an error.

Parameter types follow HA's declarations: `probatio` schema for `IntentTool` (generated from `slot_schema` / `ServiceIntentHandler.slot_schema`) and the explicit `parameters` for custom tools. Enum sources are noted.

### 2.A Generic intents (`intent` integration, `components/intent/llm.py`)

| Tool | Source | Appears when | Parameters |
|---|---|---|---|
| `intent__HassTurnOn` | `intent/__init__.py:97-111` (registration), `intent/llm.py:29-35` (listed), `:99-120` (built) | Always, list-time (`platforms` is `None`, `intent/llm.py:108`). No exposure needed. | `name`, `area`, `floor` (optional strings, `helpers/intent.py:1023-1026`). `domain`: array of string, no enum (`required_domains` is unset, `helpers/intent.py:1016-1021`). `device_class`: array of enum, 21 values (`intent/__init__.py:78-84`, `helpers/intent.py:1029-1042`). |
| `intent__HassTurnOff` | `intent/__init__.py:112-125` | Always (same as TurnOn). | Same as TurnOn. |
| `intent__HassCancelAllTimers` | `intent/llm.py:32` (listed), `intent/timers.py:984-990` | Always, list-time. | `area` (optional string). |
| `intent__HassSetPosition` | `intent/__init__.py:420-447`, listed at `intent/llm.py:33` | Only if an exposed entity's domain is in `{cover, valve}`: `if handler.platforms is None or handler.platforms & exposed_domains` (`intent/llm.py:108`). | `position` (required, integer 0-100 via `Coerce(int)` + `Percentage()`, `:426-430`). `name`, `area`, `floor`, `domain` (array of string, no enum), `device_class` (array of enum, cover and valve values). Constant `ATTR_POSITION = "position"` (`cover/const.py:21`). |
| `intent__HassStopMoving` | `intent/__init__.py:449-460`, listed at `intent/llm.py:34` | Same as SetPosition (cover or valve exposed). | `name`, `area`, `floor`, `domain`, `device_class` (no required slots). |
| `intent__HassStartTimer` | `intent/timers.py:851-859`, listed at `intent/llm.py:39` | Only if the requesting device is a timer device: `if supports_timers` (`intent/llm.py:96-97`). List-time. | `hours`, `minutes`, `seconds`: integers >= 0, at least one required (`Required(Any(...))`, `:857`). `name`, `conversation_command` (strings). |
| `intent__HassCancelTimer` | `intent/timers.py:961-969` | Timer device only. | `start_hours`, `start_minutes`, `start_seconds` (ints >= 0, one of). `name`, `area`. |
| `intent__HassIncreaseTimer` | `intent/timers.py:1015-1024` | Timer device only. | `hours`/`minutes`/`seconds` (ints, optional, `Any`). `start_*` (ints, optional). `name`, `area`. |
| `intent__HassDecreaseTimer` | `intent/timers.py:1040-1049` | Timer device only. | `hours`/`minutes`/`seconds`: one required (`Required(Any(...))`, `:1046`). `start_*`, `name`, `area`. |
| `intent__HassPauseTimer` | `intent/timers.py:1065-1073` | Timer device only. | `start_*` (ints, optional), `name`, `area`. |
| `intent__HassUnpauseTimer` | `intent/timers.py:1090-1098` | Timer device only. | `start_*`, `name`, `area`. |
| `intent__HassTimerStatus` | `intent/timers.py:1115-1123` | Timer device only. | `start_*`, `name`, `area`. |

Timer gate, exact source:

```
intent/llm.py:91-94  supports_timers = (llm_context.device_id is not None and async_device_supports_timers(hass, llm_context.device_id))
intent/timers.py:511-513  def is_timer_device(self, device_id): return device_id in self.handlers
```

A device becomes a timer device only through `async_register_timer_handler` (`intent/timers.py:526-534`). The only definitions and exports of it in the sparse clone are in `intent/__init__.py` and `intent/timers.py`. **UNVERIFIED:** which integration calls it, because those integrations are outside this clone. `custom_components/sayso` does not call it (grep: no match). So for SaySo, timer tools appear only if the satellite device was registered by something else.

Note that `intent/llm.py:121-126` returns the tool list even when no domain is exposed, so the always-on intents still appear for an empty home.

### 2.B Media player (`media_player/llm.py`)

Platform gate, exact source (`media_player/llm.py:282-291`):

```
if not llm_context.assistant: return None
exposed = [state for state in hass.states.async_all(DOMAIN) if async_should_expose(...)]
if not exposed: return None
```

If no media player is exposed, **every** `media_player__*` tool is absent.

Intents (`media_player/llm.py:293-303`, list `:46-55`). None of these check features at list time. The feature check is in the handler's `required_features` at call time.

| Tool | Source | Appears when | Parameters |
|---|---|---|---|
| `media_player__HassMediaNext` | `media_player/intent.py:76-89` | Any exposed media player (list-time). `required_features=NEXT_TRACK` and `required_states={PLAYING}` apply at call time. | `name`, `area`, `floor`, `domain` (array, enum `["media_player"]`, `required_domains`, `:86`), `device_class` (array, enum `tv`, `speaker`, `receiver`, `projector`, `const.py:230-237`). |
| `media_player__HassMediaPause` | `media_player/intent.py:133-149` | Same. Call-time: `PAUSE`, `PLAYING`. | Same as Next. |
| `media_player__HassMediaUnpause` | `media_player/intent.py:171-185` | Same. No feature gate, even at call time. | Same as Next. |
| `media_player__HassMediaPrevious` | `media_player/intent.py:90-103` | Same. Call-time: `PREVIOUS_TRACK`, `PLAYING`. | Same as Next. |
| `media_player__HassMediaPlayerMute` | `media_player/intent.py:233-257` | Same. Call-time: `VOLUME_MUTE`. | `name`, `area`, `floor`, `domain` (enum), `device_class` (enum), `is_volume_muted` (optional boolean, `:246-249`). The handler overwrites that slot with a constant (`:263-265`), so any model-supplied value is ignored. |
| `media_player__HassMediaPlayerUnmute` | same class, `False` | Same. | Same as Mute. |
| `media_player__HassSetVolume` | `media_player/intent.py:104-126` | Same. Call-time: `VOLUME_SET`. | `volume_level` (required, integer 0-100, `:113-120`; constant `ATTR_MEDIA_VOLUME_LEVEL = "volume_level"`, `const.py:58`). `name`, `area`, `floor`, `domain` (enum), `device_class` (enum). |
| `media_player__HassSetVolumeRelative` | `media_player/intent.py:401-424`, registered `:127` | Same. Call-time: `VOLUME_SET` (`:453`). | `volume_step` (required; `"up"`, `"down"`, or integer -100..100, `:408-416`). `name`, `area`, `floor`. No `domain` or `device_class`, because the class has no `required_domains` or `device_classes`. |
| `media_player__search_media` | `media_player/llm.py:129-162` (class), `:274-310` (gate) | Only if some exposed media player has both `SEARCH_MEDIA` and `PLAY_MEDIA` in `supported_features`. Gate quoted below. | `search_query` (required string, `:141-144`). `within_media_content_id` (string), `within_media_content_type` (string), `player_name`, `player_area`, `player_floor` (strings, `TARGET_SCHEMA` `:85-89`). |
| `media_player__play_media` | `media_player/llm.py:220-246` | Same gate as search. | `media_content_id` (required string), `media_content_type` (required string, `:236-242`). `player_name`, `player_area`, `player_floor`. |

Search and play gate, exact source (`media_player/llm.py:304-309`):

```
if any(
    state.attributes.get(ATTR_SUPPORTED_FEATURES, 0) & SEARCH_PLAY_FEATURES == SEARCH_PLAY_FEATURES
    for state in exposed
):
    tools.extend([MediaSearchTool(), MediaPlayTool()])
```

`SEARCH_PLAY_FEATURES = SEARCH_MEDIA | PLAY_MEDIA` (`media_player/llm.py:77-79`). Bits from `media_player/const.py`: `PLAY_MEDIA = 512` (`:214`), `SEARCH_MEDIA = 4194304` (`:227`). The gate reads the per-entity state attribute, so it depends on runtime state, not only the registry.

Call-time matching for search and play uses the same feature mask (`media_player/llm.py:74-76`, `:117`). `MediaSearchTool.async_call` returns at most 35 results (`MAX_SEARCH_RESULTS`, `:83`, `:200`).

### 2.C Todo (`todo/llm.py`, `todo/intent.py`)

Gate, exact source (`todo/llm.py:124-133`):

```
for state in sorted(hass.states.async_all(DOMAIN), key=attrgetter("name")):
    if not async_should_expose(hass, llm_context.assistant, state.entity_id): continue
    ...names.extend(intent.async_get_entity_aliases(...))
if not names: return None
```

Any exposed todo entity with at least one name or alias makes all four tools appear (list-time).

| Tool | Source | Parameters |
|---|---|---|
| `todo__get_items` | `todo/llm.py:46-113` (class), `:135` | `todo_list` (required, enum = the exposed list names and aliases at list time, `:66-67`). `status` (optional, enum `needs_action`, `completed`, `all`; default `needs_action`, `:70-76`). |
| `todo__HassListAddItem` | `todo/intent.py:13`, `:84-99`; registered `:20`; listed `todo/llm.py:30-34` | `item` (required non-empty string), `name` (required non-empty string, the list name, not an enum; `todo/intent.py:28-31`). |
| `todo__HassListCompleteItem` | `todo/intent.py:14`, `:102-133` | Same as Add. |
| `todo__HassListRemoveItem` | `todo/intent.py:15`, `:136-161` | Same as Add. |

### 2.D Broadcast (`assist_satellite/llm.py`, `assist_satellite/intent.py`)

| Tool | Source | Appears when | Parameters |
|---|---|---|---|
| `assist_satellite__HassBroadcast` | `assist_satellite/llm.py:31-41` (list), `assist_satellite/intent.py:15-30` (registration and schema), `helpers/intent.py:56` (`INTENT_BROADCAST = "HassBroadcast"`) | Whenever the `assist_satellite` intent platform has been set up. The code comment says "this platform is only queried once that has happened" (`:29`). No satellite and no exposure check at list time. | `message` (required, `probatio.Required("message"): str`, `:28-30`). |

Call-time behaviour, for a reviewer: `BroadcastIntentHandler.async_handle` (`:33-77`) selects every `assist_satellite` entity that has the `ANNOUNCE` feature, excluding the requesting device (`:51`) and `voip` config entries (`EXCLUDED_DOMAINS`, `:12`). It does **not** check exposure. So, unlike most tools, broadcast reaches satellites that are not exposed to the assistant. The `llm.py` comment ("only reaches the user's own satellites") does not match the handler. **Flagged as an observation to confirm.**

Broadcast availability does depend on the requesting device in one way: the handler excludes it from the targets (`intent_obj.device_id`, `:51`). That affects who hears the message, not whether the tool is offered.

### 2.E Other domain intents (each `components/<domain>/llm.py` + `intent.py`)

All of these share the same list-time gate pattern. If at least one exposed entity exists in the domain, the listed intents are offered (`if not any(async_should_expose(...) for state in hass.states.async_all(DOMAIN))`). The feature checks are call-time only.

| Tool | Source (list / handler) | Appears when | Parameters |
|---|---|---|---|
| `climate__HassClimateSetTemperature` | `climate/llm.py:27-55` / `climate/intent.py:34-50`; `TARGET_TEMPERATURE` at `:75` | Any exposed `climate` entity (`climate/llm.py:38-42`). | `temperature` (required, number, `Coerce(float)`, `:40`). `area`, `name`, `floor` (optional non-empty strings, `:41-43`). No `domain`. |
| `fan__HassFanSetSpeed` | `fan/llm.py:29-56` / `fan/intent.py:14-34` | Any exposed `fan` (`fan/llm.py:39-43`). | `percentage` (required, integer 0-100, `Coerce(int)` + `Percentage()`, `fan/intent.py:25-31`; constant `ATTR_PERCENTAGE = "percentage"`, `fan/const.py:26`). `name`, `area`, `floor`, `domain` (enum `["fan"]`). |
| `light__HassLightSet` | `light/llm.py:28-56` / `light/intent.py:18-50` | Any exposed `light` (`light/llm.py:39-43`). | All optional: `color` (string, `color_name_to_rgb`), `temperature` (integer >= 0, `cv.positive_int`), `brightness` (integer 0-100, `:35-44`). `name`, `area`, `floor`, `domain` (enum `["light"]`, `required_domains`, `:48`). |
| `lawn_mower__HassLawnMowerStartMowing` | `lawn_mower/llm.py:29-57` / `lawn_mower/intent.py:14-25` (constant names `INTENT_LANW_MOWER_*` are misspelled in source, `:9-10`) | Any exposed `lawn_mower` (`lawn_mower/llm.py:40-44`). Call-time: `START_MOWING = 1` (`:24`, value `lawn_mower/const.py:42`). | `name`, `area`, `floor`, `domain` (enum `["lawn_mower"]`). |
| `lawn_mower__HassLawnMowerDock` | `lawn_mower/intent.py:27-37` | Same. Call-time: `DOCK = 4` (`:36`). | Same. |
| `vacuum__HassVacuumStart` | `vacuum/llm.py:34-62` / `vacuum/intent.py:23-33` | Any exposed `vacuum` (`vacuum/llm.py:45-49`). Call-time: `START` (`:32`). | `name`, `area`, `floor`, `domain` (enum `["vacuum"]`). |
| `vacuum__HassVacuumReturnToBase` | `vacuum/intent.py:35-45` | Same. Call-time: `RETURN_HOME` (`:44`). | Same. |
| `vacuum__HassVacuumCleanArea` | `vacuum/intent.py:50-70` (schema), `:113` (call-time `CLEAN_AREA`) | Same. | `area` (required string), `name` (optional string). No `domain`. `preferred_*` are removed from the LLM schema. |
| `humidifier__HassHumidifierSetpoint` | `humidifier/llm.py:30-55` / `humidifier/intent.py:29-40` | Any exposed `humidifier` (`humidifier/llm.py:38-42`). | `name` (required non-empty string), `humidity` (required, integer 0-100). No `area`, no `domain`. |
| `humidifier__HassHumidifierMode` | `humidifier/intent.py:90-99` | Same. | `name` (required), `mode` (required string). |
| `calendar__get_events` | `calendar/llm.py:26-55` (class), `:99-116` (gate) | Any exposed calendar with at least one name or alias (`if not names: return None`, `:115`). | `calendar` (required, enum = exposed calendar names and aliases at list time, `:43-46`). `range` (required, enum `today`, `week`, `:47`). |
| `script__<object_id>` | `script/llm.py:14-45` (`ScriptTool`), `:48-63` (gate) | One tool per exposed `script` entity (`if async_should_expose(...)`, `:59`). Name is `f"{DOMAIN}__{script_name}"` (`:33`, `script_name` = object_id). | Dynamic. `ActionTool` takes the service description from the cache (`helpers/llm.py:796-820`). **UNVERIFIED:** the exact schema for each script. |
| `intent_script__<slug>` | `intent_script/llm.py:15-54` | One per configured intent script whose `platforms` intersect the exposed domains (`:29-38`). Slugified name (`:43-48`). | Dynamic, from user configuration. |

### 2.F Always-on, not device or area dependent

| Tool | Source | Appears when | Parameters |
|---|---|---|---|
| `llm__GetDateTime` | `llm/llm.py:19-25` (class), `:55-60` (`return LLMTools(tools=[GetDateTimeTool()])`) | Always (unconditional). | None. |
| `homeassistant__GetLiveContext` | `homeassistant/llm.py:228-236` (class), `:257-279` (parameters), `:636-663` (gate) | Always (unconditional, `:663`). If nothing is exposed, the call returns an error (`NO_ENTITIES_PROMPT`). | `name` (string), `domain` (string or array of string, no enum), `area` (string). All optional. |

### 2.G Not offered to the Assist API in 2026.11.1

These intent handlers exist in HA, but no LLM platform lists them, so they never reach an LLM:

- `HassMediaSearchAndPlay` (handler only, see §0).
- `HassGetState`, `HassToggle`, `HassNevermind`, `HassGetCurrentDate`, `HassGetCurrentTime`, `HassRespond`, `HassClimateGetTemperature`, `HassClimateSetFanMode` (`intent/__init__.py`, `climate/intent.py`). Checked against `intent/llm.py:29-46` and `climate/llm.py:21-24`.
- `HassOpenCover` and `HassCloseCover`, `HassGetWeather`, `HassShoppingList*` (`cover/intent.py`, `weather/intent.py`, `shopping_list/intent.py`). These domains have no `llm.py`.
- `HassHumidifier*` are offered (see §2.E). `HassSetPosition` and `HassStopMoving` are offered only for cover or valve (see §2.A).

The HA-API-only (admin) platforms `config`, `search`, `history`, `hassio`, `system_health`, `system_log`, and the `homeassistant` API's `RenderTemplate`, `SearchActions`, and `CallAction` (`homeassistant/llm.py:636-641`) are not part of the Assist API. They return `None` for `LLM_API_ASSIST` (verified per file).

## 3. Specific questions

**Does `HassMediaSearchAndPlay` exist in 2026.11.1?** Yes as a voice intent handler (`media_player/intent.py:130`, `:270-288`). No, as an LLM tool. Its tool name would be `media_player__HassMediaSearchAndPlay`, and it is not produced anywhere (`media_player/llm.py:44-55`).

**The separate search and play tools:**
- `media_player__search_media`. Params: `search_query` (required), `within_media_content_id`, `within_media_content_type`, `player_name`, `player_area`, `player_floor`.
- `media_player__play_media`. Params: `media_content_id` (required), `media_content_type` (required), `player_name`, `player_area`, `player_floor`.
- Gate: an exposed media player whose `supported_features` has both `SEARCH_MEDIA` and `PLAY_MEDIA` (`media_player/llm.py:304-309`).
- Each search result's `instruction` text tells the model to pass the same `player_*` targets to play (`media_player/llm.py:207-215`).

**Todo tools and gates:** see §2.C. Gate is "any exposed todo entity with a name or alias". There is no feature gate. `todo__get_items` has `status` as an enum.

**HassBroadcast name, parameters, gate:** `assist_satellite__HassBroadcast`, parameter `message` (required string). Gate is "the `assist_satellite` intent platform is set up" (`assist_satellite/llm.py:31-41`). No satellite, exposure, or `ANNOUNCE` check at list time. The `ANNOUNCE` check is at call time (`assist_satellite/intent.py:45-49`). See §2.D.

**Device- or area-dependent tools:**
- Device, list-time: timer tools only (`intent/llm.py:91-97`). Requires `llm_context.device_id` to be a registered timer device.
- Device, call-time: the requesting device is excluded from broadcast targets (`assist_satellite/intent.py:51`).
- Area or floor, list-time: none. Area only changes prompt text (`intent/llm.py:128-158`).
- Area or floor, call-time: `IntentTool` injects `preferred_area_id` and `preferred_floor_id` from the device's area (`helpers/llm.py:403-421`). These are removed from the model-facing schema (`helpers/llm.py:391-397`). They only break ties in matching; they do not add or remove a tool.

## 4. SaySo naming and pass-through

Source: `custom_components/sayso`.

- **Tool list.** `conversation.py:336-345`: `chat_log.async_provide_llm_data(...)`, then `turn.complete_schema = compile_llm_tools(chat_log.llm_api)`. `schema.py:366-380` compiles every `llm_api.tools` entry. `conversation.py:363`: `turn.active_schema = turn.complete_schema`. `routing.select_schema_for_domain` (`routing.py:291`) exists but is **not called** in `conversation.py`. So the model gets the full, unchanged list, which is what the task says.
- **Which API.** `CONF_LLM_HASS_API` defaults to `assist` (`__init__.py:303`, `const.py` default). The user can select another API in the config flow (`config_flow.py:243-247`).
- **Names.** SaySo does not add or change a prefix. The name is the HA `Tool.name`:
  - Intents: `intent__HassTurnOn`, `media_player__HassMediaPause`, `climate__HassClimateSetTemperature`, and so on (the namespace is the integration domain, from `IntentTool(f"{DOMAIN}__{intent_type}")`).
  - Custom tools: `media_player__search_media`, `todo__get_items`, `calendar__get_events`, `llm__GetDateTime`, `homeassistant__GetLiveContext`.
  - Scripts: `script__<object_id>`.
- **Validation.** `tool_schema.py:12` requires `^[a-zA-Z0-9_-]{1,64}$`. Every HA name above matches that pattern.
- **Name parsing.** `conversation.py:573` uses `call.name.rsplit("__", 1)[-1]` to get the bare intent name (for example `HassStartTimer`) and checks it against the intent constants.
- **Query tools.** `schema.py:120-131` marks `GetLiveContext`, `GetDateTime`, `TodoGetItemsTool`, and `HassTimerStatus` as query tools. The check uses the class name and module (`llm.llm`, `homeassistant.llm`, `todo.llm`).

Resulting SaySo names for the 2026.11.1 Assist API, per device: listed in §2. The `script__` prefix is what matters for SaySo's eval: see §5.

## 5. Diff: HA 2026.11.1 vs `schemas/sayso-tool-schema-v2.json` and `evals/runner.py`

The eval builds its list in `evals/runner.py:264-281` (`production_catalog`). It reads `sayso-tool-schema-v2.json` (`:134-139`), adds the namespace from `HA_TOOL_NAMESPACES` (`:79-107`, used by `namespaced_tool_name` at `:142-144` and `_compile_catalog` at `:222-234`), and then gates each tool on `HA_TOOL_NAMESPACES[name] in _ALWAYS_OFFERED_NAMESPACES | domains` (`:109`, `:270-273`). `_ALWAYS_OFFERED_NAMESPACES = {llm, homeassistant, intent}`.

### 5.1 Tools in HA 2026.11.1 but missing from the eval

| HA tool | Gate in HA | Why eval lacks it |
|---|---|---|
| `media_player__search_media` | An exposed media player has SEARCH_MEDIA and PLAY_MEDIA | Not in v2 schema; not a `HassIntent`, so not in `HA_TOOL_NAMESPACES`. |
| `media_player__play_media` | Same | Same. |
| `todo__get_items` | An exposed todo entity | Not in v2 schema. |
| `todo__HassListAddItem` | Same | Not in v2 schema. |
| `todo__HassListCompleteItem` | Same | Not in v2 schema. |
| `todo__HassListRemoveItem` | Same | Not in v2 schema. |
| `assist_satellite__HassBroadcast` | assist_satellite intents loaded | Not in v2 schema. |
| `intent__HassSetPosition` | An exposed cover or valve | Not in v2 schema. |
| `intent__HassStopMoving` | Same | Not in v2 schema. |
| `lawn_mower__HassLawnMowerStartMowing` | An exposed lawn_mower | Not in v2 schema. |
| `lawn_mower__HassLawnMowerDock` | Same | Not in v2 schema. |
| `humidifier__HassHumidifierSetpoint` | An exposed humidifier | Not in v2 schema. |
| `humidifier__HassHumidifierMode` | Same | Not in v2 schema. |
| `calendar__get_events` | An exposed calendar | Not in v2 schema. |
| `script__<object_id>` (dynamic) | An exposed script | **Present but misnamed in the eval**, see §5.4. |
| `intent_script__*` (dynamic) | Configured intent script | Not modelled. |

### 5.2 Tools in the eval but not in HA 2026.11.1

- **`media_player__HassMediaSearchAndPlay`.** It is in `schemas/sayso-tool-schema-v2.json` (`HassMediaSearchAndPlay`) and `evals/runner.py:94` maps it to `media_player`. HA 2026.11.1 does not offer it. **This is the largest single mismatch.** It also means the eval teaches the model a tool that does not exist in production, while the real tools `media_player__search_media` and `media_player__play_media` are absent from the eval.

### 5.3 Gating differences (tool in both, different condition)

| Tool | Eval gate | HA gate | Effect |
|---|---|---|---|
| `intent__HassStartTimer`, `HassCancelTimer`, `HassIncreaseTimer`, `HassDecreaseTimer`, `HassPauseTimer`, `HassUnpauseTimer`, `HassTimerStatus` | Always (`intent` is in `_ALWAYS_OFFERED_NAMESPACES`, `runner.py:109`) | Only when `device_id` is a registered timer device (`intent/llm.py:91-97`) | The eval always shows timer tools. Production shows them only for timer devices. |
| `intent__HassSetPosition`, `intent__HassStopMoving` | Not in v2 | Only with an exposed cover or valve | Missing from eval entirely (§5.1). |
| `intent__HassCancelAllTimers` | Always | Always | Same. |
| `media_player__*` (other) | Domain exposed | Any exposed media player | Same. |
| `climate__`, `fan__`, `light__`, `vacuum__` | Domain exposed | Any exposed entity of that domain | Same, but HA does not check the feature bit at list time, and neither does the eval. Same. |
| `script__<object_id>` | All home script entities (`_script_tools`, `runner.py:242-262`) | Exposed scripts only (`script/llm.py:59`) | **UNVERIFIED** whether the eval's home dump is already filtered to exposed scripts. |

Feature-gate note: the eval has no feature gates at all (the v2 schema has none), which matches HA at list time for all tools except the media search and play tools.

### 5.4 Naming mismatches

- **Script tools.** HA names them `script__<object_id>` (`script/llm.py:33`). The eval names them `<object_id>` with no prefix (`runner.py:255`, `_script_tool_name` at `:237-239`). The eval also prefixes a digit-leading object_id with `_` (`:238-239`); HA does not. SaySo's name regex accepts digit-leading names (`tool_schema.py:12`), so the prefix is not needed for validity.
- **Other names.** The eval's namespace table gives the same names as HA for every v2 intent. The `intent__`, `media_player__`, `climate__`, `fan__`, `light__`, `vacuum__`, `llm__`, `homeassistant__` prefixes all match HA.

### 5.5 Parameter differences (tools in both)

Checked against the v2 `parameters` block.

| Tool | Difference | HA 2026.11.1 | v2 schema |
|---|---|---|---|
| `media_player__HassMediaPlayerMute`, `media_player__HassMediaPlayerUnmute` | `is_volume_muted` type | Boolean, optional (`media_player/intent.py:246-249`), and the handler ignores any model value (`:263-265`) | `string` |
| `light__HassLightSet` | `domain` enum | `["light"]` (`light/intent.py:48` with `required_domains`) | array, no enum |
| `intent__HassTurnOn`, `intent__HassTurnOff` | `device_class` | 21-value enum (`intent/__init__.py:78-84`) | Matches (21 values) |
| `media_player__HassSetVolumeRelative` | `volume_step` | `"up"`, `"down"`, or -100..100 integer | `anyOf` (matches) |
| `intent__HassStartTimer`, `HassDecreaseTimer` | Requirement | At least one of hours, minutes, seconds (`Required(Any(...))`) | `anyOf` required (matches) |
| `intent__HassIncreaseTimer` | Requirement | None required (`Any`, optional) | None required (matches) |
| `homeassistant__GetLiveContext` | `domain` | string or list of strings, no enum | string or array (matches) |
| `media_player__HassSetVolume` | `volume_level` | integer 0-100 (`:113-120`) | integer 0-100 (matches) |
| `light__HassLightSet` | `brightness` | integer 0-100 | integer 0-100 (matches) |
| `intent__HassClimateSetTemperature` | `temperature` | number (`Coerce(float)`) | number (matches) |

Everything else in the v2 schema matches HA 2026.11.1 at the level of names, required fields, and enums that I checked. The main parameter differences are `is_volume_muted` (type) and the missing `domain` enum on `HassLightSet`.

### 5.6 Summary of the diff

- HA has 14 fixed tools the eval does not offer, plus dynamic script and intent-script tools (§5.1), including the entire todo, broadcast, humidifier, calendar, lawn-mower, and cover/valve position families, and the media search and play tools.
- The eval offers one tool HA does not (`HassMediaSearchAndPlay`, §5.2).
- The eval always offers the 7 timer intents. HA offers them only for timer devices (§5.3).
- Script tools are named differently (§5.4).
- Parameter differences are small (§5.5).
- Caveat: the v2 snapshot is HA 2026.8.3 (its `home_assistant_version`), not 2026.11.1. Some of the diff may be version drift. This doc does not verify 2026.8.3.

## 6. Files consulted

Read from the sparse clone at tag 2026.11.1:

- `homeassistant/helpers/llm.py`: `IntentTool` (`:364-434`), `async_get_match_preferences` (`:344-362`), `APIInstance` (`:233-289`), `ActionTool` (`:796-820`).
- `homeassistant/helpers/intent.py`: `IntentHandler` and `slot_schema` (`:884-916`), `DynamicServiceIntentHandler.slot_schema` (`:1016-1079`), `ServiceIntentHandler` (`:1284`), constants (`:40-66`).
- `components/llm/__init__.py` (`AssistAPI`, `async_get_tools`), `components/llm/llm.py` (`GetDateTime`).
- `components/intent/__init__.py` (on/off/position/stop/timer registration, `ONOFF_DEVICE_CLASSES`), `intent/llm.py`, `intent/timers.py`.
- `components/homeassistant/llm.py` (`GetLiveContext`, `async_get_tools`), `homeassistant/exposed_entities.py`.
- `components/media_player/llm.py`, `intent.py`, `const.py`.
- `components/todo/llm.py`, `intent.py`, `const.py`.
- `components/assist_satellite/llm.py`, `intent.py`, `const.py`.
- `components/climate/llm.py`, `intent.py`, `const.py`.
- `components/fan/llm.py`, `intent.py`, `const.py`.
- `components/light/llm.py`, `intent.py`.
- `components/lawn_mower/llm.py`, `intent.py`, `const.py`.
- `components/vacuum/llm.py`, `intent.py`, `const.py`.
- `components/humidifier/llm.py`, `intent.py`.
- `components/calendar/llm.py`.
- `components/script/llm.py`.
- `components/intent_script/llm.py`.
- `components/cover/const.py`, `valve/const.py`, `button/const.py`, `switch/const.py` (device-class enums, `ATTR_POSITION`).
- The other 6 `llm.py` files (`config`, `hassio`, `history`, `search`, `system_health`, `system_log`) were checked. They return `None` for `LLM_API_ASSIST`.

Enumeration: `gh api repos/home-assistant/core/git/trees/2026.11.1?recursive=1` (34,565 paths). The `llm.py` list is the 21 files named in the header. No other `llm.py` exists at this tag.

Not verified, flagged:
- **UNVERIFIED:** which integrations call `async_register_timer_handler` (outside this clone).
- **UNVERIFIED:** whether `broadcast` should respect exposure. The code (§2.D) does not. This is an observation, not a claim about intent.
- **UNVERIFIED:** script tool parameter schemas. They depend on the service description at runtime.
- **UNVERIFIED:** exact `ServiceIntentHandler` domain enum for `SetPosition` and `StopMoving`. I derived "array of string, no enum" from `helpers/intent.py:1016-1021`, because they set no `required_domains`. I did not run the code.
- **UNVERIFIED:** whether `intent_script` and the `script` tool's exposure filter are applied before the eval's home dump is built.
- The v2 schema is pinned to HA 2026.8.3. Differences may be version drift.
- Not run: no HA instance, no SaySo eval, no tests. This is a source-reading result.

## 7. Verification

- `test -s docs/HA_2026_11_TOOLS.md` (see the report).
- `grep -c '2026.11.1' docs/HA_2026_11_TOOLS.md` (see the report).
- No other files in the repository were changed. No commit.
