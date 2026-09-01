# What’s next (after first live toggle)

Status: post-demo plan  
Companion: [Architecture](ARCHITECTURE.md), [MVP plan](MVP_PLAN.md)

## Baseline (do not regress)

On 2026-09-01, one Mac text command changed one real Home Assistant entity:

```text
python -m sayso_satellite "turn off the corner lamp"
→ completed / state_changed
→ switch.corner_lamp
```

Shipped on `ajax/planning`:

- `2f2f2d8` — ChatML few-shot, one compact candidate, LFM JSON-only parser
- `8ad86ca` — inferred `light` also retrieves `switch`; named targets retry on the whole graph; `action_request.domain` comes from the resolved entity

Still true:

- Checkpoint stays `mlx-community/LFM2.5-230M-OptiQ-4bit`. Do not change models.
- Parser stays strict. No speculative JSON repair. No unique-name bypass that skips the model.
- Prompt at most **one** candidate to the 230M (two or more makes it echo the user JSON).
- Safety barriers, HA permissions, and state verification stay on the live path.

There is **no** entity named “floor lamp”. Plug lamps in this home are switches (`switch.corner_lamp`, `switch.shelf_lamp`, `switch.sidetable_lamp`, `switch.livingroom_lampshade`). `turn off the floor lamp` currently asks which of Corner lamp / Shelf lamp.

`GET /api/v1/ready` still returns `model_ready: false` after MLX `load()`. Text already works; readiness does not.

---

## Order

1. Honest readiness + architecture snapshot
2. HA naming so “floor lamp” is unambiguous (config, not a model change)
3. Inherit device area onto snapshot entities (HACS bump)
4. Recorded-file voice through the same text path

Do not start wake/VAD, bake-offs, or a larger eval corpus until 1–4 are done or explicitly skipped.

---

## 1. `model_ready` and architecture doc

**Why.** Operators and the satellite treat 503 `/ready` as “not up.” The LFM is already loaded in `__main__` via `build_mlx_runtime_for_server()` → `runtime.load()`. `ReadinessState.set_model_ready` exists and is tested; nothing in the live entrypoint calls it. `docs/ARCHITECTURE.md` still says the first demo is blocked on `model_output_invalid` and cites uncommitted WIP at `d1f03dc`.

**Code**

- After constructing the app in `sayso-server/src/sayso_server/__main__.py` (or inside `create_aiohttp_app` when a loaded `MlxModelRuntime` is passed), call `readiness.set_model_ready(True)`.
- Keep `ha_connected` driven only by the HA WebSocket session.
- `ready` stays `model_ready and ha_connected`. Do not lie that HA is connected.

**Tests**

- `sayso-server/src/sayso_server/test_main.py` (or `test_app.py`): app built with a loaded fake/MLX runtime reports `model_ready: true`; without a loaded runtime it stays false.
- Existing `test_readiness.py` contract unchanged.

**Docs**

- Rewrite the header and “Remaining plan: first physical-device demo” in `docs/ARCHITECTURE.md` to match `8ad86ca`: demo succeeded; ChatML few-shot; switch-as-light; current-area miss then whole-graph name retry.
- Point snapshot at `8ad86ca`, not `d1f03dc` uncommitted WIP.

**Verify**

- Restart `python -m sayso_server`, wait for HA reconnect, `GET /api/v1/ready` with bearer token → `model_ready: true`, `ha_connected: true`, HTTP 200.
- One text command still toggles (corner lamp or equivalent).

**Not this task:** Whisper load is lazy on first `/api/v1/audio`. Do not block LFM `model_ready` on STT.

---

## 2. HA naming for “floor lamp”

**Why.** Retrieval and name-match already work for plug lamps. The utterance “floor lamp” overlaps the token `lamp` on more than one switch, so the orchestrator correctly asks for clarification. Few-shot still uses a fixture named Floor Lamp; that is fine once a live entity actually has that name or alias.

**Do this in Home Assistant (preferred)**

Pick the physical floor lamp (likely Corner lamp or Shelf lamp) and either:

- Rename the entity to **Floor Lamp**, or
- Add alias **floor lamp** on that entity only.

Reload SaySo (or wait for the next snapshot/delta). Then:

```text
python -m sayso_satellite "turn off the floor lamp"
```

Expect `completed` / `state_changed` on that one `switch.*`, not clarification.

**Do not** teach the 230M a second few-shot or raise `retrieve_candidates` `limit` above 1 to “fix” the name. If HA naming is refused, keep saying “corner lamp” / “shelf lamp.”

**Optional code (only if HA aliases never land):** add a satellite-local alias table. That is a new product surface; skip unless naming in HA is impossible.

**Verify**

- Named command hits one entity; the other lamp does not change.
- `turn off the floor lamp` with two equally named lamps still clarifies (ambiguity rule stays).

---

## 3. Device area on the Home Graph snapshot

**Why.** Exposure already treats an entity as in an area if the **device** is in that area (`custom_components/sayso/exposure.py`). Snapshot serialization only copies `entry.area_id` (`snapshot.py` `_serialize_registry_entity` / `_serialize_scene_or_script`). Live graph therefore shows `area_id=None` on almost every lamp. Current-area resolution would be empty without the whole-graph name fallback in `8ad86ca`. Fallback is a safety net, not room-correct targeting.

**Code (integration, needs HACS bump)**

- Effective area: `entry.area_id or device.area_id` (same rule as exposure).
- Apply to entities, scenes, and scripts that have a `device_id`.
- Entity-level area still wins when set.

**Tests**

- `custom_components/sayso/test_snapshot.py`: entity with `area_id=None` and device in `living_room` serializes `area_id: "living_room"`. Entity with its own area is unchanged. No device → omit `area_id`.

**Release**

- Bump `custom_components/sayso/manifest.json` (today `0.1.1`) and HACS metadata.
- Reload the integration so HA sends a new snapshot. Server-only restart is not enough.

**Verify**

- Live log or a one-off dump of graph lights/switches: corner/shelf/sidetable lamps show `living_room` (or their real HA area) instead of `None`.
- Mac satellite origin stays `living_room`. `turn off the corner lamp` still completes **from current-area match**, not only the whole-graph retry.
- Unassigned entities in other rooms do not all become living-room devices.

**Keep** the whole-graph named-target retry. It still helps when the user names a device in another room.

---

## 4. Recorded-file voice (same path as text)

**Why.** The first demo was text. Audio plumbing already exists; it is not wired as a supervised live check.

Already in tree:

| Piece | Where | Gap |
|---|---|---|
| `POST /api/v1/audio` | `audio_api.py` | Works if STT is loaded |
| `VoicePipelineController` | STT → same text controller | Whisper `load()` is on first clip, not process start |
| `MlxWhisperSttRuntime` | `mlx_stt.py`, default `mlx-community/whisper-small` | Not part of `/ready` |
| Satellite `--audio-file` | `sayso_satellite.__main__` | Default HTTP timeout **30s**; Whisper + LFM often needs ~180s like text |
| `read_pcm16_file` / `PushToTalkCapture` | `capture.py` | File path works; no CLI mic loop |

**Code**

- Raise satellite default timeout for audio (and document `timeout=` / env) so live clips do not die at 30s.
- Optional: `stt_runtime.load()` at server start **after** LFM is ready, or a separate `stt_ready` flag. Do not fold Whisper failure into LFM `model_ready`.
- Add one recorded 16 kHz mono PCM fixture (short “turn off the corner lamp”) under `evals/fixtures/` or `sayso-satellite/` and a test or runbook command that posts it.

**Verify**

```text
python -m sayso_satellite --audio-file <pcm>
```

Expect the same `completed` / `state_changed` as the text command (or a clear STT transcript mismatch, not a timeout).

**Not this task:** live microphone CLI, wake word, VAD, continuous listen (`ARCHITECTURE.md` tasks 58–64). `PushToTalkCapture` + `MicSource` stay library-only until file voice is green.

---

## Later (not in this file’s critical path)

- Wire `ConversationStore` into `create_live_text_controller` for “turn it off.”
- Operator runbook for the three processes (server `.env`, HA URL/token, satellite).
- Eval dry-run allowlist before pointing corpora at the live home (MVP task 48).
- Bake-offs and extra eval corpora.
- Orchestrator multi-entity fan-out (still first sorted id).

---

## Constraints checklist (every task)

- Same LFM checkpoint unless this file is explicitly superseded.
- No JSON repair; invalid model text stays `model_output_invalid`.
- One candidate in the LFM prompt.
- No native Cursor `Task` for product writes; Ajax Model Router + Composer for bounded code.
- Do not commit `.env` or `context.json`.
