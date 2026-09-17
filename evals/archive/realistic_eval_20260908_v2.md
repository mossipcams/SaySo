# Realistic eval: LFM2.5-230M-Base baseline

**Base produced 0/100 exact action matches. It abstained from attempting a tool call in 14/20 no-call cases.** All 120 inference requests completed without errors.

The existing `repoparse` scorer reports **19/120 (15.8%)**, entirely from no-call cases. Its parser only accepts single-quoted string arguments. It therefore hides five double-quoted tool attempts in no-call cases and incorrectly credits those cases as abstentions. A separate audit of the same saved outputs using Python AST parsing and literal arguments gives **14/120 (11.7%)**, with unchanged expected labels. No model rerun or case selection was used for that diagnostic. Neither metric evaluates the usefulness or correctness of clarification prose.

| Category | Existing parser | Literal-syntax diagnostic | Cases |
|---|---:|---:|---:|
| ordinary | 0 | 0 | 10 |
| light_fan_settings | 0 | 0 | 10 |
| climate | 0 | 0 | 10 |
| media | 0 | 0 | 10 |
| timers | 0 | 0 | 10 |
| routines_vacuum | 0 | 0 | 10 |
| status | 0 | 0 | 10 |
| aliases | 0 | 0 | 10 |
| multi_action | 0 | 0 | 10 |
| exclusion | 0 | 0 | 10 |
| ambiguity | 10 | 8 | 10 |
| unavailable | 9 | 6 | 10 |

Among the 100 action cases, the literal-syntax diagnostic found 70 missing calls, 21 wrong tool names, and 9 correct tool-name sets with incorrect arguments. No action case became correct merely by accepting double quotes. Six of the 20 no-call cases attempted a call, including an invented `turn_on_garage_heater()` for an absent device. The remaining abstentions often incorrectly claimed that tools were unavailable; they are not verified helpful clarifications.

## Frozen evaluation set

- 120 agent-authored synthetic cases, 12 categories with 10 cases each, across five independently specified households with 19 devices each. This is a balanced diagnostic suite, not a frequency-weighted traffic sample or recorded human speech.
- 46 distinct device target names: 12 present in training context names/aliases, 34 absent. Ordinary words and room/fixture relationships are retained rather than inventing unusual names merely for uniqueness.
- Zero normalized prompt overlap with the actual Run 010 training file or existing locked evals; zero identical training contexts. Familiar-name reuse is deliberate. Semantic concepts overlap with training because generalization on those tasks is the purpose.
- 100 call cases and 20 no-call cases. Each group independently has equal uppercase/lowercase starts. Ten explicit alias cases and ten exclusion cases; all 20 multi-call cases require two calls.
- Every expected call validates against the tools offered for its case. Ambiguity cases offer the requested action; absent-device cases offer the relevant action; unavailable-capability cases explicitly withhold that tool. Aliases and exclusions were checked against the authored household. Wrong/missing-call mutations fail scoring.
- All turns carry `train_on_turn: false`. This eval is separate from training and the existing locked recipe/gold/shadow files. Keep this revision fixed for checkpoint comparison.

Eval SHA256: `35e6be66fd42e73cad015836e09f584209e6f395846a6476338acae14758ed43`.

Training render SHA256: `e088b3a6066f7fba5ff9a1f073acf4b3b92335f0f3f1f8f895abf6b43ba8ff15`.

Revision 2 corrects tool availability in the first draft; no expected labels were changed after inference began, and no model answers were inspected before the revision was fixed. Partial earlier runs were discarded.

## Inference and scoring

- Model: original `LFM2.5-230M-Base`, Q8_0 GGUF; no trained adapter. GGUF SHA256 `0983bae98dc7eefade32ce9be7e24d7257be452637f6ee9720ad1bf9e2b0a4f2`.
- Existing `training/scripts/eval_v3_rawparse.py`, source snapshot `52b89a4` (the scorer is unchanged from this PR’s base), same exact tool/argument labels for both reports. The AST diagnostic only reads literals; it never executes model output.
- Temperature 0, maximum 256 output tokens, context 8192, isolated localhost CPU llama.cpp server with two threads. Prompts span 1,610–2,325 tokens; no input truncation.
- Server flag `--override-kv tokenizer.ggml.add_bos_token=bool:false` prevents adding BOS twice to the HF-rendered prompt. Server token IDs matched the HF tokenizer on all 120 cases. Use the same flag for subsequent checkpoint comparisons. Earlier locked-suite baselines used different serving settings and different cases, so their percentages are not directly comparable.
- Raw inference outputs, parser errors, per-case expected labels, and the separate diagnostic are retained. The isolated server was stopped after the run; Run 010 training continues.

## Reproduce the frozen JSONL

From the repository root, expand the shared contexts and tool schemas. This
reconstructs the exact evaluated bytes; the hash assertion protects the snapshot.
The expanded dataset remains gitignored.

```sh
python3 - <<'PY_RENDER'
import hashlib
import json
from pathlib import Path

fixture = json.loads(Path("training/fixtures/realistic_eval_20260908_v2.json").read_text())
rows = [
    {
        "messages": [fixture["contexts"][case["context"]], *case["messages"]],
        "tools": [fixture["tools"][name] for name in case["tool_names"]],
        "metadata": case["metadata"],
    }
    for case in fixture["cases"]
]
text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
assert len(rows) == 120
assert hashlib.sha256(text.encode()).hexdigest() == fixture["render_sha256"]
out = Path("training/datasets/sayso_quality_eval_realistic_20260908_v2.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(text)
print(out)
PY_RENDER
```

On the training host, start a separate Base server in one terminal:

```sh
/opt/llama.cpp/build/bin/llama-server \
  --host 127.0.0.1 --port 18081 \
  --model /srv/models/LFM2.5-230M-Base-Q8_0.gguf \
  --ctx-size 8192 --n-gpu-layers 0 --threads 2 --parallel 1 --jinja \
  --override-kv tokenizer.ggml.add_bos_token=bool:false
```

Run the existing scorer in another terminal after the server is ready:

```sh
/srv/training-runs/.venv/bin/python training/scripts/eval_v3_rawparse.py \
  --server http://127.0.0.1:18081 \
  --tokenizer /srv/models/LFM2.5-230M-Base \
  --eval-set training/datasets/sayso_quality_eval_realistic_20260908_v2.jsonl \
  --out training/artifacts/base-realistic-120-v2.json --timeout 120
```

This reproduces the **existing-parser** metric. The 14/120 result above is a
separate diagnostic of saved outputs, not a change to the production scorer.
The original report is pinned by its VM path below. The parser limitation remains
unfixed in this data-only change.

## Artifacts retained on the VM

Bundle: `/srv/training-runs/sayso-realistic-20260908`.

- Frozen JSONL: `evals-pinned/sayso_quality_eval_realistic_20260908_v2.jsonl`
- Raw generations and scores: `results/base-realistic-120-v2.json`
- Separate literal-syntax diagnostic: `results/base-realistic-120-v2-literal-diagnostic.json`
- Model, scorer and template hashes: `results/base-realistic-120-v2-provenance.json`
- Full tokenization check: `evals-pinned/realistic-eval-v2-token-audit.json`

The fixture below is the reviewable source snapshot: shared system contexts,
shared offered-tool schemas, and all 120 cases with their expected messages.
Raw run outputs and the expanded JSONL are not committed. This new eval has not
yet been run against a trained Run 010 checkpoint. It is not yet included in the
training generator's automatic exclusion list or the host's three-suite epoch
launcher: retain the explicit overlap check before any future training rebuild.

## Review the cases

| Case | Request | Expected calls / abstention |
|---|---|---|
| aliases_01 | Could you turn the prep lights on for me? | HassTurnOn(name='Kitchen Counter Lights', domain=['light']) |
| aliases_02 | switch the sofa lamp off, please. | HassTurnOff(name='Living Room Ceiling Light', domain=['light']) |
| aliases_03 | I'd like my bedside light on now. | HassTurnOn(name="Morgan's Bedside Lamp", domain=['light']) |
| aliases_04 | turn the big screen off for tonight. | HassTurnOff(name='Sofa TV', domain=['media_player']) |
| aliases_05 | Please switch the coffee plug on now. | HassTurnOn(name='Coffee Machine Outlet', domain=['switch']) |
| aliases_06 | i'm done cooking; switch the prep lights off. | HassTurnOff(name='Kitchen Counter Lights', domain=['light']) |
| aliases_07 | Get the sofa lamp switched on for me. | HassTurnOn(name='Living Room Ceiling Light', domain=['light']) |
| aliases_08 | could you switch my bedside light off now? | HassTurnOff(name="Morgan's Bedside Lamp", domain=['light']) |
| aliases_09 | Please put the big screen on. | HassTurnOn(name='Sofa TV', domain=['media_player']) |
| aliases_10 | i'm finished; turn the coffee plug off. | HassTurnOff(name='Coffee Machine Outlet', domain=['switch']) |
| ambiguity_01 | Could you turn on the reading light in the living room? | clarify |
| ambiguity_02 | switch the bedside lamp off in the bedroom, please. | clarify |
| ambiguity_03 | Dim the reading light in the living room to 30 percent. | clarify |
| ambiguity_04 | turn the bedroom bedside lamp blue for me. | clarify |
| ambiguity_05 | Please set the reading light in the living room to 25 percent. | clarify |
| ambiguity_06 | please switch off the reading light in the living room. | clarify |
| ambiguity_07 | Make the bedside lamp in the bedroom a little brighter. | clarify |
| ambiguity_08 | set the living room reading light to warm white. | clarify |
| ambiguity_09 | Can you turn on the bedside lamp in the bedroom? | clarify |
| ambiguity_10 | run the routine for me, please. | clarify |
| climate_01 | I'd like Hallway Thermostat set to 67 degrees. | HassClimateSetTemperature(name='Hallway Thermostat', temperature=67) |
| climate_02 | bring Downstairs Thermostat to 70 degrees, please. | HassClimateSetTemperature(name='Downstairs Thermostat', temperature=70) |
| climate_03 | Can you adjust Main Thermostat to 69 degrees? | HassClimateSetTemperature(name='Main Thermostat', temperature=69) |
| climate_04 | for Hallway Heating, use 72 degrees. | HassClimateSetTemperature(name='Hallway Heating', temperature=72) |
| climate_05 | Put Central Heating at 68 degrees for me. | HassClimateSetTemperature(name='Central Heating', temperature=68) |
| climate_06 | please change the setpoint on Hallway Thermostat to 71 degrees. | HassClimateSetTemperature(name='Hallway Thermostat', temperature=71) |
| climate_07 | Make it 66 degrees on Downstairs Thermostat. | HassClimateSetTemperature(name='Downstairs Thermostat', temperature=66) |
| climate_08 | set Main Thermostat to 73 degrees now, please. | HassClimateSetTemperature(name='Main Thermostat', temperature=73) |
| climate_09 | Could you lower Hallway Heating to 65 degrees? | HassClimateSetTemperature(name='Hallway Heating', temperature=65) |
| climate_10 | i'd prefer 74 degrees on Central Heating. | HassClimateSetTemperature(name='Central Heating', temperature=74) |
| exclusion_01 | Switch off the kitchen lights, but leave Kitchen Ceiling Light alone. | HassTurnOff(name='Kitchen Counter Lights', domain=['light']); HassTurnOff(name='Kitchen Sink Light', domain=['light']) |
| exclusion_02 | turn on all kitchen lights and leave Kitchen Pantry Light as it is. | HassTurnOn(name='Breakfast Bar Pendants', domain=['light']); HassTurnOn(name='Kitchen Worktop Lights', domain=['light']) |
| exclusion_03 | Dim the kitchen lights to 20 percent, but leave Kitchen Under Cabinet Lights unchanged. | HassLightSet(name='Kitchen Ceiling Spots', domain=['light'], brightness=20); HassLightSet(name='Kitchen Cooker Light', domain=['light'], brightness=20) |
| exclusion_04 | please turn the kitchen lights off and leave Kitchen Breakfast Light on. | HassTurnOff(name='Kitchen Island Lights', domain=['light']); HassTurnOff(name='Kitchen Cabinet Strip', domain=['light']) |
| exclusion_05 | Make the kitchen lights blue, but leave Kitchen Pendant Lights alone. | HassLightSet(name='Kitchen Sink Light', domain=['light'], color='blue'); HassLightSet(name='Kitchen Ceiling Light', domain=['light'], color='blue') |
| exclusion_06 | leave Kitchen Ceiling Light alone and switch the other kitchen lights off. | HassTurnOff(name='Kitchen Counter Lights', domain=['light']); HassTurnOff(name='Kitchen Sink Light', domain=['light']) |
| exclusion_07 | Turn the kitchen lights on; leave Kitchen Pantry Light unchanged. | HassTurnOn(name='Breakfast Bar Pendants', domain=['light']); HassTurnOn(name='Kitchen Worktop Lights', domain=['light']) |
| exclusion_08 | set the kitchen lights to 45 percent brightness and leave Kitchen Under Cabinet Lights as it is. | HassLightSet(name='Kitchen Ceiling Spots', domain=['light'], brightness=45); HassLightSet(name='Kitchen Cooker Light', domain=['light'], brightness=45) |
| exclusion_09 | Leave Kitchen Breakfast Light on, but switch off every other kitchen light. | HassTurnOff(name='Kitchen Island Lights', domain=['light']); HassTurnOff(name='Kitchen Cabinet Strip', domain=['light']) |
| exclusion_10 | change the kitchen lights to warm white and leave Kitchen Pendant Lights alone. | HassLightSet(name='Kitchen Sink Light', domain=['light'], color='warm white'); HassLightSet(name='Kitchen Ceiling Light', domain=['light'], color='warm white') |
| light_fan_settings_01 | Bring Kitchen Counter Lights down to 35 percent brightness. | HassLightSet(name='Kitchen Counter Lights', domain=['light'], brightness=35) |
| light_fan_settings_02 | i'd like Sofa Reading Lamp at 60 percent brightness. | HassLightSet(name='Sofa Reading Lamp', domain=['light'], brightness=60) |
| light_fan_settings_03 | Dim Morgan's Bedside Lamp to fifteen percent, please. | HassLightSet(name="Morgan's Bedside Lamp", domain=['light'], brightness=15) |
| light_fan_settings_04 | make the color of Living Room Reading Lamp blue, please. | HassLightSet(name='Living Room Reading Lamp', domain=['light'], color='blue') |
| light_fan_settings_05 | Change Kitchen Ceiling Light to warm white for me. | HassLightSet(name='Kitchen Ceiling Light', domain=['light'], color='warm white') |
| light_fan_settings_06 | set the white temperature of Chris's Bedside Lamp to 2700 kelvin. | HassLightSet(name="Chris's Bedside Lamp", domain=['light'], temperature=2700) |
| light_fan_settings_07 | I'd like Sofa Reading Lamp at 4000 kelvin, please. | HassLightSet(name='Sofa Reading Lamp', domain=['light'], temperature=4000) |
| light_fan_settings_08 | run Bedroom Tower Fan at a quarter of its full speed. | HassFanSetSpeed(name='Bedroom Tower Fan', domain=['fan'], percentage=25) |
| light_fan_settings_09 | Set Bedroom Fan to sixty five percent speed. | HassFanSetSpeed(name='Bedroom Fan', domain=['fan'], percentage=65) |
| light_fan_settings_10 | could you bring Bedroom Standing Fan down to 40 percent speed? | HassFanSetSpeed(name='Bedroom Standing Fan', domain=['fan'], percentage=40) |
| media_01 | Resume playback on Living Room TV, please. | HassMediaUnpause(name='Living Room TV', domain=['media_player'], device_class=['tv']) |
| media_02 | can you pause what's playing on Lounge Television? | HassMediaPause(name='Lounge Television', domain=['media_player'], device_class=['tv']) |
| media_03 | Bring Living Room Television volume to 30 percent. | HassSetVolume(name='Living Room Television', domain=['media_player'], device_class=['tv'], volume_level=30) |
| media_04 | mute the sound on Sofa TV, please. | HassMediaPlayerMute(name='Sofa TV', domain=['media_player'], device_class=['tv']) |
| media_05 | Turn the volume down on Family TV a bit. | HassSetVolumeRelative(name='Family TV', volume_step='down') |
| media_06 | put Living Room TV back on play. | HassMediaUnpause(name='Living Room TV', domain=['media_player'], device_class=['tv']) |
| media_07 | Please pause playback on Lounge Television for a moment. | HassMediaPause(name='Lounge Television', domain=['media_player'], device_class=['tv']) |
| media_08 | set the sound level of Living Room Television to 55 percent. | HassSetVolume(name='Living Room Television', domain=['media_player'], device_class=['tv'], volume_level=55) |
| media_09 | Could you silence Sofa TV? | HassMediaPlayerMute(name='Sofa TV', domain=['media_player'], device_class=['tv']) |
| media_10 | i'd like Family TV a little louder. | HassSetVolumeRelative(name='Family TV', volume_step='up') |
| multi_action_01 | Switch Kitchen Counter Lights on and Coffee Maker Plug off for me. | HassTurnOn(name='Kitchen Counter Lights', domain=['light']); HassTurnOff(name='Coffee Maker Plug', domain=['switch']) |
| multi_action_02 | turn on Sofa Reading Lamp, then turn off Lounge Television. | HassTurnOn(name='Sofa Reading Lamp', domain=['light']); HassTurnOff(name='Lounge Television', domain=['media_player']) |
| multi_action_03 | I'd like Morgan's Bedside Lamp on and Bedroom Tower Fan off. | HassTurnOn(name="Morgan's Bedside Lamp", domain=['light']); HassTurnOff(name='Bedroom Tower Fan', domain=['fan']) |
| multi_action_04 | lock Main Door Lock and switch Kitchen Cabinet Strip off. | HassTurnOn(name='Main Door Lock'); HassTurnOff(name='Kitchen Cabinet Strip', domain=['light']) |
| multi_action_05 | Could you turn on Bathroom Shower Light and turn off Living Room TV Backlight? | HassTurnOn(name='Bathroom Shower Light', domain=['light']); HassTurnOff(name='Living Room TV Backlight', domain=['light']) |
| multi_action_06 | turn Kitchen Sink Light on, and also switch Kitchen Counter Lights off. | HassTurnOn(name='Kitchen Sink Light', domain=['light']); HassTurnOff(name='Kitchen Counter Lights', domain=['light']) |
| multi_action_07 | Please switch Kitchen Kettle Plug on; turn Robin's Bedside Lamp off too. | HassTurnOn(name='Kitchen Kettle Plug', domain=['switch']); HassTurnOff(name="Robin's Bedside Lamp", domain=['light']) |
| multi_action_08 | open Bedroom Roller Blind and stop Bedroom Tower Fan. | HassTurnOn(name='Bedroom Roller Blind'); HassTurnOff(name='Bedroom Tower Fan', domain=['fan']) |
| multi_action_09 | Put Living Room Reading Lamp on and switch Sofa TV off, please. | HassTurnOn(name='Living Room Reading Lamp', domain=['light']); HassTurnOff(name='Sofa TV', domain=['media_player']) |
| multi_action_10 | make sure Front Entrance Lock is locked and turn Taylor's Bedside Lamp off. | HassTurnOn(name='Front Entrance Lock'); HassTurnOff(name="Taylor's Bedside Lamp", domain=['light']) |
| ordinary_01 | Can you switch the Kitchen Counter Lights on for cooking? | HassTurnOn(name='Kitchen Counter Lights', domain=['light']) |
| ordinary_02 | please switch off Kitchen Kettle Plug now. | HassTurnOff(name='Kitchen Kettle Plug', domain=['switch']) |
| ordinary_03 | I'd like Living Room Floor Lamp on, please. | HassTurnOn(name='Living Room Floor Lamp', domain=['light']) |
| ordinary_04 | switch Sam's Bedside Lamp off for me. | HassTurnOff(name="Sam's Bedside Lamp", domain=['light']) |
| ordinary_05 | Please lock Front Entrance Lock before I leave. | HassTurnOn(name='Front Entrance Lock') |
| ordinary_06 | unlock Front Door Lock for me, please. | HassTurnOff(name='Front Door Lock') |
| ordinary_07 | Could you open Bedroom Window Shade all the way? | HassTurnOn(name='Bedroom Window Shade') |
| ordinary_08 | close Bedroom Roller Blind for the night. | HassTurnOff(name='Bedroom Roller Blind') |
| ordinary_09 | Please get Bedroom Fan running. | HassTurnOn(name='Bedroom Fan', domain=['fan']) |
| ordinary_10 | switch off Bathroom Shower Light when you can. | HassTurnOff(name='Bathroom Shower Light', domain=['light']) |
| routines_vacuum_01 | Have Downstairs Robot Vacuum start cleaning, please. | HassVacuumStart(name='Downstairs Robot Vacuum', domain=['vacuum']) |
| routines_vacuum_02 | send Kitchen Robot Vacuum back to its dock. | HassVacuumReturnToBase(name='Kitchen Robot Vacuum', domain=['vacuum']) |
| routines_vacuum_03 | Use Roomba to vacuum the kitchen. | HassVacuumCleanArea(name='Roomba', area='Kitchen') |
| routines_vacuum_04 | put on the Relax Lighting scene, please. | HassTurnOn(name='Relax Lighting', domain=['scene']) |
| routines_vacuum_05 | Please run my Welcome Home routine. | welcome_home() |
| routines_vacuum_06 | start cleaning with Downstairs Robot Vacuum now. | HassVacuumStart(name='Downstairs Robot Vacuum', domain=['vacuum']) |
| routines_vacuum_07 | Please dock Kitchen Robot Vacuum again. | HassVacuumReturnToBase(name='Kitchen Robot Vacuum', domain=['vacuum']) |
| routines_vacuum_08 | get Roomba to clean the living room. | HassVacuumCleanArea(name='Roomba', area='Living Room') |
| routines_vacuum_09 | I'd like the Relax Lighting scene activated. | HassTurnOn(name='Relax Lighting', domain=['scene']) |
| routines_vacuum_10 | activate the routine named Evening Cleanup for me. | evening_cleanup() |
| status_01 | Is Front Door Lock locked right now? | GetLiveContext(name='Front Door Lock', domain='lock') |
| status_02 | did I leave Breakfast Bar Pendants switched on? | GetLiveContext(name='Breakfast Bar Pendants', domain='light') |
| status_03 | Is Bedroom Tower Fan running at the moment? | GetLiveContext(name='Bedroom Tower Fan', domain='fan') |
| status_04 | what mode is Hallway Heating currently in? | GetLiveContext(name='Hallway Heating', domain='climate') |
| status_05 | Is Family TV still playing? | GetLiveContext(name='Family TV', domain='media_player') |
| status_06 | are Bedroom Blinds open or closed? | GetLiveContext(name='Bedroom Blinds', domain='cover') |
| status_07 | Could you check whether Kitchen Kettle Plug is on? | GetLiveContext(name='Kitchen Kettle Plug', domain='switch') |
| status_08 | is Roomba charging right now? | GetLiveContext(name='Roomba', domain='vacuum') |
| status_09 | Tell me if Sam's Bedside Lamp is off. | GetLiveContext(name="Sam's Bedside Lamp", domain='light') |
| status_10 | what's the current state of Bathroom Shower Light? | GetLiveContext(name='Bathroom Shower Light', domain='light') |
| timers_01 | Set a twelve minute pasta timer for me. | HassStartTimer(minutes=12, name='pasta') |
| timers_02 | can you put the tea timer on pause? | HassPauseTimer(name='tea') |
| timers_03 | How long is left on my laundry timer? | HassTimerStatus(name='laundry') |
| timers_04 | clear every timer in the kitchen, please. | HassCancelAllTimers(area='Kitchen') |
| timers_05 | Give me a timer for forty five minutes called bread. | HassStartTimer(minutes=45, name='bread') |
| timers_06 | i need a two hour timer called laundry. | HassStartTimer(hours=2, name='laundry') |
| timers_07 | Please hold the pasta timer for a moment. | HassPauseTimer(name='pasta') |
| timers_08 | tell me how much time the bread timer has left. | HassTimerStatus(name='bread') |
| timers_09 | Please cancel every timer in the bedroom. | HassCancelAllTimers(area='Bedroom') |
| timers_10 | start a twenty second timer called stretch. | HassStartTimer(seconds=20, name='stretch') |
| unavailable_01 | Set Kitchen Counter Lights to 42 percent brightness for me. | unsupported |
| unavailable_02 | add oat milk to Shopping List, please. | unsupported |
| unavailable_03 | Please turn on the garage heater. | unsupported |
| unavailable_04 | start my Holiday Mode routine, please. | unsupported |
| unavailable_05 | Please start an eight minute rice timer. | unsupported |
| unavailable_06 | i'd like Bedroom Ceiling Fan at 55 percent speed. | unsupported |
| unavailable_07 | Please open the driveway gate. | unsupported |
| unavailable_08 | could you put Main Thermostat at 69 degrees? | unsupported |
| unavailable_09 | Adjust Sofa TV to 22 percent volume. | unsupported |
| unavailable_10 | please start cleaning with Downstairs Vacuum. | unsupported |
