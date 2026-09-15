"""Drive SaySo's real EmbeddedEngine against a real GGUF inside the HA image.

Run by scripts/verify_embedded_backend.sh. This is the check that caught the
parser only handling single-quoted arguments while LFM2.5 emits double quotes.
"""

import asyncio
import glob
import sys
from pathlib import Path

sys.path.insert(0, "/repo")

from custom_components.sayso.inference import EmbeddedEngine, extract_tool_calls

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "HassTurnOn",
            "description": "Turns on a device or entity",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "area": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "HassTurnOff",
            "description": "Turns off a device or entity",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
]


async def main() -> int:
    (model,) = glob.glob("/models/*.gguf")
    engine = EmbeddedEngine(Path(model), n_ctx=2048, n_threads=4, timeout=120)
    await engine.async_start()
    print(f"engine started, model_name={engine.model_name}")

    failures = 0
    try:
        for utterance in (
            "turn on the kitchen lights",
            "turn off the lamp",
            "what is the weather like",
        ):
            result = await engine.async_chat_completion(
                [
                    {"role": "system", "content": "You control Home Assistant."},
                    {"role": "user", "content": utterance},
                ],
                tools=TOOLS,
                temperature=0,
                max_tokens=128,
            )
            calls = [(c.name, c.arguments) for c in result.tool_calls]
            print(f"\n  {utterance!r}")
            print(f"    content   = {result.content!r}")
            print(f"    toolcalls = {calls}")
            if not calls and not result.content:
                failures += 1
    finally:
        await engine.async_shutdown()
        print("\nengine shut down cleanly")

    # The engine must survive shutdown without leaving a usable handle behind.
    try:
        await engine.async_chat_completion([{"role": "user", "content": "hi"}])
        print("FAIL: engine still served a request after shutdown")
        return 1
    except Exception as err:
        print(f"post-shutdown request correctly refused: {type(err).__name__}")

    # Parser sanity against the exact text llama.cpp produced above.
    text, calls = extract_tool_calls("[HassTurnOn(name='O'Malley's lamp')]")
    assert calls[0].arguments["name"] == "O'Malley's lamp", calls
    print("apostrophe-safe parse OK")

    print("E2E_PASS" if failures == 0 else f"E2E_FAIL ({failures} empty responses)")
    return 1 if failures else 0


sys.exit(asyncio.run(main()))
