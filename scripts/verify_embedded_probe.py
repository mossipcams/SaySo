#!/usr/bin/env python3
"""Load an LFM2 GGUF through llama-cpp-python and report what the bindings expose.

Run inside the Home Assistant container by scripts/verify_embedded_backend.sh.
Exits non-zero if the native dependency cannot load or run the model.
"""

from __future__ import annotations

import glob
import platform
import sys
import time

from llama_cpp import Llama

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
    }
]
MESSAGES = [
    {"role": "system", "content": "You control Home Assistant. Use the tools."},
    {"role": "user", "content": "turn on the kitchen lights"},
]


def main() -> int:
    (model,) = glob.glob("/models/*.gguf")
    print(f"machine={platform.machine()} libc={platform.libc_ver()}")

    started = time.perf_counter()
    llm = Llama(model_path=model, n_ctx=2048, n_threads=4, verbose=False)
    print(f"LOAD_OK load_ms={(time.perf_counter() - started) * 1000:.0f}")

    arch = llm.metadata.get("general.architecture")
    print(f"arch={arch} chat_format={llm.chat_format}")
    if arch != "lfm2":
        print(f"FAIL: bundled llama.cpp did not recognise lfm2 (got {arch})")
        return 1

    started = time.perf_counter()
    out = llm.create_chat_completion(
        messages=MESSAGES, tools=TOOLS, tool_choice="auto", temperature=0, max_tokens=128
    )
    print(f"INFER_OK infer_ms={(time.perf_counter() - started) * 1000:.0f}")

    message = out["choices"][0]["message"]
    structured = bool(message.get("tool_calls"))
    print(f"raw_content={message.get('content')!r}")
    # Recorded deliberately: the bindings ship libllama only, not llama.cpp's
    # common/chat.cpp, so LFM2 tool calls arrive as text and SaySo must parse
    # them itself. See docs/PLAN_EMBEDDED_INFERENCE.md.
    print(f"structured_tool_calls={structured}")

    import llama_cpp

    has_parser = hasattr(llama_cpp.llama_cpp._lib, "common_chat_parse")
    print(f"exposes_common_chat_parse={has_parser}")

    print("PROBE_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
