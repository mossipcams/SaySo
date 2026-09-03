"""Detect GPU capabilities for Axolotl training configs."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GpuProfile:
    """Resolved training precision settings for this machine."""

    name: str
    fp16: bool
    bf16: bool
    flash_attention: bool
    notes: str


def detect_gpu() -> GpuProfile:
    """Detect GPU and return safe training settings."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return GpuProfile(
            name="none",
            fp16=False,
            bf16=False,
            flash_attention=False,
            notes="No NVIDIA GPU detected; use CPU or document skip.",
        )

    if result.returncode != 0 or not result.stdout.strip():
        return GpuProfile(
            name="none",
            fp16=False,
            bf16=False,
            flash_attention=False,
            notes="nvidia-smi failed",
        )

    line = result.stdout.strip().splitlines()[0]
    gpu_name = line.split(",")[0].strip().lower()

    # GTX 10xx (Pascal): FP16 works, no native BF16, no flash-attn
    if any(tag in gpu_name for tag in ("gtx 10", "gtx 16", "pascal")):
        return GpuProfile(
            name=gpu_name,
            fp16=True,
            bf16=False,
            flash_attention=False,
            notes="Pascal-class GPU: FP16 only, no BF16 or flash-attn",
        )

    # Ampere+ generally supports BF16 and flash attention
    if any(tag in gpu_name for tag in ("rtx 30", "rtx 40", "a100", "h100", "ampere", "ada")):
        return GpuProfile(
            name=gpu_name,
            fp16=True,
            bf16=True,
            flash_attention=True,
            notes="Modern GPU: BF16 and flash-attn available",
        )

    return GpuProfile(
        name=gpu_name,
        fp16=True,
        bf16=False,
        flash_attention=False,
        notes="Unknown GPU: conservative FP16, no BF16/flash-attn",
    )


if __name__ == "__main__":
    profile = detect_gpu()
    print(f"GPU: {profile.name}")
    print(f"FP16: {profile.fp16}")
    print(f"BF16: {profile.bf16}")
    print(f"Flash Attention: {profile.flash_attention}")
    print(f"Notes: {profile.notes}")
