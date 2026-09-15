#!/usr/bin/env bash
# Prove llama-cpp-python installs and loads an LFM2 GGUF inside the real Home
# Assistant container on amd64 and arm64.
#
# This is the gate for embedded CPU inference (docs/PLAN_EMBEDDED_INFERENCE.md).
# Requires Docker with binfmt/multi-arch emulation.
#
#   scripts/verify_embedded_backend.sh [HA_VERSION]

set -euo pipefail

HA_VERSION="${1:-2026.8.3}"
WHEEL_INDEX="https://abetlen.github.io/llama-cpp-python/whl/cpu"
MODEL_REPO="LiquidAI/LFM2.5-350M-GGUF"
MODEL_FILE="LFM2.5-350M-Q4_K_M.gguf"
WORK="${TMPDIR:-/tmp}/sayso_verify"
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPTS")"

mkdir -p "$WORK"
cp "$SCRIPTS/verify_embedded_probe.py" "$WORK/probe.py"
cp "$SCRIPTS/verify_embedded_e2e.py" "$WORK/e2e.py"

if [ ! -f "$WORK/$MODEL_FILE" ]; then
  echo "==> downloading $MODEL_FILE"
  curl -fsSL -o "$WORK/$MODEL_FILE" \
    "https://huggingface.co/$MODEL_REPO/resolve/main/$MODEL_FILE"
fi

run_arch() {
  local platform="$1" image="$2"
  echo
  echo "########## $platform ($image:$HA_VERSION) ##########"
  docker run --rm --platform "$platform" -v "$WORK:/models" -v "$REPO:/repo:ro" \
    --entrypoint sh "ghcr.io/home-assistant/$image:$HA_VERSION" -c '
      set -e
      echo "python: $(python3 -c "import sys;print(sys.version.split()[0])")"
      echo "libc  : $(python3 -c "import platform;print(platform.libc_ver())")"
      echo "cc    : $(which gcc cc 2>/dev/null || echo "none (sdist build impossible)")"

      echo "--- control: plain PyPI install, as a manifest requirement would do ---"
      if python3 -m uv pip install --quiet llama-cpp-python \
           --index-strategy unsafe-first-match --upgrade >/dev/null 2>&1; then
        echo "UNEXPECTED: PyPI install succeeded"
      else
        echo "EXPECTED FAIL: no wheel on PyPI and no compiler in the image"
      fi

      echo "--- prebuilt wheel, --no-deps to preserve HA pinned numpy ---"
      npre=$(python3 -c "import numpy;print(numpy.__version__)")
      UV_EXTRA_INDEX_URL="$UV_EXTRA_INDEX_URL '"$WHEEL_INDEX"'" \
        python3 -m uv pip install --quiet --no-deps llama-cpp-python diskcache \
        --index-strategy unsafe-first-match
      npost=$(python3 -c "import numpy;print(numpy.__version__)")
      echo "numpy $npre -> $npost"
      [ "$npre" = "$npost" ] || { echo "FAIL: numpy was disturbed"; exit 1; }

      python3 /models/probe.py

      echo "--- end-to-end through the SaySo EmbeddedEngine ---"
      python3 /models/e2e.py
    '
}

run_arch linux/amd64 amd64-homeassistant
run_arch linux/arm64 aarch64-homeassistant

echo
echo "Both architectures passed."
