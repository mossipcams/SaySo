#!/usr/bin/env bash
# Publish a SaySo GGUF as a GitHub Release asset and repoint the integration at it.
#
# Weights cannot live in the repo: every useful quant is over GitHub's 100 MB
# file limit, and HACS re-downloads the integration directory on every update.
# Release assets allow 2 GB and stay out of git history.
#
# Run this on the machine that holds the model (the training box):
#
#   scripts/publish_model.sh /srv/models/sayso-lfm25-230m-q4_k_m.gguf
#   scripts/publish_model.sh <model.gguf> [tag]          # default tag: model-v1
#   scripts/publish_model.sh <model.gguf> --dry-run      # show what would happen
#
# TITLE="SaySo Gauntlet v1" sets the release title (default: "SaySo model <tag>").
#
# Then commit the const.py change it makes.

set -euo pipefail

MODEL="${1:-}"
TAG="${2:-model-v1}"
DRY_RUN=0
[ "${2:-}" = "--dry-run" ] && { TAG="model-v1"; DRY_RUN=1; }
[ "${3:-}" = "--dry-run" ] && DRY_RUN=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONST="$REPO_ROOT/custom_components/sayso/const.py"

if [ -z "$MODEL" ]; then
  # Print the header comment block and stop at the first line that is not one.
  awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "${BASH_SOURCE[0]}"
  exit 2
fi

if [ ! -f "$MODEL" ]; then
  echo "error: no such file: $MODEL" >&2
  exit 1
fi

# A truncated or wrong-format file would fail at load time on every user's box,
# after a few hundred megabytes of download. Check the magic bytes here instead.
if [ "$(head -c 4 "$MODEL")" != "GGUF" ]; then
  echo "error: $MODEL is not a GGUF file (bad magic bytes)" >&2
  exit 1
fi

FILENAME="$(basename "$MODEL")"
SIZE_MB=$(( $(wc -c < "$MODEL") / 1000000 ))
echo "==> model:  $FILENAME (${SIZE_MB} MB)"

echo "==> hashing"
if command -v sha256sum >/dev/null; then
  SHA="$(sha256sum "$MODEL" | cut -d' ' -f1)"
else
  SHA="$(shasum -a 256 "$MODEL" | cut -d' ' -f1)"
fi
echo "    sha256: $SHA"

SLUG="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
URL="https://github.com/$SLUG/releases/download/$TAG/$FILENAME"
echo "==> target: $URL"

if [ "$DRY_RUN" = "1" ]; then
  echo "==> dry run: no upload, no edit"
else
  if gh release view "$TAG" >/dev/null 2>&1; then
    echo "==> release $TAG exists, uploading asset"
  else
    echo "==> creating release $TAG"
    # A separate tag from release-please's version tags: the model and the
    # integration are versioned independently, so a patch bump does not mean
    # re-uploading hundreds of megabytes.
    gh release create "$TAG" \
      --title "${TITLE:-SaySo model ${TAG#model-}}" \
      --notes "${NOTES:-GGUF weights for the SaySo conversation agent. Downloaded automatically on first setup.}" \
      --latest=false
  fi
  gh release upload "$TAG" "$MODEL" --clobber
fi

echo "==> repointing $CONST"
python3 - "$CONST" "$URL" "$FILENAME" "$SHA" "$DRY_RUN" <<'PY'
import re, sys

const_path, url, filename, sha, dry = sys.argv[1:6]
source = original = open(const_path).read()

wrapped = '(\n    "%s"\n    "%s"\n)' % (url[: url.rindex("/") + 1], url[url.rindex("/") + 1 :])
edits = [
    (r'DEFAULT_MODEL_URL = \((?:\s*"[^"]*"\s*)+\)|DEFAULT_MODEL_URL = "[^"]*"',
     f"DEFAULT_MODEL_URL = {wrapped}"),
    (r'DEFAULT_MODEL_FILENAME = "[^"]*"',
     f'DEFAULT_MODEL_FILENAME = "{filename}"'),
    (r'DEFAULT_MODEL_SHA256: str \| None = (?:None|"[^"]*")',
     f'DEFAULT_MODEL_SHA256: str | None = "{sha}"'),
]
for pattern, replacement in edits:
    source, count = re.subn(pattern, replacement, source, count=1)
    if count != 1:
        sys.exit(f"error: could not rewrite {pattern.split(' ')[0]} in {const_path}")

if dry == "1":
    print("    (dry run) would write:")
    for line in source.splitlines():
        if "DEFAULT_MODEL_" in line or line.strip().startswith('"https://github.com'):
            print("      " + line)
else:
    open(const_path, "w").write(source)
    print("    updated DEFAULT_MODEL_URL, DEFAULT_MODEL_FILENAME, DEFAULT_MODEL_SHA256")
PY

echo
if [ "$DRY_RUN" = "1" ]; then
  echo "Dry run complete. Re-run without --dry-run to publish."
else
  echo "Done. Review and commit:"
  echo "    git diff custom_components/sayso/const.py"
fi
