#!/usr/bin/env bash
# Regenerates docs/api/ from src/ docstrings via a headless Sphinx build
# (see docs/api_src/conf.py). Runs as the "prestart"/"prebuild" npm hook.
#
# Output lives at docs/docs/api/, a *sibling* of docs/docs/documentation/
# (the main docs plugin's content root) -- both live under the common
# docs/docs/ parent, but neither instance's configured `path` is nested
# inside the other's. Nesting one plugin instance's path inside another
# instance's own path causes SSG rendering to crash ("Cannot read properties
# of undefined (reading 'id')" in DocItem) -- confirmed by testing
# byte-identical content at a sibling path vs. a nested path, only the
# nested one broke. Keep them as siblings under the shared parent.
#
# Prefers the repo's .venv (local dev, where sphinx-build usually isn't on
# PATH unless the venv is activated) and falls back to plain `sphinx-build`
# on PATH (CI, where dependencies are installed globally via
# `pip install -e ".[docs]"`, no .venv involved).
set -euo pipefail

cd "$(dirname "$0")/.."  # docs/

VENV_SPHINX_BUILD="../.venv/bin/sphinx-build"
if [ -x "$VENV_SPHINX_BUILD" ]; then
  SPHINX_BUILD="$VENV_SPHINX_BUILD"
elif command -v sphinx-build >/dev/null 2>&1; then
  SPHINX_BUILD="sphinx-build"
else
  echo "error: sphinx-build not found (.venv/bin/sphinx-build missing and not on PATH)." >&2
  echo "Run: pip install -e \".[docs]\"" >&2
  exit 1
fi

rm -rf docs/api
"$SPHINX_BUILD" -b markdown api_src docs/api
rm -rf docs/api/.doctrees
