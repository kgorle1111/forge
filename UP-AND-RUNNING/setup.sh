#!/usr/bin/env bash
# The Forge — idempotent setup. Run from anywhere; sets up everything needed.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== The Forge setup =="

command -v python3 >/dev/null || { echo "FAIL: python3 not found. Install Python 3.10+."; exit 1; }
echo "python: $(python3 --version)"

[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install --quiet --disable-pip-version-check -e .
echo "install: ok"

.venv/bin/python tests/test_forge.py >/dev/null && echo "tests: all passing"

if curl -sf -m 3 "${FORGE_OLLAMA:-http://localhost:11434}/api/tags" >/dev/null 2>&1; then
  echo "ollama: reachable (real local LLM will be used)"
else
  echo "ollama: NOT reachable — Forge still works via the offline stub."
  echo "        For real lessons: install from https://ollama.com then 'ollama pull qwen2.5-coder:32b'"
fi

echo
echo "Ready. Next:"
echo "  .venv/bin/forge init       # pick your engine (ollama / anthropic / stub)"
echo "  .venv/bin/forge web        # dashboard at http://127.0.0.1:8765"
echo "  .venv/bin/forge learn \"<topic>\""
