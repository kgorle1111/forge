#!/usr/bin/env bash
# WS-QA stranger simulation: wheel -> fresh venv + fresh HOME -> init/learn/
# review/doctor/web, exactly as a new user would hit them. macOS + CI ubuntu.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
WEB_PID=""
OCCUPY_PID=""

cleanup() {
    [ -n "$WEB_PID" ] && kill "$WEB_PID" 2>/dev/null || true
    [ -n "$OCCUPY_PID" ] && kill "$OCCUPY_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    rm -rf "$TMP"
}
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

# perms helper: BSD and GNU stat disagree on flags (-f is a *different valid
# flag* on GNU, so ||-fallback silently compares garbage) — use python
perms() {
    python3 -c 'import os,stat,sys; print(oct(stat.S_IMODE(os.stat(sys.argv[1]).st_mode))[2:])' "$1"
}

# hermetic env: no leakage from the developer's real setup
unset FORGE_STUB FORGE_ENGINE FORGE_MODEL FORGE_GRADER FORGE_DB \
      FORGE_CONFIG FORGE_LEARNER FORGE_OLLAMA ANTHROPIC_API_KEY 2>/dev/null || true
export FORGE_OLLAMA="http://127.0.0.1:9"   # dead port: never probe a real Ollama

echo "== build wheel =="
uv build --out-dir "$TMP/dist" "$ROOT" >/dev/null
WHEEL="$(ls "$TMP"/dist/*.whl)"
echo "wheel: $WHEEL"

echo "== fresh venv + install (wheel only) =="
T0=$(date +%s)
uv venv "$TMP/venv" >/dev/null 2>&1
uv pip install --python "$TMP/venv/bin/python" "$WHEEL" >/dev/null
FORGE="$TMP/venv/bin/forge"
[ -x "$FORGE" ] || fail "forge entry point not installed"

export HOME="$TMP/home"
mkdir -p "$HOME"

echo "== forge init --engine stub =="
"$FORGE" init --engine stub
CFG="$HOME/.forge/config.json"
[ -f "$CFG" ] || fail "config.json not created at $CFG"
[ "$(perms "$CFG")" = "600" ] || fail "config.json mode is $(perms "$CFG"), want 600"

echo "== init overwrite: ollama -> back to stub =="
"$FORGE" init --engine ollama --model tinyllama | grep -q "tinyllama" \
    || fail "init did not report the ollama model"
grep -q '"engine": "ollama"' "$CFG" || fail "config not overwritten to ollama"
grep -q '"model": "tinyllama"' "$CFG" || fail "config lost the model"
"$FORGE" init --engine stub
grep -q '"engine": "stub"' "$CFG" || fail "config not switched back to stub"
[ "$(perms "$CFG")" = "600" ] || fail "config mode drifted after overwrite"

echo "== non-interactive learn (stub) =="
# stub curriculum: 3 concepts x 2 questions = 6 answers; extra lines are unread
FORGE_STUB=1 "$FORGE" learn "qa topic" --max-failures 1 <<'EOF'
remember alpha beta gamma
remember alpha beta gamma
remember alpha beta gamma
remember alpha beta gamma
remember alpha beta gamma
remember alpha beta gamma
remember alpha beta gamma
remember alpha beta gamma
remember alpha beta gamma
remember alpha beta gamma
remember alpha beta gamma
remember alpha beta gamma
EOF
T1=$(date +%s)
[ -f "$HOME/.forge/forge.db" ] || fail "forge.db not created"
FORGE_STUB=1 "$FORGE" stats | grep -q "qa topic" \
    || fail "learned topic missing from stats"
echo "TIMING: $((T1 - T0))s install-to-first-lesson"
# the roadmap gate is <2 min; a regression past it is a bug, not a log line
[ $((T1 - T0)) -lt 120 ] || fail "install-to-first-lesson took $((T1 - T0))s (gate: <120s)"

echo "== review exits cleanly =="
# everything was just mastered -> nothing due; must not hang on stdin
FORGE_STUB=1 "$FORGE" review </dev/null | grep -q "Nothing due" \
    || fail "review did not exit cleanly with nothing due"

echo "== doctor with dead ollama: nonzero + prescriptive =="
set +e
DOCTOR_OUT="$(FORGE_ENGINE=ollama FORGE_OLLAMA=http://127.0.0.1:1 "$FORGE" doctor 2>&1)"
DOCTOR_RC=$?
set -e
[ "$DOCTOR_RC" -ne 0 ] || fail "doctor exited 0 with unreachable ollama"
echo "$DOCTOR_OUT" | grep -q -e "ollama serve" -e "forge init" \
    || fail "doctor failure text is not prescriptive: $DOCTOR_OUT"

echo "== web on an occupied port =="
PORT=8971
"$TMP/venv/bin/python" -c "
import socketserver, http.server
socketserver.TCPServer.allow_reuse_address = False
with socketserver.TCPServer(('127.0.0.1', $PORT), http.server.BaseHTTPRequestHandler) as s:
    s.serve_forever()
" &
OCCUPY_PID=$!
sleep 1
set +e
FORGE_STUB=1 "$FORGE" web --port "$PORT" >"$TMP/web_out.txt" 2>&1 &
WEB_PID=$!
for _ in $(seq 1 20); do
    kill -0 "$WEB_PID" 2>/dev/null || break
    sleep 0.5
done
if kill -0 "$WEB_PID" 2>/dev/null; then
    kill "$WEB_PID"
    fail "forge web hung instead of failing on an occupied port"
fi
wait "$WEB_PID"
WEB_RC=$?
set -e
WEB_PID=""
[ "$WEB_RC" -ne 0 ] || fail "forge web exited 0 on an occupied port"
# fixed in P4 triage (was DEFECT WS-QA-1): occupied port must yield a
# prescriptive one-liner, never a traceback
grep -q "Traceback" "$TMP/web_out.txt" \
    && fail "forge web printed a raw traceback on an occupied port"
grep -q "already in use" "$TMP/web_out.txt" \
    || fail "web failure output is not prescriptive: $(cat "$TMP/web_out.txt")"

echo "TIMING: $((T1 - T0))s install-to-first-lesson"
echo "QA FRESH INSTALL: ALL CHECKS PASSED"
