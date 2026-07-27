# Up and running in 3 minutes

Everything you need, in order. Nothing else is required.

## What you need

| Requirement | Why | Check |
|---|---|---|
| Python 3.10+ | runs The Forge | `python3 --version` |
| An engine (pick one below) | the brain | `forge doctor` after setup |

Engines, in order of recommendation:

- **Local Ollama** — private, free. Install from [ollama.com](https://ollama.com), then `ollama pull` any model.
- **Cloud API** — `export ANTHROPIC_API_KEY=...` (a topic session is typically cents on claude-haiku-4-5). The key lives in your environment only — never sent over HTTP to Forge.
- **Demo stub** — nothing to install; instant and deterministic, no AI quality.

## Setup (one command)

```bash
bash UP-AND-RUNNING/setup.sh
.venv/bin/forge init
```

`setup.sh` creates the venv, installs The Forge, runs the full test suite, and
checks whether Ollama is reachable. Safe to re-run anytime; it is idempotent.
`forge init` picks your engine interactively — or non-interactively:

```bash
.venv/bin/forge init --engine ollama --model llama3.2
.venv/bin/forge init --engine anthropic     # needs ANTHROPIC_API_KEY exported
.venv/bin/forge init --engine stub          # offline demo
```

Once published to PyPI, install becomes one line: `uv tool install forge-learning && forge init` (or just `uvx forge-learning web`).

## Use it

```bash
.venv/bin/forge web                     # dashboard -> http://127.0.0.1:8765
.venv/bin/forge learn "fourier transforms"   # or pure terminal mode
.venv/bin/forge learn "biology exam" --from my_notes.md   # learn YOUR material
.venv/bin/forge learn "stoicism" --speak     # lessons & questions read aloud
.venv/bin/forge review                  # daily: drill what's due
.venv/bin/forge stats                   # memory + recall % + streak + forecast
.venv/bin/forge export                  # Anki-importable backup of all lessons
.venv/bin/forge graph                   # knowledge graph -> forge_graph.html
```

Walk away mid-session anytime (Ctrl-C) — `forge learn` with the same topic
resumes at the exact concept you left.

## The daily practice (this is where the learning happens)

1. **Every day, first**: `forge review` (or the Review button). 2–10 minutes.
   Skipping reviews is the only way to break the system — the schedule is the product.
2. **When you want something new**: `forge learn "<topic>"`. Expect to fail
   quizzes. Failing is the mechanism, not a bug — each failure re-teaches from
   a new angle and the struggle itself strengthens the memory.
3. **Weekly**: glance at `forge stats` / the dashboard graph. Red nodes and
   high-lapse rows are your genuine weak spots; the system already interleaves
   them into new sessions automatically.

## If something breaks

Start with the diagnostic — every line below comes from its output:

```bash
.venv/bin/forge doctor
```

| `forge doctor` says | Fix |
|---|---|
| `engine health: FAIL — Ollama not reachable at http://localhost:11434 — start it with `` `ollama serve` ``, or switch engines with FORGE_ENGINE=anthropic or FORGE_STUB=1` | `ollama serve` (or open the Ollama app), then re-run `forge doctor` |
| `engine health: FAIL — model 'NAME' not installed — run `` `ollama pull NAME` `` or unset FORGE_MODEL` | `ollama pull NAME`, or `unset FORGE_MODEL` to let Forge auto-pick |
| `engine health: FAIL — no Anthropic API key — set ANTHROPIC_API_KEY or run `` `forge init` `` | `export ANTHROPIC_API_KEY=...` (shell env only — never paste keys into web forms) |
| `engine: FAIL — engine 'NAME' is unavailable: ...` | your forced engine (`FORGE_ENGINE` or config) is down — fix it per the message, or `forge init --engine stub` to keep working offline |
| `engine: stub (stub)` when you expected a real model | nothing forced stub, but no engine was found — start Ollama or export a key, then check `forge doctor` again |
| `integrity:` anything other than `ok`, or `missing lessons:` / `duplicate concept names:` lines | `forge doctor --fix` |
| `pdftotext: missing` | only matters for `--from file.pdf`: `brew install poppler` (macOS) / `apt install poppler-utils` |

Other symptoms:

- **Slow lessons** → 32B models are heavy; `ollama pull llama3.2` and `FORGE_MODEL=llama3.2 forge web`.
- **Start over / test safely** → `FORGE_DB=/tmp/sandbox.db forge web` uses a throwaway memory. Your real memory is the single file `~/.forge/forge.db` — back it up by copying it.
- **Anything else** → `.venv/bin/python tests/test_forge.py` tells you in seconds if the install is healthy.

## Environment knobs (all optional)

| Variable | Default | Purpose |
|---|---|---|
| `FORGE_ENGINE` | unset | force `ollama`, `anthropic`, or `stub` |
| `FORGE_MODEL` | auto-pick | force a specific model |
| `FORGE_OLLAMA` | `http://localhost:11434` | Ollama on another host/port |
| `FORGE_ANTHROPIC_URL` | `https://api.anthropic.com` | Anthropic API base URL |
| `FORGE_CONFIG` | `~/.forge/config.json` | engine config written by `forge init` |
| `FORGE_DB` | `~/.forge/forge.db` | where your memory lives |
| `FORGE_STUB` | unset | `1` forces the offline no-LLM stub (demos/tests) |
