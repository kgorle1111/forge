# Up and running in 3 minutes

Everything you need, in order. Nothing else is required.

## What you need

| Requirement | Why | Check |
|---|---|---|
| Python 3.10+ | runs The Forge | `python3 --version` |
| Ollama + a model | the local brain (already installed on this Mac) | `curl -s localhost:11434/api/tags` |

You already have `qwen2.5-coder:32b` and `deepseek-r1:32b` pulled. The Forge
auto-picks `qwen2.5-coder:32b` (better at structured output than a reasoning
model). No internet, no API keys, no accounts.

## Setup (one command)

```bash
cd "/Users/kannishknaidu/the learning machine"
bash UP-AND-RUNNING/setup.sh
```

This creates the venv, installs The Forge, runs the full test suite, and tells
you whether Ollama is reachable. Already done once by Claude — safe to re-run
anytime; it is idempotent.

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

Tip: add an alias so it's one word from anywhere:

```bash
echo 'alias forge="\"/Users/kannishknaidu/the learning machine/.venv/bin/forge\""' >> ~/.zshrc
```

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

- **"model: stub" when you expected the real model** → Ollama isn't running: `ollama serve` (or open the Ollama app).
- **Slow lessons** → 32B models are heavy; pull a smaller one (`ollama pull llama3.2`) and run `FORGE_MODEL=llama3.2 forge web`.
- **Start over / test safely** → `FORGE_DB=/tmp/sandbox.db forge web` uses a throwaway memory. Your real memory is the single file `~/.forge/forge.db` — back it up by copying it.
- **Anything else** → `.venv/bin/python tests/test_forge.py` tells you in seconds if the install is healthy.

## Environment knobs (all optional)

| Variable | Default | Purpose |
|---|---|---|
| `FORGE_MODEL` | auto-pick | force a specific Ollama model |
| `FORGE_OLLAMA` | `http://localhost:11434` | Ollama on another host/port |
| `FORGE_DB` | `~/.forge/forge.db` | where your memory lives |
| `FORGE_STUB` | unset | `1` forces the offline no-LLM stub (demos/tests) |
