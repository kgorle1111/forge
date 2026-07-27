# ⚒ The Forge

**An AI/ML-powered learning tool that refuses to let you fool yourself.**

## What to build → what this is

The brief: build an educational tool that uses AI/ML to make knowledge more
**accessible**, **engaging**, and **personalized**. The Forge does this by
replacing passive studying with a closed loop of AI-taught, AI-graded active
recall:

- **Accessible** — give it any topic, or your own notes/PDFs, and it drafts a
  first-principles curriculum in seconds. No pre-built course, no syllabus,
  no subject-matter expert required.
- **Engaging** — it never lets you re-read. It teaches a concept, takes the
  lesson away, and makes you rebuild it from memory, graded against a 0.8
  mastery bar. Fail, and it re-teaches from a *different angle* — analogy,
  worked example, misconception-busting, visual — not the same paragraph again.
- **Personalized** — every answer you give tunes a per-concept spaced-repetition
  schedule to your own forgetting curve, ranks your genuine weak spots, and
  interleaves them back into future quizzes so nothing you've learned stays safe.

100% local: your own Ollama model, one SQLite file, zero cloud, zero API keys,
zero telemetry.

---

## How AI/ML is actually used

Four LLM-driven agents run in a loop (LangGraph), each with one job:

| Agent | Role |
|---|---|
| **Synthesizer** | breaks a topic into first-principles concepts, teaches one |
| **Inquisitor** | removes the lesson, forces retrieval with a quiz |
| **Evaluator** | grades the answer against a 0.8 mastery threshold, with a non-LLM heuristic floor so a broken model can never auto-pass you |
| **Orchestrator** | commits mastered concepts to memory, schedules the next review with SM-2 |

A failed answer routes back to the Synthesizer, which rotates through six
teaching angles instead of repeating itself. The grading model is swappable
independently of the teaching model (`FORGE_GRADER`) for stricter evaluation.

## Demo

```bash
bash UP-AND-RUNNING/setup.sh        # one command: venv, install, tests, health check
.venv/bin/forge web                 # dashboard -> http://127.0.0.1:8765
```

No Ollama installed? Run in stub mode for an instant offline demo:

```bash
FORGE_STUB=1 FORGE_DB=/tmp/forge_demo.db .venv/bin/forge web --port 8766
```

or the terminal:

```bash
.venv/bin/forge learn "fourier transforms"
.venv/bin/forge learn "my sourdough notes" --from notes.md --from paper.pdf
.venv/bin/forge review              # do this daily — it IS the product
.venv/bin/forge stats               # recall %, streak, 14-day forecast
.venv/bin/forge graph               # knowledge graph -> forge_graph.html
```

Full command reference, daily-practice guide, and troubleshooting:
[UP-AND-RUNNING/README.md](UP-AND-RUNNING/README.md).

---

## Why it works — the science, honestly applied

Almost everything people do to learn (re-reading, highlighting, watching
videos) produces *fluency* — the material feels familiar, so you believe you
know it. Familiarity is not retrieval. The Forge is built exclusively from the
techniques with the strongest measured effects:

1. **The testing effect** (Roediger & Karpicke 2006): retrieving a memory
   strengthens it far more than re-studying it. The Forge never lets you
   re-read — the Inquisitor removes the lesson and forces reconstruction.
2. **Mastery gating** (Bloom 1984): learners tutored to mastery before
   advancing performed ~2 standard deviations above classroom learners. The
   Evaluator is that gate: 0.8 or you don't pass, ever.
3. **Encoding variability / desirable difficulties** (Bjork): a failed recall
   followed by a *re-explanation in a new form* builds more durable, more
   transferable memory than repeating the same explanation. Failure routes
   back to the Synthesizer, which rotates through six angles: first
   principles → analogy → worked example → misconceptions → spatial → visual.
4. **Spaced repetition on your forgetting curve** (Ebbinghaus; SM-2): each
   concept carries an ease factor that adapts to *your* answers. Intervals
   grow ~2.5× per success and collapse to 1 day on a lapse.
5. **Interleaving** (Rohrer & Taylor 2007): mixed practice beats blocked
   practice. Your historically weakest old concept gets injected into every
   new topic's quiz.

The uncomfortable design consequence: **The Forge is supposed to feel harder
than studying normally.** That difficulty is the signal that encoding is
happening.

## Architecture

```
                         ┌─────────────────────────────────────────┐
                         │              LangGraph loop             │
                         │                                         │
  topic ──► SYNTHESIZER ─┼─► INQUISITOR ─► EVALUATOR ──pass──► ORCHESTRATOR ──► next concept
            breaks topic │   removes the    grades vs           commits to memory,      │
            into first   │   lesson, forces  0.8 mastery        SM-2 schedules review   │
            principles,  │   reconstruction  threshold               │                  │
            teaches one  │                      │                    └──all mastered──► END
            concept      │                     fail                                    
                ▲        │                      │                                       
                └────────┴──────────────────────┘  re-teach from the next angle        
```

```
forge/
  state.py      ForgeState TypedDict — the contract between agents
  graph.py      LangGraph wiring (the loop above) + shared review runner
  agents.py     the four agents, their prompts, grading, mastery gate
  memory.py     SQLite: SM-2 scheduler, attempts, weaknesses, FTS5 lessons
  sources.py    local source ingestion, chunking, manifests, citations
  llm.py        Ollama (stdlib HTTP) + deterministic offline stub
  web.py        dashboard server (stdlib http.server, localhost-only)
  dashboard.html  the dashboard UI (vanilla JS, no CDN, works offline)
  cli.py        terminal UI
  graphview.py  knowledge-graph HTML export
tests/test_forge.py   end-to-end tests, fully offline, ~2 seconds
```

Design keystone: agents receive their collaborators (`llm`, `memory`, `ask`,
`emit`) by injection, so the **identical loop** drives the terminal, the web
dashboard, and the test suite.

## The dashboard

`forge web` → http://127.0.0.1:8765

- **Session panel** — the live conversation with the agents: curriculum, lessons, quizzes, verdicts.
- **Source picker/drop zone** — attach or drag in a local `.txt`, `.md`, or `.pdf` to ground the next Learn session.
- **Retrieval cockpit** — due queue, weak concepts, and active progress/resume panels.
- **Knowledge graph** — every concept you've mastered, colored by strength (green→red), white ring = due for review.
- **Memory & schedule table** — risk-colored sorting, topic filtering, concept detail drawer, and due/suspend actions.

Binds to 127.0.0.1 only. No auth because it never leaves your machine.

## Reliability posture

- **Offline-first**: no Ollama → deterministic stub keeps every feature and test working (this is how the demo runs without setup).
- **The gate can't be bribed**: if the LLM dies mid-session, grading falls back to deterministic key-point matching — a broken model can never auto-pass you.
- **One-file state**: the entire learning history is `~/.forge/forge.db`.
- **Low maintenance by construction**: one dependency (`langgraph`), stdlib everything else, no services, no keys to rotate.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `FORGE_MODEL` | auto (prefers coder/instruct) | force an Ollama model |
| `FORGE_GRADER` | same as teaching model | separate grading model for stricter evaluation |
| `FORGE_OLLAMA` | `http://localhost:11434` | Ollama location |
| `FORGE_DB` | `~/.forge/forge.db` | memory file |
| `FORGE_STUB` | unset | `1` = no-LLM demo/offline mode |

## Testing

```bash
.venv/bin/python tests/test_forge.py
```

Covers the grading heuristic, SM-2 interval math, the full mastery loop
(including being trapped and re-taught), interleaving of old weaknesses,
graph export, and the web dashboard driven over HTTP.
