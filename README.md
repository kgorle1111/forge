# ⚒ The Forge

**A closed-loop learning machine that refuses to let you fool yourself.**

You give it any topic. It builds a first-principles curriculum, teaches you one
concept, takes the lesson away, and makes you rebuild it from memory. Score
under 0.8 and you don't move — you get re-taught from a *different cognitive
angle* and tested again, as many times as it takes. Pass, and the concept
enters a spaced-repetition schedule fitted to your personal forgetting curve.
Every answer you ever give makes the system better at teaching you.

100% local: your Ollama models, one SQLite file, zero cloud, zero API keys,
zero telemetry, nothing to maintain.

---

## Quickstart

```bash
bash UP-AND-RUNNING/setup.sh        # one command: venv, install, tests, health check
.venv/bin/forge web                 # dashboard -> http://127.0.0.1:8765
```

or pure terminal:

```bash
.venv/bin/forge learn "fourier transforms"
.venv/bin/forge learn "my sourdough notes" --from notes.md --from paper.pdf
.venv/bin/forge learn "stoicism" --speak                     # voice mode (macOS)
.venv/bin/forge learn "fractions" --learner kiddo            # separate local DB
.venv/bin/forge review              # do this daily — it IS the product
.venv/bin/forge due                 # highest-risk due cards
.venv/bin/forge today               # daily retrieval brief
.venv/bin/forge weak                # fragile concepts and why
.venv/bin/forge practice --limit 10 # capped high-yield review
.venv/bin/forge cram --days 7       # pull forward near-deadline reviews
.venv/bin/forge stuck               # repeatedly failed concepts
.venv/bin/forge daily5              # five-card daily retrieval sprint
.venv/bin/forge sprint --mode weak --limit 10
.venv/bin/forge drill --topic "fourier transforms" --concept "basis"
.venv/bin/forge exam "fourier transforms"  # topic-wide retrieval exam
.venv/bin/forge missed              # recurring missed key points
.venv/bin/forge calibration         # confidence vs. score
.venv/bin/forge source-coverage "fourier transforms"
.venv/bin/forge source-glossary "fourier transforms"
.venv/bin/forge source-quotes "fourier transforms" "phase shift"
.venv/bin/forge source-quiz "fourier transforms"
.venv/bin/forge unsupported "fourier transforms"
.venv/bin/forge avalanche           # workload warning before reviews pile up
.venv/bin/forge vacation 3          # shift active due dates by 3 days
.venv/bin/forge performance --by hour
.venv/bin/forge analytics
.venv/bin/forge maintain
.venv/bin/forge csv-export cards.csv
.venv/bin/forge next
.venv/bin/forge progress
.venv/bin/forge notify --dry-run     # local daily review brief
.venv/bin/forge semantic "memory transfer"
.venv/bin/forge bridge               # connect weak concepts across topics
.venv/bin/forge weekly-sheet         # printable review sheet
.venv/bin/forge capstone "fourier transforms"
.venv/bin/forge config
.venv/bin/forge profiles
.venv/bin/forge json-export backup.json
.venv/bin/forge stats               # + recall %, streak, 14-day forecast
.venv/bin/forge export              # Anki-importable TSV of everything learned
.venv/bin/forge graph               # knowledge graph -> forge_graph.html
```

Interrupted mid-topic? Run `forge learn` with the same topic — it resumes at
the exact concept you left.

Full setup details, daily practice guide, and troubleshooting:
[UP-AND-RUNNING/README.md](UP-AND-RUNNING/README.md).

---

## Why it works — the science, honestly applied

Almost everything people do to learn (re-reading, highlighting, watching
videos) produces *fluency* — the material feels familiar, so you believe you
know it. Familiarity is not retrieval. The Forge is built exclusively from the
techniques with the strongest measured effects:

1. **The testing effect** (Roediger & Karpicke 2006): retrieving a memory
   strengthens it far more than re-studying it. So The Forge never lets you
   re-read — the Inquisitor removes the lesson and forces reconstruction.
2. **Mastery gating** (Bloom 1984): learners tutored to mastery before
   advancing performed ~2 standard deviations above classroom learners. The
   Evaluator is that gate: 0.8 or you don't pass, ever.
3. **Encoding variability / desirable difficulties** (Bjork): a failed recall
   followed by a *re-explanation in a new form* builds more durable, more
   transferable memory than repeating the same explanation. Failure routes you
   back to the Synthesizer, which rotates through six angles: first
   principles → analogy → worked example → misconceptions → spatial → visual.
4. **Spaced repetition on your forgetting curve** (Ebbinghaus; SM-2): each
   concept carries an ease factor that adapts to *your* answers. Intervals
   grow ~2.5× per success (which lands each review in the 10–20%
   retention-decay window — the point of maximum strengthening per minute
   spent) and collapse to 1 day on a lapse.
5. **Interleaving** (Rohrer & Taylor 2007): mixed practice beats blocked
   practice. The Inquisitor injects your historically weakest old concept into
   every new topic's quiz, so nothing you've learned is ever safe from being asked.

The uncomfortable design consequence: **The Forge is supposed to feel harder
than studying normally.** That difficulty is the signal that encoding is
happening.

## How it gets smarter as you use it

All adaptation flows through one SQLite file (`~/.forge/forge.db`):

- every answer updates a per-concept **ease factor** → intervals fit your brain, not an average brain;
- **lapses** rank your genuine weak spots → they surface first in reviews and get interleaved into new topics;
- every **lesson** is stored and searchable (FTS5) → reviews re-quiz you against exactly what you were taught;
- every **attempt** is kept forever → future schedulers (FSRS) can be fitted to your real history without losing anything.

---

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
tests/test_forge.py   27 end-to-end tests, fully offline, ~2 seconds
```

Design keystone: agents receive their collaborators (`llm`, `memory`, `ask`,
`emit`) by injection, so the **identical loop** drives the terminal, the web
dashboard, the test suite, and any future interface (voice, etc.).

## The dashboard

`forge web` → http://127.0.0.1:8765

- **Session panel** — the live conversation with the agents: curriculum, lessons, quizzes, verdicts.
- **Source picker/drop zone** — attach or drag in a local `.txt`, `.md`, or `.pdf` to ground the next Learn session.
- **Retrieval cockpit** — due queue, weak concepts, and active progress/resume panels.
- **Knowledge graph** — every concept you've mastered, colored by strength (green→red), white ring = due for review.
- **Memory & schedule table** — risk-colored sorting, topic filtering, concept detail drawer, and due/suspend actions.
- **Local controls** — learner switching, dashboard settings, keyboard flow, and session transcript download.
- **Visual previews** — SVG teaching artifacts render inline in the event log.
- Header chips: active model, total concepts, due-now count.

Binds to 127.0.0.1 only. No auth because it never leaves your machine.

## Reliability & security posture

- **Offline-first**: no Ollama → deterministic stub keeps every feature and test working.
- **The gate can't be bribed**: if the LLM dies mid-session, grading falls back to deterministic key-point matching — a broken model can never auto-pass you.
- **One-file state**: back up your entire learning history by copying `~/.forge/forge.db`.
- **Input validation at the trust boundary**: the web API validates and bounds all input; server is localhost-only.
- **Low maintenance by construction**: one dependency (langgraph), stdlib everything else, no services, no migrations, no keys to rotate.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `FORGE_MODEL` | auto (prefers coder/instruct) | force an Ollama model |
| `FORGE_GRADER` | same as teaching model | separate grading model (e.g. `deepseek-r1:32b` for stricter evaluation, at a latency cost) |
| `FORGE_OLLAMA` | `http://localhost:11434` | Ollama location |
| `FORGE_DB` | `~/.forge/forge.db` | memory file |
| `FORGE_STUB` | unset | `1` = no-LLM mode (demos, tests, CI) |

`forge learn --max-failures N` caps the mastery trap for scripted runs
(default 0 = the mandate: no escape but mastery).
Add `--learner NAME` to any subcommand to use `~/.forge/NAME.db` instead.

## Testing

```bash
.venv/bin/python tests/test_forge.py
```

Covers: the grading heuristic, SM-2 interval math (growth + lapse reset), the
full mastery loop **including being trapped and re-taught**, interleaving of
old weaknesses, graph export, and the entire web dashboard driven over HTTP.

## Feature matrix — built vs. deliberately deferred

Built: mastery gate · 6-angle re-teaching · **targeted re-teach of exactly the
key points you missed** · **per-question feedback (what you missed, why)** ·
**6th visual/SVG teaching angle after repeated failure** ·
SM-2 adaptive scheduling · **FSRS-style recall-probability estimates** ·
**risk-ranked due queue** · **review debt summary** · **topic health** ·
**practice/cram/stuck retrieval commands** · **safe rename/suspend/due-now card actions** ·
**DB backup/explicit-confirm restore** · **retrieval transcript export** ·
**hard-mode review** · **confidence calibration** · **follow-up retrieval for partial answers** ·
**missed-key-point rollups** · **deterministic retrieval prompts** ·
**cloze export** · **topic and weekly exams** · **learner reconstruction notes** ·
**avalanche warnings** · **vacation due-date shifts** · **weekday/hour performance reports** ·
**retire/unretire and pin/unpin controls** · **topic/concept merge** ·
**safe delete with exact confirmation** · **full JSON export/import** ·
**source chunking and multi-source manifests** · **source-cited curriculum/lesson prompts** ·
**source coverage, glossary, and unsupported-claim reports** ·
**automatic curriculum rebuild when source docs change** ·
**dashboard retrieval cockpit, filter/sort ledger, detail drawer, card actions, settings, drag/drop source upload, transcript download, and inline SVG previews** ·
**CLI config/profile/completion/JSON modes** · **local review notifications** ·
**voice selection, push-to-talk hooks, dictation cleanup, and answer readback** ·
**semantic recall experiment, weak-concept bridges, printable weekly sheets, tagged Anki export, and capstone prompts** ·
**14-day review forecast** · **streak tracking** · lapse detection · weakness
ranking with reasons · interleaved retrieval · **learn from your own documents (`--from`)**
including PDFs via local `pdftotext` · **document upload in the web dashboard**
· **session resume after interruption** · **voice mode (`--speak`, macOS)** ·
**separate grader model (`FORGE_GRADER`)** · **Anki export** · full attempt
history · FTS5 lesson recall · knowledge-graph view · web dashboard · terminal
UI · **named learner profiles (`--learner`)** · offline stub mode ·
LLM-with-heuristic-floor grading · failure circuit breaker · one-file backup.

The longer product backlog is in [IDEAS.md](IDEAS.md): 500 scored retrieval
ideas, the selected top 100, and implementation status for each. The selected
top 100 are implemented and covered by the offline suite. The post-top-100
lazy audit and the generated 1000-candidate refinement list live in
[IDEAS.md](IDEAS.md) and [MORE_IDEAS.md](MORE_IDEAS.md).

Deferred, each with its trigger (details in [CLAUDE.md](CLAUDE.md)): Twilio
voice ambush-quizzes (needs an account) · vector embeddings (when FTS5 misses
semantic links) · full FSRS-4.5 scheduling (when ~1k real reviews exist to fit
it) · richer Manim/Matplotlib visual lessons (when the generated SVG concept map
is insufficient) · speech-to-text answers (when whisper.cpp is installed). All
are interfaces to the loop; the loop never changes — that's what keeps this
maintainable.
