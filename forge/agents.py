"""The four Forge agents as LangGraph nodes.

Every node reads its collaborators (llm, memory, ask, emit) from
config["configurable"], so the same graph runs against the terminal,
a test script, or any future interface.
"""
import json
import re

from .llm import Ollama
from .sources import source_citation_block
from .visuals import render_visual_lesson

MASTERY_THRESHOLD = 0.8

# a failure never repeats the same explanation — the Synthesizer re-teaches
# from a genuinely different cognitive angle (encoding variability effect)
ANGLES = [
    "first principles: derive it from the ground up",
    "analogy: map it onto an everyday physical experience",
    "worked example: one concrete end-to-end walk-through",
    "misconceptions: state the wrong intuitions and correct them",
    "spatial: describe it as a picture or diagram in words",
    "visual: provide a compact visual map of parts, links, and flow",
]

SYNTH_CONCEPTS = (
    'Break the topic "{topic}" into 3-5 first-principles concepts ordered from '
    'foundational to advanced. Reply as JSON: '
    '{{"concepts": [{{"name": "...", "summary": "one sentence"}}]}}'
)
SYNTH_LESSON = (
    'Teach the concept "{name}" (part of topic "{topic}"; summary: {summary}). '
    "Teaching angle: {angle}. Under 250 words. Concrete and precise, no filler."
)
QUIZ = (
    'Write 3 active-recall questions on the concept "{name}" based only on this '
    "lesson:\n---\n{lesson}\n---\n"
    "Each question must force reconstruction from memory — no yes/no, no multiple "
    'choice. Reply as JSON: {{"questions": [{{"prompt": "...", '
    '"key_points": ["fact the answer must contain", "..."]}}]}}'
)
GRADE = (
    "Grade this active-recall answer from 0.0 to 1.0.\n"
    "Question: {q}\nRequired key points: {kp}\nStudent answer: {a}\n"
    'Reply as JSON: {{"score": 0.0, "feedback": "one sentence"}}'
)


def _json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


def heuristic_detail(answer: str, key_points: list[str]) -> tuple[float, list[str]]:
    """Deterministic grading: (fraction of key points hit, the ones missed)."""
    if not key_points:
        return 0.0, []
    ans = set(re.findall(r"[a-z0-9]+", answer.lower()))
    missed = []
    for kp in key_points:
        words = [w for w in re.findall(r"[a-z0-9]+", kp.lower()) if len(w) > 2]
        if not (words and sum(w in ans for w in words) / len(words) >= 0.6):
            missed.append(kp)
    return (len(key_points) - len(missed)) / len(key_points), missed


def heuristic_grade(answer: str, key_points: list[str]) -> float:
    return heuristic_detail(answer, key_points)[0]


def grade(llm, question: str, key_points: list[str], answer: str) -> dict:
    """Returns {"score", "missed", "feedback"}. LLM refines the score and adds
    feedback when available; the heuristic is always the floor and never lets a
    dead model open the gate."""
    score, missed = heuristic_detail(answer, key_points)
    feedback = ""
    if isinstance(llm, Ollama):
        try:
            j = _json(llm.ask(GRADE.format(q=question, kp=key_points, a=answer),
                              as_json=True))
            score = min(1.0, max(0.0, float(j["score"])))
            feedback = str(j.get("feedback", ""))[:240]
        except Exception:
            pass  # model down mid-session or bad JSON — heuristic still gates
    return {"score": score, "missed": missed, "feedback": feedback}


def _source_block(state, query: str = "") -> str:
    manifest = state.get("source_manifest") or {}
    if manifest.get("docs"):
        snippets = source_citation_block(
            manifest, query or state.get("current_topic", ""), limit=5)
        return (
            "\nBase everything strictly on this local source manifest. "
            "Use the selected snippets below and cite snippet labels like [S1] "
            "for source-backed claims."
            f"{snippets}"
        )
    src = state.get("source_material") or ""
    # concatenated, never .format()ed — user documents may contain braces
    return f"\nBase everything strictly on this source material:\n---\n{src}\n---" if src else ""


def synthesizer(state, config):
    cfg = config["configurable"]
    llm, emit, memory = cfg["llm"], cfg["emit"], cfg["memory"]
    topic = state["current_topic"]
    out = {}
    if not state.get("concepts"):
        manifest = state.get("source_manifest") or {}
        if manifest.get("docs"):
            source_status = memory.save_source_manifest(topic, manifest)
            if source_status["old_digest"] and source_status["changed"] and memory.load_progress(topic):
                memory.clear_progress(topic)
                emit("[Forge] Source changed; rebuilding the curriculum from the new manifest.")
        saved = memory.load_progress(topic)
        if saved:  # interrupted session — pick up exactly where it stopped
            out["concepts"], out["concept_idx"] = saved
            emit(f"[Forge] Resuming '{topic}' at concept "
                 f"{out['concept_idx'] + 1}/{len(out['concepts'])}.")
        else:
            raw = _json(llm.ask(SYNTH_CONCEPTS.format(topic=topic)
                                + _source_block(state, topic), as_json=True))
            out["concepts"] = raw["concepts"]
            out["concept_idx"] = 0
            memory.save_progress(topic, out["concepts"], 0)
            emit(f"\n== Curriculum for '{topic}' ==")
            for i, c in enumerate(out["concepts"]):
                emit(f"  {i + 1}. {c['name']} — {c['summary']}")
    concepts = out["concepts"] if "concepts" in out else state["concepts"]
    idx = out["concept_idx"] if "concept_idx" in out else state.get("concept_idx", 0)
    angle = ANGLES[state.get("angle", 0) % len(ANGLES)]
    c = concepts[idx]
    missed = state.get("missed") or []
    focus = (f" The learner previously failed to recall: {'; '.join(missed[:6])}. "
             "Address those points head-on." if missed else "")
    lesson = llm.ask(SYNTH_LESSON.format(name=c["name"], topic=topic,
                                         summary=c["summary"], angle=angle)
                     + focus + _source_block(state, f"{c['name']} {c['summary']}"))
    emit(f"\n-- Concept {idx + 1}/{len(concepts)}: {c['name']} [angle: {angle.split(':')[0]}] --")
    emit(lesson)
    out["lesson"] = lesson
    if angle.startswith("visual:"):
        path = render_visual_lesson(topic, c["name"], lesson, missed)
        emit(f"[Visual] Diagram: {path}")
        out["visual_artifact"] = path
    return out


def inquisitor(state, config):
    cfg = config["configurable"]
    llm, emit, ask, memory = cfg["llm"], cfg["emit"], cfg["ask"], cfg["memory"]
    c = state["concepts"][state["concept_idx"]]
    qs = _json(llm.ask(QUIZ.format(name=c["name"], lesson=state["lesson"]),
                       as_json=True))["questions"]
    # interleaving: resurface the historically weakest concept from other topics.
    # It grades and reschedules THAT concept — it never gates the current one,
    # so an old weakness can't trap you on new material.
    weak = memory.weakest(1, exclude_topic=state["current_topic"])
    if weak and memory.lesson_for(weak[0]["topic"], weak[0]["concept"]):
        w = weak[0]
        qs.append({
            "prompt": f"Interleaved review — explain '{w['concept']}' "
                      f"(topic: {w['topic']}) from memory.",
            "key_points": [s.strip() for s in
                           memory.lesson_for(w["topic"], w["concept"]).split(".")
                           if s.strip()][:3],
            "interleaved": {"topic": w["topic"], "concept": w["concept"]},
        })
    emit("\n** Active recall — the lesson is gone. Reconstruct it. **")
    answers = []
    for i, q in enumerate(qs):
        answers.append(ask(f"[{i + 1}/{len(qs)}] {q['prompt']}\n> "))
    return {"questions": qs, "answers": answers}


def evaluator(state, config):
    cfg = config["configurable"]
    emit, memory = cfg["emit"], cfg["memory"]
    grader = cfg.get("grader") or cfg["llm"]
    gate_scores, all_missed, qn = [], [], 0
    for q, a in zip(state["questions"], state["answers"]):
        g = grade(grader, q["prompt"], q["key_points"], a)
        if "interleaved" in q:
            w = q["interleaved"]
            entry = memory.record(w["topic"], w["concept"], g["score"],
                                  missed=g["missed"], feedback=g["feedback"],
                                  answer=a)
            emit(f"[Evaluator] Interleaved recall of '{w['concept']}': {g['score']:.2f} "
                 f"— next review in {entry['next_interval_days']:.1f} day(s).")
            continue
        qn += 1
        gate_scores.append(g["score"])
        all_missed += g["missed"]
        detail = f" — missed: {'; '.join(g['missed'][:3])}" if g["missed"] else ""
        fb = f" {g['feedback']}" if g["feedback"] else ""
        emit(f"[Evaluator] Q{qn}: {g['score']:.2f}{detail}{fb}")
    score = sum(gate_scores) / len(gate_scores)
    passed = score >= MASTERY_THRESHOLD
    out = {"score": score, "missed": [] if passed else all_missed[:6]}
    if passed:
        n_passed = state["concept_idx"] + 1
        prev = state.get("mastery_level", 0.0)
        out["mastery_level"] = (prev * (n_passed - 1) + score) / n_passed
        emit(f"\n[Evaluator] PASS — score {score:.2f}. Mastery advancing.")
    else:
        out["failed_attempts"] = state.get("failed_attempts", 0) + 1
        out["angle"] = state.get("angle", 0) + 1
        emit(f"\n[Evaluator] FAIL — score {score:.2f} < {MASTERY_THRESHOLD}. "
             "Re-teaching from a different angle. No way forward but through.")
    return out


def route_after_evaluation(state) -> str:
    if state["score"] >= MASTERY_THRESHOLD:
        return "advance"
    cap = state.get("max_failed_attempts", 0)
    if cap and state.get("failed_attempts", 0) >= cap:
        return "halt"  # circuit breaker for scripted/test runs; 0 = mandate mode
    return "retry"


def orchestrator(state, config):
    """Gatekeeper passed — commit to long-term memory, schedule the review,
    move to the next concept or declare topic mastery."""
    cfg = config["configurable"]
    memory, emit = cfg["memory"], cfg["emit"]
    c = state["concepts"][state["concept_idx"]]
    gate_answers = [a for q, a in zip(state.get("questions", []), state.get("answers", []))
                    if "interleaved" not in q]
    entry = memory.record(state["current_topic"], c["name"], state["score"],
                          lesson=state["lesson"], answer="\n".join(gate_answers))
    emit(f"[Orchestrator] '{c['name']}' scheduled for review in "
         f"{entry['next_interval_days']:.1f} day(s).")
    if cfg.get("correct_lessons"):
        note = cfg["ask"]("Optional: write your corrected reconstruction to store as the lesson, or press Enter.\n> ")
        if note.strip():
            memory.set_learner_note(state["current_topic"], c["name"], note,
                                    replace_lesson=True)
            emit("[Forge] Stored your reconstruction as the lesson.")
    nxt = state["concept_idx"] + 1
    out = {"spaced_repetition_log": [entry], "concept_idx": nxt, "angle": 0,
           "missed": []}
    if nxt >= len(state["concepts"]):
        out["mastery"] = True
        out["done"] = True
        memory.clear_progress(state["current_topic"])
        emit(f"\n== TOPIC MASTERED: {state['current_topic']} "
             f"(mastery level {state.get('mastery_level', 0):.2f}) ==")
    else:
        memory.save_progress(state["current_topic"], state["concepts"], nxt)
    return out


def route_after_orchestration(state) -> str:
    return "done" if state.get("done") else "next"
