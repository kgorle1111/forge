"""LangGraph wiring: the closed mastery loop.

synthesizer -> inquisitor -> evaluator --fail--> synthesizer   (the trap)
                                       --pass--> orchestrator --more concepts--> synthesizer
                                                              --all mastered---> END
"""
import time
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .agents import (QUIZ, _json, evaluator, grade, inquisitor, orchestrator,
                     route_after_evaluation, route_after_orchestration,
                     synthesizer)
from .llm import get_grader, get_llm
from .memory import DAY, Memory
from .sources import source_manifest as build_source_manifest
from .state import ForgeState


def build_graph():
    g = StateGraph(ForgeState)
    g.add_node("synthesizer", synthesizer)
    g.add_node("inquisitor", inquisitor)
    g.add_node("evaluator", evaluator)
    g.add_node("orchestrator", orchestrator)
    g.set_entry_point("synthesizer")
    g.add_edge("synthesizer", "inquisitor")
    g.add_edge("inquisitor", "evaluator")
    g.add_conditional_edges("evaluator", route_after_evaluation,
                            {"retry": "synthesizer", "advance": "orchestrator",
                             "halt": END})
    g.add_conditional_edges("orchestrator", route_after_orchestration,
                            {"next": "synthesizer", "done": END})
    return g.compile(checkpointer=MemorySaver())


def run_topic(topic: str, ask, emit=print, llm=None, memory=None,
              max_failed_attempts: int = 0, source: str = "",
              grader=None, correct_lessons: bool = False,
              source_manifest: dict | None = None) -> ForgeState:
    graph = build_graph()
    config = {
        "configurable": {
            # resume lives in Memory.progress, not the checkpointer -> fresh thread
            "thread_id": str(uuid.uuid4()),
            "llm": llm or get_llm(),
            "grader": grader if grader is not None else (llm or get_grader()),
            "memory": memory or Memory(),
            "ask": ask,
            "emit": emit,
            "correct_lessons": correct_lessons,
        },
        "recursion_limit": 1000,  # the loop is the product; don't let langgraph end it
    }
    if source and source_manifest is None:
        source_manifest = build_source_manifest([("source", source)])
    initial: ForgeState = {
        "current_topic": topic,
        "concept_idx": 0,
        "angle": 0,
        "missed": [],
        "source_material": source,
        "source_manifest": source_manifest or {},
        "failed_attempts": 0,
        "max_failed_attempts": max_failed_attempts,
        "mastery_level": 0.0,
        "mastery": False,
        "spaced_repetition_log": [],
    }
    return graph.invoke(initial, config)


def run_review(ask, emit=print, llm=None, memory=None, grader=None,
               limit: int | None = None, days: int | None = None,
               hard: bool = False, confidence: bool = False,
               followup: bool = True) -> int:
    """Quiz every concept the scheduler says is due; returns count reviewed.
    Shared by the CLI and the web dashboard."""
    memory = memory or Memory()
    grader = grader if grader is not None else (llm or get_grader())
    llm = llm or get_llm()
    due = memory.due_queue(limit=limit, days=days)
    if not due:
        nxt = memory.db.execute(
            "SELECT MIN(due) AS d FROM cards WHERE suspended=0 AND retired=0"
        ).fetchone()["d"]
        wait = (f" Next review in {(nxt - time.time()) / DAY:.1f} day(s)."
                if nxt else " Learn a topic first.")
        emit(f"[Forge] Nothing due.{wait}")
        return 0
    emit(f"[Forge] {len(due)} concept(s) due for review.")
    for card in due:
        lesson = memory.lesson_for(card["topic"], card["concept"])
        qs = _json(llm.ask(QUIZ.format(name=card["concept"], lesson=lesson),
                           as_json=True))["questions"][:2]
        heading = f"\n-- Review: {card['concept']} --" if hard else (
            f"\n-- Review: {card['concept']} ({card['topic']}) --")
        emit(heading)
        scores, all_missed, feedbacks, answers, confidences = [], [], [], [], []
        for i, q in enumerate(qs):
            a = ask(f"[{i + 1}/{len(qs)}] {q['prompt']}\n> ")
            conf = _ask_confidence(ask) if confidence else None
            g = grade(grader, q["prompt"], q["key_points"], a)
            if followup and 0.0 < g["score"] < 0.8 and g["missed"]:
                more = ask("Close. Retrieve the missing piece now, without looking.\n> ")
                g2 = grade(grader, q["prompt"], q["key_points"], f"{a}\n{more}")
                if g2["score"] > g["score"]:
                    a = f"{a}\n{more}"
                    g = g2
            scores.append(g["score"])
            all_missed += g["missed"]
            feedbacks.append(g["feedback"])
            answers.append(a)
            if conf is not None:
                confidences.append(conf)
            detail = f" — missed: {'; '.join(g['missed'][:3])}" if g["missed"] else ""
            fb = f" {g['feedback']}" if g["feedback"] else ""
            emit(f"[Evaluator] Q{i + 1}: {g['score']:.2f}{detail}{fb}")
        s = sum(scores) / len(scores)
        entry = memory.record(card["topic"], card["concept"], s, lesson=lesson,
                              missed=all_missed[:10],
                              feedback=" ".join(f for f in feedbacks if f),
                              confidence=(sum(confidences) / len(confidences)
                                          if confidences else None),
                              answer="\n".join(answers))
        verdict = "retained" if s >= 0.8 else "lapsed — interval reset"
        emit(f"[Forge] score {s:.2f} ({verdict}); next in "
             f"{entry['next_interval_days']:.1f} day(s).")
    return len(due)


def _ask_confidence(ask) -> float | None:
    raw = ask("Confidence 0-5 before grading?\n> ").strip()
    try:
        return max(0.0, min(5.0, float(raw)))
    except ValueError:
        return None
