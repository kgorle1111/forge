"""End-to-end checks for The Forge. Runs fully offline against the Stub LLM.

    .venv/bin/python tests/test_forge.py
"""
import base64
import contextlib
import http.client
import io
import json
import os
import plistlib
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["FORGE_STUB"] = "1"

from forge.agents import heuristic_grade
from forge.graph import run_review, run_topic
from forge.graphview import export
from forge.llm import Stub, get_llm
from forge.memory import DAY, Memory, retrievability
from forge.profiles import select_profile
from forge.sources import (chunk_source, glossary_terms, read_source,
                           read_uploaded_source, relevant_chunks,
                           source_citation_block,
                           source_manifest as build_source_manifest,
                           unsupported_terms)


class Spy(Stub):
    """Stub that records every prompt it is asked."""
    def __init__(self):
        self.prompts = []

    def ask(self, prompt, as_json=False):
        self.prompts.append(prompt)
        return super().ask(prompt, as_json)


def test_heuristic_grade():
    assert heuristic_grade("alpha beta gamma", ["alpha", "beta", "gamma"]) == 1.0
    assert heuristic_grade("no idea", ["alpha", "beta"]) == 0.0
    assert heuristic_grade("only alpha here", ["alpha", "beta"]) == 0.5
    assert heuristic_grade("anything", []) == 0.0


def test_sm2_scheduler():
    m = Memory(":memory:")
    now = 1_000_000.0
    e1 = m.record("t", "c", 1.0, lesson="l", now=now)
    assert e1["next_interval_days"] == 1.0
    e2 = m.record("t", "c", 1.0, now=now)
    assert e2["next_interval_days"] == 6.0
    e3 = m.record("t", "c", 1.0, now=now)
    assert e3["next_interval_days"] > 6.0  # interval * ease: the 10-20% rule growth
    lapse = m.record("t", "c", 0.2, now=now)
    assert lapse["next_interval_days"] == 1.0  # failure resets the interval
    card = m.due_cards(now + 2 * DAY)[0]
    assert card["lapses"] == 1 and card["reps"] == 0
    assert m.lesson_for("t", "c") == "l"


def test_mastery_loop_traps_until_pass():
    assert isinstance(get_llm(), Stub)
    m = Memory(":memory:")
    log, calls = [], []

    def ask(prompt):
        calls.append(prompt)
        # flunk the entire first quiz, then answer every question correctly
        return "no idea" if len(calls) <= 2 else "remember alpha beta gamma"

    final = run_topic("quantum tunneling", ask=ask, emit=log.append,
                      llm=Stub(), memory=m)

    assert final["mastery"] is True
    assert final["failed_attempts"] == 1          # trapped once, re-taught, escaped
    assert final["mastery_level"] >= 0.8
    assert len(final["spaced_repetition_log"]) == 3  # one entry per concept
    assert len(m.stats()) == 3                     # all concepts committed to memory
    assert all(r["due"] > 0 for r in m.stats())
    joined = "\n".join(log)
    assert "FAIL" in joined and "different angle" in joined
    assert "TOPIC MASTERED" in joined
    # the re-teach used a different angle than the first attempt
    assert "[angle: first principles]" in joined and "[angle: analogy]" in joined


def test_interleaving_pulls_old_weakness():
    m = Memory(":memory:")
    m.record("old topic", "old concept", 0.4, lesson="the old lesson text")  # weak card
    prompts = []

    def ask(prompt):
        prompts.append(prompt)
        return "remember alpha beta gamma"  # right for new topic, WRONG for old one

    final = run_topic("new topic", ask=ask, emit=lambda *_: None, llm=Stub(), memory=m)
    # a failed interleaved recall must never block mastery of the new topic...
    assert final["mastery"] is True
    assert any("Interleaved review" in p and "old concept" in p for p in prompts)
    # ...but it must have re-graded and rescheduled the old concept itself
    old = next(r for r in m.stats() if r["concept"] == "old concept")
    assert old["n_attempts"] >= 2 and old["lapses"] >= 1


def test_graph_export():
    m = Memory(":memory:")
    m.record("t", "c", 1.0, lesson="l")
    path = os.path.join(tempfile.mkdtemp(), "g.html")
    export(path, memory=m)
    html = open(path).read()
    assert "canvas" in html and '"label": "c"' in html


def test_retrievability_forecast_streak():
    now = 1_000_000.0
    assert retrievability(0, now) is None                       # unreviewed card
    assert abs(retrievability(10.0, now, now) - 0.9) < 1e-9     # exactly on time
    assert retrievability(10.0, now + 5 * DAY, now) > 0.9       # reviewed early
    assert retrievability(10.0, now - 10 * DAY, now) < 0.85     # overdue decays
    m = Memory(":memory:")
    m.record("t", "a", 1.0, now=now)          # due day 1
    m.record("t", "b", 1.0, now=now)
    m.record("t", "b", 1.0, now=now)          # due day 6
    fc = m.forecast(days=14, now=now)
    assert fc[1] == 1 and fc[6] == 1 and sum(fc) == 2
    m.record("t", "c", 1.0, now=now - DAY)    # attempt yesterday too
    assert m.streak(now=now) == 2             # yesterday + today


def test_retrieval_surfaces_rank_and_explain_cards():
    m = Memory(":memory:")
    now = 1_000_000.0
    m.record("math", "fractions", 1.0, lesson="fractions split wholes", now=now)
    m.record("math", "decimals", 1.0, lesson="decimals are place value", now=now)
    m.record("math", "decimals", 1.0, now=now)
    m.record("physics", "force", 0.2, lesson="force changes motion", now=now)
    due = m.due_queue(now=now + 2 * DAY)
    assert due[0]["concept"] == "force"
    assert due[0]["risk"] > due[-1]["risk"]
    assert "lapse" in " ".join(due[0]["reasons"])
    weak = m.weak_cards(2, now=now + 2 * DAY)
    assert weak[0]["concept"] == "force"
    topics = {r["topic"]: r for r in m.topic_summaries(now=now + 2 * DAY)}
    assert topics["physics"]["weak"] == 1
    assert topics["math"]["concepts"] == 2
    detail = m.card_detail("math", "fractions", now=now + 2 * DAY)
    assert detail["lesson"] == "fractions split wholes"
    assert detail["attempts"][0]["score"] == 1.0
    assert m.search_lessons("place", 1)[0]["concept"] == "decimals"
    debt = m.review_debt(now=now + 2 * DAY)
    assert debt["due"] == 2 and debt["peak_risk"] >= due[0]["risk"]


def test_card_mutations_are_safe_and_review_aware():
    m = Memory(":memory:")
    now = 1_000_000.0
    m.record("old topic", "fragile", 0.2, lesson="fragile lesson", now=now - 2 * DAY)
    m.record("other topic", "fragile", 1.0, lesson="collision", now=now)
    m.save_progress("old topic", [{"name": "fragile", "summary": "s"}], 0)
    try:
        m.rename_topic("old topic", "other topic")
        assert False, "rename_topic allowed a concept collision"
    except ValueError:
        pass
    out = m.rename_topic("old topic", "new topic")
    assert out["cards"] == 1
    assert m.lesson_for("new topic", "fragile") == "fragile lesson"
    assert m.load_progress("new topic")[0][0]["name"] == "fragile"
    m.rename_concept("new topic", "fragile", "durable")
    assert m.lesson_for("new topic", "durable") == "fragile lesson"
    assert m.load_progress("new topic")[0][0]["name"] == "durable"
    assert m.due_queue(now=now)[0]["concept"] == "durable"
    m.suspend_card("new topic", "durable")
    assert m.due_queue(now=now) == []
    assert m.forecast(days=3, now=now)[0] == 0
    m.unsuspend_card("new topic", "durable")
    assert m.due_queue(now=now)
    m.suspend_card("new topic", "durable")
    d = m.due_now("new topic", "durable", now=now)
    assert d["suspended"] == 0 and d["due"] == now
    assert m.stuck_cards(now=now)[0]["concept"] == "durable"
    transcript = m.transcript()
    assert "Forge Retrieval Transcript" in transcript and "new topic / durable" in transcript


def test_attempt_metadata_prompts_cloze_notes_and_calibration():
    m = Memory(":memory:")
    now = 1_000_000.0
    m.record("topic", "concept", 0.4, lesson="alpha beta gamma delta",
             now=now, missed=["beta", "gamma"], feedback="tighten it",
             confidence=4, answer="alpha")
    m.record("other", "weak", 0.2, lesson="weak lesson", now=now)
    m.record("topic", "concept", 1.0, now=now + DAY, confidence=5,
             answer="alpha beta gamma")
    detail = m.card_detail("topic", "concept", now=now + DAY)
    assert detail["best_answer"] == "alpha beta gamma"
    assert detail["attempts"][0]["confidence"] == 5
    missed = m.missed_points("topic", "concept")
    assert missed[0]["point"] == "beta" and missed[0]["count"] == 1
    cal = m.calibration()
    assert cal["n"] == 2 and cal["avg_confidence"] == 4.5
    prompts = m.retrieval_prompts("topic", "concept")
    kinds = {p["kind"] for p in prompts}
    assert {"counterexample", "compare", "breaks", "one_sentence", "key_points"} <= kinds
    cloze = m.cloze_cards("topic")
    assert cloze and "{{c1::" in cloze[0]["cloze"]
    m.note_from_best_answer("topic", "concept", replace_lesson=True)
    assert m.learner_note("topic", "concept") == "alpha beta gamma"
    assert m.lesson_for("topic", "concept") == "alpha beta gamma"


def test_scheduler_lifecycle_merge_delete_and_json_roundtrip():
    m = Memory(":memory:")
    now = 1_000_000.0
    for i in range(25):
        m.record("storm", f"card {i}", 1.0, lesson=f"lesson {i}", now=now - DAY)
    warning = m.avalanche_warning(days=3, daily_budget=5, now=now)
    assert warning["warning"] is True and warning["peak_count"] >= 25
    m.shift_due(2, topic="storm", concept="card 0")
    assert all(r["concept"] != "card 0" for r in m.due_queue(now=now))
    m.pin_card("storm", "card 0", now=now)
    assert m.due_queue(limit=1, now=now)[0]["concept"] == "card 0"
    m.unpin_card("storm", "card 0")
    m.retire_card("storm", "card 1")
    assert all(r["concept"] != "card 1" for r in m.due_queue(now=now))
    m.unretire_card("storm", "card 1")
    assert any(r["concept"] == "card 1" for r in m.due_queue(now=now))
    assert m.performance_by_weekday() and m.performance_by_hour()

    m.record("merge old", "same", 0.4, lesson="old same", now=now)
    m.record("merge new", "same", 1.0, lesson="new same", now=now)
    out = m.merge_topic("merge old", "merge new")
    assert out["merged"] == 1
    assert m.card_detail("merge new", "same")["n_attempts"] == 2
    m.record("merge new", "other", 1.0, lesson="other lesson", now=now)
    m.merge_concept("merge new", "other", "same")
    assert m.card_detail("merge new", "same")["n_attempts"] == 3
    try:
        m.delete_concept("merge new", "same", "wrong")
        assert False, "delete_concept accepted wrong confirmation"
    except ValueError:
        pass
    assert m.delete_concept("merge new", "same", "merge new/same") == 1
    try:
        m.delete_topic("storm", "wrong")
        assert False, "delete_topic accepted wrong confirmation"
    except ValueError:
        pass
    assert m.delete_topic("storm", "storm") >= 24

    m.save_source_manifest("source only", build_source_manifest([
        ("source.md", "retrieval glossary alpha beta"),
    ]))
    exported = m.export_json()
    assert exported["source_docs"]
    imported = Memory(":memory:")
    result = imported.import_json(exported)
    assert result["cards"] == len(exported["cards"])
    assert len(imported.stats()) == len(exported["cards"])
    assert len(imported.attempt_timeline(1000)) == len(exported["attempts"])
    assert result["source_docs"] == 1
    assert imported.source_manifest("source only")["docs"][0]["name"] == "source.md"


def test_followup_retrieval_can_recover_partial_review():
    m = Memory(":memory:")
    now = 1_000_000.0
    m.record("topic", "concept", 1.0, lesson="alpha beta gamma", now=now - 2 * DAY)
    prompts = []

    def ask(prompt):
        prompts.append(prompt)
        if prompt.startswith("Close."):
            return "beta gamma"
        return "alpha"

    reviewed = run_review(ask=ask, emit=lambda *_: None, llm=Stub(), memory=m,
                          limit=1, followup=True)
    assert reviewed == 1
    assert any(p.startswith("Close.") for p in prompts)
    last = m.attempt_timeline(1)[0]
    assert last["score"] >= 0.8
    assert "beta gamma" in last["answer"]


def test_correct_lesson_stores_learner_reconstruction():
    m = Memory(":memory:")

    def ask(prompt):
        if prompt.startswith("Optional:"):
            return "learner corrected reconstruction"
        return "remember alpha beta gamma"

    final = run_topic("correctable", ask=ask, emit=lambda *_: None,
                      llm=Stub(), memory=m, correct_lessons=True)
    assert final["mastery"] is True
    first = m.stats()[0]
    assert m.learner_note(first["topic"], first["concept"]) == "learner corrected reconstruction"
    assert m.lesson_for(first["topic"], first["concept"]) == "learner corrected reconstruction"


def test_resume_after_interruption():
    m = Memory(":memory:")
    n = {"i": 0}

    def ask(prompt):
        n["i"] += 1
        if n["i"] > 2:                        # concept 1 passed, then walk away
            raise KeyboardInterrupt
        return "remember alpha beta gamma"

    try:
        run_topic("resumable", ask=ask, emit=lambda *_: None, llm=Stub(), memory=m)
        assert False, "interrupt did not propagate"
    except KeyboardInterrupt:
        pass
    assert m.load_progress("resumable")[1] == 1   # parked at concept 2
    log = []
    final = run_topic("resumable", ask=lambda p: "remember alpha beta gamma",
                      emit=log.append, llm=Stub(), memory=m)
    assert final["mastery"] is True
    assert any("Resuming 'resumable' at concept 2/3" in l for l in log)
    assert m.load_progress("resumable") is None   # progress cleared on mastery


def test_feedback_and_targeted_reteach():
    m = Memory(":memory:")
    spy, log, calls = Spy(), [], []

    def ask(prompt):
        calls.append(prompt)
        return "no idea" if len(calls) <= 2 else "remember alpha beta gamma"

    final = run_topic("feedback topic", ask=ask, emit=log.append, llm=spy, memory=m)
    assert final["mastery"] is True
    # per-question verdicts name exactly what was missed
    assert any("Q1: 0.00 — missed: alpha" in l for l in log)
    # the re-teach lesson prompt targets the missed key points head-on
    lesson_prompts = [p for p in spy.prompts if p.startswith("Teach the concept")]
    assert any("previously failed to recall" in p and "alpha" in p
               for p in lesson_prompts)


def test_source_material_grounds_curriculum():
    spy = Spy()
    final = run_topic("sourced", ask=lambda p: "remember alpha beta gamma",
                      emit=lambda *_: None, llm=spy, memory=Memory(":memory:"),
                      source="THE SACRED SOURCE {with braces}")
    assert final["mastery"] is True
    assert any("THE SACRED SOURCE {with braces}" in p for p in spy.prompts
               if "concepts" in p)  # curriculum prompt carries the document
    lessons = [p for p in spy.prompts if p.startswith("Teach the concept")]
    assert any("[S1]" in p and "cite snippet labels" in p for p in lessons)


def test_source_ingestion_text_pdf_and_upload():
    root = tempfile.mkdtemp()
    txt = os.path.join(root, "notes.md")
    open(txt, "w", encoding="utf-8").write("TEXT SOURCE {with braces}")
    assert read_source(txt) == "TEXT SOURCE {with braces}"

    bindir = os.path.join(root, "bin")
    os.mkdir(bindir)
    fake = os.path.join(bindir, "pdftotext")
    open(fake, "w", encoding="utf-8").write("#!/bin/sh\nprintf 'PDF SOURCE {local}\\n'\n")
    os.chmod(fake, 0o755)
    pdf = os.path.join(root, "paper.pdf")
    open(pdf, "wb").write(b"%PDF-1.4\n")
    old_path = os.environ.get("PATH", "")
    try:
        os.environ["PATH"] = bindir + os.pathsep + old_path
        assert read_source(pdf) == "PDF SOURCE {local}"
    finally:
        os.environ["PATH"] = old_path

    upload = base64.b64encode(b"UPLOAD SOURCE {browser}").decode()
    assert read_uploaded_source("upload.txt", upload) == "UPLOAD SOURCE {browser}"


def test_source_manifest_chunking_ranking_and_glossary():
    long = "Alpha retrieval strengthens memory. " * 80 + "Rareterm anchors transfer."
    chunks = chunk_source(long, size=180, overlap=30)
    assert len(chunks) > 3 and chunks[0]["label"] == "S1"
    manifest = build_source_manifest([
        ("retrieval.md", long),
        ("calibration.md", "Calibration links confidence to recall accuracy."),
    ])
    labels = [c["label"] for d in manifest["docs"] for c in d["chunks"]]
    assert len(labels) == len(set(labels)) and len(manifest["docs"]) == 2
    ranked = relevant_chunks(manifest, "Rareterm transfer", 2)
    assert ranked[0]["source"] == "retrieval.md"
    assert "[S" in source_citation_block(manifest, "confidence recall")
    glossary = glossary_terms("alpha alpha beta retrieval", 3)
    assert glossary[0] == {"term": "alpha", "count": 2}
    assert unsupported_terms("alpha delta epsilon", "alpha beta")[:2] == ["delta", "epsilon"]


def test_memory_source_manifest_reports_and_rebuilds_on_change():
    m = Memory(":memory:")
    old = build_source_manifest([("notes.md", "old retrieval source")])
    status = m.save_source_manifest("sourced topic", old)
    assert status["docs"] == 1 and status["changed"] is True
    m.save_progress("sourced topic", [{"name": "stale", "summary": "old"}], 0)
    new = build_source_manifest([("notes.md", "new alpha beta gamma retrieval source")])
    log = []
    final = run_topic("sourced topic", ask=lambda p: "remember alpha beta gamma",
                      emit=log.append, llm=Stub(), memory=m,
                      source_manifest=new)
    assert final["mastery"] is True
    assert any("Source changed" in x for x in log)
    assert not any("Resuming 'sourced topic'" in x for x in log)
    assert m.source_manifest("sourced topic")["digest"] == new["digest"]

    coverage = m.source_coverage("sourced topic")
    assert coverage and coverage[0]["snippets"] >= 1
    unsupported = m.unsupported_claims("sourced topic")
    assert unsupported and "remember" in unsupported[0]["terms"]
    glossary = m.source_glossary("sourced topic")
    assert any(r["term"] == "retrieval" for r in glossary)
    snippets = m.source_snippets("sourced topic", "retrieval", 1)
    assert snippets and snippets[0]["label"].startswith("S")


def test_learner_profile_selects_separate_db():
    old_home, old_db, old_learner = (os.environ.get("HOME"),
                                    os.environ.get("FORGE_DB"),
                                    os.environ.get("FORGE_LEARNER"))
    try:
        os.environ["HOME"] = tempfile.mkdtemp()
        path = select_profile("student-1")
        assert path.endswith(os.path.join(".forge", "student-1.db"))
        assert os.environ["FORGE_DB"] == path
        assert os.environ["FORGE_LEARNER"] == "student-1"
        m = Memory()
        m.record("profile topic", "profile concept", 1.0)
        assert len(m.stats()) == 1
    finally:
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home
        if old_db is None:
            os.environ.pop("FORGE_DB", None)
        else:
            os.environ["FORGE_DB"] = old_db
        if old_learner is None:
            os.environ.pop("FORGE_LEARNER", None)
        else:
            os.environ["FORGE_LEARNER"] = old_learner


def test_visual_angle_writes_svg_artifact():
    old_dir = os.environ.get("FORGE_ARTIFACT_DIR")
    try:
        outdir = tempfile.mkdtemp()
        os.environ["FORGE_ARTIFACT_DIR"] = outdir
        calls, log = [], []

        def ask(prompt):
            calls.append(prompt)
            return "no idea" if len(calls) <= 10 else "remember alpha beta gamma"

        final = run_topic("visual topic", ask=ask, emit=log.append,
                          llm=Stub(), memory=Memory(":memory:"))
        assert final["mastery"] is True
        visual = [l for l in log if l.startswith("[Visual] Diagram: ")]
        assert visual
        path = visual[0].split(": ", 1)[1]
        assert os.path.exists(path)
        assert "<svg" in open(path, encoding="utf-8").read()
    finally:
        if old_dir is None:
            os.environ.pop("FORGE_ARTIFACT_DIR", None)
        else:
            os.environ["FORGE_ARTIFACT_DIR"] = old_dir


def test_anki_export():
    from forge import cli
    os.environ["FORGE_DB"] = os.path.join(tempfile.mkdtemp(), "exp.db")
    Memory().record("topic x", "concept y", 1.0, lesson="line one\nline two")
    path = os.path.join(tempfile.mkdtemp(), "anki.tsv")
    cli.main(["export", path])
    body = open(path, encoding="utf-8").read()
    assert "#separator:tab" in body
    assert "concept y — reconstruct from memory (topic x)\tline one<br>line two" in body


def test_cli_retrieval_commands_and_review_limit():
    from forge import cli

    old_db = os.environ.get("FORGE_DB")
    try:
        os.environ["FORGE_DB"] = os.path.join(tempfile.mkdtemp(), "cli.db")
        m = Memory()
        now = time.time() - 3 * DAY
        m.record("cli topic", "fragile", 0.2, lesson="fragile lesson alpha", now=now)
        m.record("cli topic", "steady", 1.0, lesson="steady lesson beta", now=now)
        m.record("future topic", "soon", 1.0, lesson="soon lesson gamma", now=time.time())

        def run(args):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cli.main(args)
            return buf.getvalue()

        assert "fragile" in run(["due"])
        assert "lapse" in run(["weak"])
        assert "cli topic" in run(["topics"])
        assert "fragile lesson" in run(["inspect", "cli topic", "fragile"])
        assert "steady" in run(["search", "beta"])
        assert "peak risk" in run(["debt"])
        assert "Top retrieval targets" in run(["today"])
        assert "fragile" in run(["stuck"])
        transcript_path = os.path.join(tempfile.mkdtemp(), "transcript.md")
        assert "wrote transcript" in run(["transcript", transcript_path])
        assert "Forge Retrieval Transcript" in open(transcript_path, encoding="utf-8").read()
        before = len(Memory().attempt_timeline(20))
        old_stdin = sys.stdin
        try:
            sys.stdin = io.StringIO("remember alpha beta gamma\nremember alpha beta gamma\n")
            run(["practice", "--limit", "1"])
        finally:
            sys.stdin = old_stdin
        after = len(Memory().attempt_timeline(20))
        assert after == before + 1
        before = after
        old_stdin = sys.stdin
        try:
            sys.stdin = io.StringIO("remember alpha beta gamma\nremember alpha beta gamma\n")
            run(["cram", "--days", "7", "--limit", "1"])
        finally:
            sys.stdin = old_stdin
        assert len(Memory().attempt_timeline(20)) == before + 1
    finally:
        if old_db is None:
            os.environ.pop("FORGE_DB", None)
        else:
            os.environ["FORGE_DB"] = old_db


def test_cli_backup_restore_and_card_actions():
    from forge import cli

    old_db = os.environ.get("FORGE_DB")
    try:
        root = tempfile.mkdtemp()
        os.environ["FORGE_DB"] = os.path.join(root, "actions.db")

        def run(args):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cli.main(args)
            return buf.getvalue()

        Memory().record("topic a", "concept a", 1.0, lesson="alpha lesson")
        backup = os.path.join(root, "backup.db")
        assert "backed up" in run(["backup", backup])
        Memory().record("topic b", "concept b", 1.0, lesson="beta lesson")
        assert len(Memory().stats()) == 2
        assert "restored" in run(["restore", backup, "--confirm", backup])
        assert [r["topic"] for r in Memory().stats()] == ["topic a"]

        assert "renamed topic" in run(["rename-topic", "topic a", "topic z"])
        assert "renamed concept" in run(["rename-concept", "topic z", "concept a", "concept z"])
        assert Memory().lesson_for("topic z", "concept z") == "alpha lesson"
        assert "suspended" in run(["suspend", "topic z", "concept z"])
        assert Memory().due_queue(days=30) == []
        assert "unsuspended" in run(["unsuspend", "topic z", "concept z"])
        assert "due now" in run(["due-now", "topic z", "concept z"])
        assert Memory().due_queue()
    finally:
        if old_db is None:
            os.environ.pop("FORGE_DB", None)
        else:
            os.environ["FORGE_DB"] = old_db


def test_cli_retrieval_metadata_and_prompt_tools():
    from forge import cli

    old_db = os.environ.get("FORGE_DB")
    try:
        root = tempfile.mkdtemp()
        os.environ["FORGE_DB"] = os.path.join(root, "meta.db")
        m = Memory()
        now = time.time() - 3 * DAY
        m.record("meta topic", "meta concept", 0.2,
                 lesson="alpha beta gamma delta", now=now)
        m.record("meta topic", "second concept", 1.0,
                 lesson="epsilon zeta eta theta", now=now)
        m.save_source_manifest("meta topic", build_source_manifest([
            ("notes.md", "alpha beta gamma retrieval strengthens durable memory."),
            ("more.md", "epsilon zeta theta calibration."),
        ]))

        def run(args, stdin=""):
            buf = io.StringIO()
            old_stdin = sys.stdin
            try:
                sys.stdin = io.StringIO(stdin)
                with contextlib.redirect_stdout(buf):
                    cli.main(args)
            finally:
                sys.stdin = old_stdin
            return buf.getvalue()

        out = run(["review", "--limit", "1", "--hard", "--confidence", "--no-followup"],
                  "alpha\n2\nalpha\n3\n")
        assert "-- Review: meta concept --" in out
        assert "(meta topic)" not in out
        assert "avg confidence" in run(["calibration"])
        assert "beta" in run(["missed"])
        assert "[counterexample]" in run(["prompts", "meta topic", "meta concept"])
        cloze_path = os.path.join(root, "cloze.tsv")
        assert "cloze cards" in run(["cloze", cloze_path])
        assert "{{c1::" in open(cloze_path, encoding="utf-8").read()
        assert "learner note stored" in run(["note", "meta topic", "meta concept",
                                            "--set", "my reconstruction",
                                            "--replace-lesson"])
        assert "my reconstruction" in run(["note", "meta topic", "meta concept"])
        assert Memory().lesson_for("meta topic", "meta concept") == "my reconstruction"
        Memory().record("meta topic", "meta concept", 1.0,
                        answer="passing reconstruction")
        assert "promoted" in run(["promote-answer", "meta topic", "meta concept"])
        source = os.path.join(root, "source.md")
        open(source, "w", encoding="utf-8").write("Retrieval strengthens durable memory.")
        assert "retrieval" in run(["source-checklist", source])
        assert "notes.md" in run(["source-manifest", "meta topic"])
        assert "[S1]" in run(["source-chunks", "meta topic", "--query", "alpha"])
        assert "cover" in run(["source-coverage", "meta topic"])
        assert "eta" in run(["unsupported", "meta topic", "second concept"])
        assert "retrieval" in run(["source-glossary", "meta topic"])

        # daily5, exam, and weekly are all actual retrieval sessions.
        before = len(Memory().attempt_timeline(100))
        run(["daily5", "--no-followup"], "remember alpha beta gamma\nremember alpha beta gamma\n")
        run(["exam", "meta topic", "--no-followup"],
            "remember alpha beta gamma\nremember alpha beta gamma\n"
            "remember alpha beta gamma\nremember alpha beta gamma\n")
        run(["weekly", "--limit", "1", "--no-followup"],
            "remember alpha beta gamma\nremember alpha beta gamma\n")
        assert len(Memory().attempt_timeline(100)) >= before + 3
    finally:
        if old_db is None:
            os.environ.pop("FORGE_DB", None)
        else:
            os.environ["FORGE_DB"] = old_db


def test_cli_scheduler_lifecycle_and_json_commands():
    from forge import cli

    old_db = os.environ.get("FORGE_DB")
    try:
        root = tempfile.mkdtemp()
        os.environ["FORGE_DB"] = os.path.join(root, "schedule.db")
        now = time.time() - 2 * DAY
        m = Memory()
        m.record("sched", "one", 1.0, lesson="one lesson", now=now)
        m.record("sched", "two", 1.0, lesson="two lesson", now=now)
        m.record("target", "one", 1.0, lesson="target lesson", now=now)

        def run(args):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cli.main(args)
            return buf.getvalue()

        assert "steady" in run(["avalanche", "--days", "3", "--budget", "10"])
        assert "shifted" in run(["vacation", "2", "--topic", "sched", "--concept", "two"])
        assert "weekday" in run(["performance"])
        assert "hour" in run(["performance", "--by", "hour"])
        assert "retired" in run(["retire", "sched", "one"])
        assert Memory().card_detail("sched", "one")["retired"] == 1
        assert "unretired" in run(["unretire", "sched", "one"])
        assert "pinned" in run(["pin", "sched", "one"])
        assert Memory().due_queue(limit=1)[0]["concept"] == "one"
        assert "unpinned" in run(["unpin", "sched", "one"])
        assert "merged" in run(["merge-topic", "sched", "target"])
        assert "merged concept" in run(["merge-concept", "target", "two", "one"])
        json_path = os.path.join(root, "forge.json")
        assert "exported JSON" in run(["json-export", json_path])
        assert "deleted concept" in run(["delete-concept", "target", "one",
                                         "--confirm", "target/one"])
        assert not Memory().stats()
        assert "imported" in run(["json-import", json_path])
        assert Memory().stats()
        assert "deleted topic" in run(["delete-topic", "target", "--confirm", "target"])
        assert not Memory().stats()
    finally:
        if old_db is None:
            os.environ.pop("FORGE_DB", None)
        else:
            os.environ["FORGE_DB"] = old_db


def test_cli_config_profiles_voice_and_final_retrieval_tools():
    from forge import cli

    old_home, old_db, old_learner, old_dictation = (
        os.environ.get("HOME"), os.environ.get("FORGE_DB"),
        os.environ.get("FORGE_LEARNER"), os.environ.get("FORGE_DICTATION_TEXT"))
    try:
        root = tempfile.mkdtemp()
        home = os.path.join(root, "home")
        os.environ["HOME"] = home
        os.environ["FORGE_LEARNER"] = "source"
        os.environ["FORGE_DB"] = os.path.join(home, ".forge", "source.db")
        now = time.time() - 3 * DAY
        m = Memory()
        m.record("alpha topic", "retrieval", 0.2,
                 lesson="retrieval alpha beta gamma", now=now)
        m.record("beta topic", "calibration", 0.2,
                 lesson="calibration beta gamma delta", now=now)

        def run(args, stdin=""):
            buf = io.StringIO()
            old_stdin = sys.stdin
            try:
                sys.stdin = io.StringIO(stdin)
                with contextlib.redirect_stdout(buf):
                    cli.main(args)
            finally:
                sys.stdin = old_stdin
            return buf.getvalue()

        cfg = json.loads(run(["--json", "config"]))
        assert cfg["stub"] is True and cfg["learner"] == "source"
        due_json = json.loads(run(["--json", "due"]))
        assert due_json and due_json[0]["topic"] in {"alpha topic", "beta topic"}
        assert "source" in run(["profiles"])
        assert "copied profile source -> target" in run(["profile-copy", "source", "target"])
        assert "target" in run(["profiles"])
        assert "_arguments" in run(["completion", "--shell", "zsh"])
        assert "peak risk" in run(["notify", "--dry-run"])
        assert "retrieval" in run(["semantic", "retrieval memory"])
        assert "Bridge:" in run(["bridge"])
        sheet = os.path.join(root, "sheet.html")
        assert "weekly review sheet" in run(["weekly-sheet", sheet])
        assert "Forge Weekly Review" in open(sheet, encoding="utf-8").read()
        assert "Capstone: alpha topic" in run(["capstone", "alpha topic"])

        export_path = os.path.join(root, "anki-tags.tsv")
        assert "exported" in run(["export", export_path, "--tags"])
        exported = open(export_path, encoding="utf-8").read()
        assert "topic_alpha_topic" in exported and "\tforge " in exported

        assert cli.cleanup_dictation("[noise] um alpha beta like gamma") == "alpha beta gamma"
        os.environ["FORGE_DICTATION_TEXT"] = "um alpha beta gamma"
        out = run(["review", "--limit", "1", "--push-to-talk",
                   "--cleanup-dictation", "--readback", "--no-followup"])
        assert "Heard: alpha beta gamma" in out
    finally:
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home
        if old_db is None:
            os.environ.pop("FORGE_DB", None)
        else:
            os.environ["FORGE_DB"] = old_db
        if old_learner is None:
            os.environ.pop("FORGE_LEARNER", None)
        else:
            os.environ["FORGE_LEARNER"] = old_learner
        if old_dictation is None:
            os.environ.pop("FORGE_DICTATION_TEXT", None)
        else:
            os.environ["FORGE_DICTATION_TEXT"] = old_dictation


def test_post_top100_lazy_reports_prompts_and_exports():
    from forge import cli

    old_db = os.environ.get("FORGE_DB")
    try:
        root = tempfile.mkdtemp()
        os.environ["FORGE_DB"] = os.path.join(root, "post100.db")
        now = time.time() - 2 * DAY
        m = Memory()
        m.record("post topic", "mechanism", 0.4,
                 lesson="alpha beta gamma causal mechanism", now=now)
        m.record("other topic", "boundary", 1.0,
                 lesson="delta invariant boundary condition", now=now)
        m.save_progress("post topic", [{"name": "next", "summary": "s"}], 0)
        m.save_source_manifest("post topic", build_source_manifest([
            ("notes.md", "alpha beta source quote"),
        ]))

        prompts = {p["kind"] for p in m.retrieval_prompts("post topic", "mechanism")}
        assert {"three_mechanisms", "missing_step", "failure_mode",
                "boundary_condition", "proof_sketch", "diagram_label",
                "invariant", "no_jargon"} <= prompts
        analytics = m.analytics_summary(now=time.time())
        assert analytics["reviews_due_this_week"] >= 1
        assert analytics["lapses_by_topic"]["post topic"] >= 1
        assert m.sprint_cards("weak", 1)[0]["concept"] == "mechanism"
        assert m.progress_rows()[0]["next_concept"] == "next"
        report = m.maintenance_report()
        assert report["integrity"] == "ok" and report["cards"] == 2
        source_report = m.source_report("post topic")
        assert source_report["chunks"] == 1
        assert m.source_quiz("post topic", "glossary", 1)[0]["kind"] == "glossary"
        assert m.source_quiz("post topic", "section", 1)[0]["label"] == "S1"

        def run(args, stdin=""):
            buf = io.StringIO()
            old_stdin = sys.stdin
            try:
                sys.stdin = io.StringIO(stdin)
                with contextlib.redirect_stdout(buf):
                    cli.main(args)
            finally:
                sys.stdin = old_stdin
            return buf.getvalue()

        assert "mechanism" in run(["queue"])
        assert "post topic" in run(["health"])
        assert "Most expensive" in run(["analytics"])
        assert json.loads(run(["--json", "analytics"]))["reviews_due_this_week"] >= 1
        assert "integrity: ok" in run(["maintain"])
        assert "rebuilt FTS" in run(["rebuild-fts"])

        # doctor --fix removes orphan attempts/notes and rebuilds FTS, without
        # touching live cards
        m.db.execute("INSERT INTO notes(topic, concept, lesson) VALUES('ghost', 'x', 'y')")
        m.db.execute("INSERT INTO attempts(card_id, ts, score) VALUES(99999, ?, 1.0)",
                     (time.time(),))
        m.db.commit()
        before = m.maintenance_report()
        assert len(before["orphan_notes"]) == 1 and len(before["orphan_attempts"]) == 1
        doctor_out = run(["doctor", "--fix"])
        assert "removed 1 orphan attempt" in doctor_out and "1 orphan note" in doctor_out
        after = m.maintenance_report()
        assert not after["orphan_notes"] and not after["orphan_attempts"]
        assert after["cards"] == before["cards"]  # cards themselves untouched
        assert "digest=" in run(["source-report", "post topic"])
        assert "[S1]" in run(["source-quotes", "post topic", "alpha"])
        assert "Define" in run(["source-quiz", "post topic"])
        assert "reconstruct" in run(["source-quiz", "post topic", "--kind", "section"])
        assert "post topic" in run(["progress"])
        assert "forge learn" in run(["resume-oldest"])
        assert "post topic" in run(["next"])
        assert "tomorrow:" in run(["tomorrow"])
        assert "weekend:" in run(["weekend"])
        Memory().record("post topic", "today miss", 0.2,
                        lesson="today miss lesson", now=time.time())
        assert "post topic / today miss" in run(["failed-today"])
        assert "average score" in run(["score-today"])
        before = len(Memory().attempt_timeline(1000))
        answers = "remember alpha beta gamma\nremember alpha beta gamma\n"
        assert "Review:" in run(["drill", "--topic", "post topic", "--concept", "mechanism",
                                  "--no-followup"], answers)
        assert len(Memory().attempt_timeline(1000)) == before + 1
        before += 1
        assert "Review:" in run(["sprint", "--mode", "weak", "--limit", "1",
                                  "--no-followup"], answers)
        assert len(Memory().attempt_timeline(1000)) == before + 1
        assert "cleared progress" in run(["clear-progress", "post topic"])
        assert Memory().progress_rows() == []

        paths = {
            "cards.csv": ["csv-export"],
            "attempts.jsonl": ["jsonl-export"],
            "topics.md": ["markdown-export"],
            "source.json": ["source-export", "post topic"],
            "graph.json": ["graph-json"],
            "forecast.json": ["forecast-export"],
        }
        for name, args in paths.items():
            path = os.path.join(root, name)
            assert "exported" in run(args + [path])
            assert os.path.exists(path) and os.path.getsize(path) > 0
        checksum = run(["checksum", os.path.join(root, "cards.csv")]).strip()
        assert len(checksum) == 64
        anki = os.path.join(root, "source-anki.tsv")
        assert "exported" in run(["export", anki, "--source-citations"])
        assert "Sources:<br>[S1]" in open(anki, encoding="utf-8").read()
    finally:
        if old_db is None:
            os.environ.pop("FORGE_DB", None)
        else:
            os.environ["FORGE_DB"] = old_db


def test_schedule_install_status_remove():
    from forge import cli

    old_home, old_platform, old_path = (os.environ.get("HOME"), sys.platform,
                                        os.environ.get("PATH", ""))
    fake_home = tempfile.mkdtemp()
    fake_bin = tempfile.mkdtemp()
    launchctl = os.path.join(fake_bin, "launchctl")
    with open(launchctl, "w") as f:
        f.write("#!/bin/sh\nexit 0\n")
    os.chmod(launchctl, 0o755)
    os.environ["HOME"] = fake_home
    os.environ["PATH"] = fake_bin + os.pathsep + old_path
    sys.platform = "darwin"

    def run(args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.main(args)
        return buf.getvalue()

    try:
        assert "not installed" in run(["schedule", "status"])
        assert "installed" in run(["schedule", "install", "--hour", "10"])
        plist_path = os.path.join(fake_home, "Library/LaunchAgents/com.forge.notify.plist")
        assert os.path.exists(plist_path)
        with open(plist_path, "rb") as f:
            plist = plistlib.load(f)
        assert plist["StartCalendarInterval"]["Hour"] == 10
        assert plist["ProgramArguments"][-1] == "notify"
        status = run(["schedule", "status"])
        assert "installed" in status and "not installed" not in status
        assert "removed" in run(["schedule", "remove"])
        assert not os.path.exists(plist_path)
    finally:
        sys.platform = old_platform
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home
        os.environ["PATH"] = old_path


def test_web_dashboard():
    from http.server import ThreadingHTTPServer
    from urllib.parse import quote

    from forge import web
    os.environ["FORGE_DB"] = os.path.join(tempfile.mkdtemp(), "web.db")
    server = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    def call(method, path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request(method, path, json.dumps(body) if body else None)
        return json.loads(conn.getresponse().read())

    state = call("GET", "/api/state")
    assert state["model"] == "stub" and state["learner"] in ("default", os.environ.get("FORGE_LEARNER"))
    uploaded = base64.b64encode(b"WEB SOURCE {browser braces}").decode()
    assert call("POST", "/api/learn", {
        "topic": "webtest",
        "source_name": "notes.md",
        "source_data": uploaded,
    })["ok"] is True
    assert "error" in call("POST", "/api/learn", {"topic": ""})  # validation
    ended, deadline = False, time.time() + 30
    while not ended and time.time() < deadline:
        for e in call("GET", "/api/events"):
            if e["type"] == "ask":
                call("POST", "/api/answer", {"text": "remember alpha beta gamma"})
            elif e["type"] == "end":
                ended = True
        time.sleep(0.05)
    assert ended, "web learn session never finished"
    assert len(call("GET", "/api/stats")) == 3
    assert call("GET", "/api/graphdata")["nodes"]
    assert call("GET", "/api/due") == []
    assert call("GET", "/api/topics")[0]["topic"] == "webtest"
    assert call("GET", "/api/weak")
    assert call("GET", "/api/debt")["due"] == 0
    found = call("GET", "/api/search?q=alpha")
    assert found and found[0]["topic"] == "webtest"
    card = call("GET", "/api/card?topic=webtest&concept=webtest%3A%20foundations")
    assert card["concept"] == "webtest: foundations"
    assert call("GET", "/api/progress") == []
    Memory().save_progress("resume web", [{"name": "next", "summary": "s"}], 0)
    progress = call("GET", "/api/progress")
    assert progress[0]["topic"] == "resume web" and progress[0]["next_concept"] == "next"
    assert "Forge Retrieval Transcript" in call("GET", "/api/transcript")["text"]
    assert call("POST", "/api/due-now", {
        "topic": "webtest", "concept": "webtest: foundations",
    })["concept"] == "webtest: foundations"
    assert call("GET", "/api/due")
    assert call("POST", "/api/suspend", {
        "topic": "webtest", "concept": "webtest: foundations",
    })["suspended"] == 1
    assert call("GET", "/api/due") == []
    # artifact endpoint only serves SVGs inside the artifact dir (traversal guard)
    artdir = tempfile.mkdtemp()
    os.environ["FORGE_ARTIFACT_DIR"] = artdir
    svg = os.path.join(artdir, "visual.svg")
    open(svg, "w", encoding="utf-8").write("<svg xmlns='http://www.w3.org/2000/svg'></svg>")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", "/api/artifact?path=" + quote(svg))
    assert b"<svg" in conn.getresponse().read()
    outside = os.path.join(tempfile.mkdtemp(), "outside.svg")
    open(outside, "w", encoding="utf-8").write("<svg xmlns='http://www.w3.org/2000/svg'></svg>")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", "/api/artifact?path=" + quote(outside))
    assert conn.getresponse().status == 404
    os.environ.pop("FORGE_ARTIFACT_DIR", None)
    # non-local Host headers are refused everywhere (DNS-rebinding guard)
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", "/api/state", headers={"Host": "evil.example.com"})
    assert conn.getresponse().status == 403
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", "/api/answer", json.dumps({"text": "x"}),
                 headers={"Host": "evil.example.com"})
    assert conn.getresponse().status == 403
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", "/")
    page = conn.getresponse().read()
    assert b"The Forge" in page
    assert b"Retrieval cockpit" in page and b"settingsDrawer" in page
    assert b"transcriptBtn" in page and b"dragover" in page
    assert call("POST", "/api/learner", {"learner": "web-student"})["ok"] is True
    assert call("GET", "/api/state")["learner"] == "web-student"
    server.shutdown()


# === WS-ENGINE ===
import http.server  # noqa: E402
import stat  # noqa: E402
from contextlib import contextmanager  # noqa: E402

from forge import config as forge_config  # noqa: E402
from forge import llm as forge_llm  # noqa: E402
from forge.llm import AnthropicEngine, Ollama  # noqa: E402
from forge.llm import get_grader as _get_grader  # noqa: E402

_FAKE_KEY = "sk-ant-test-XYZZY-not-a-real-key"
_ENGINE_VARS = ("FORGE_STUB", "FORGE_ENGINE", "FORGE_MODEL", "FORGE_CONFIG",
                "ANTHROPIC_API_KEY", "FORGE_ANTHROPIC_URL", "FORGE_GRADER")


@contextmanager
def _engine_env(**vals):
    saved = {k: os.environ.get(k) for k in _ENGINE_VARS}
    saved_url = forge_llm.OLLAMA_URL
    try:
        for k in _ENGINE_VARS:
            os.environ.pop(k, None)
        for k, v in vals.items():
            os.environ[k] = v
        forge_llm.OLLAMA_URL = "http://127.0.0.1:9"  # dead port by default
        yield
    finally:
        forge_llm.OLLAMA_URL = saved_url
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _MockAPI(http.server.BaseHTTPRequestHandler):
    mode = "ok"  # "ok" | "401"
    last = {}

    def log_message(self, *args):
        pass

    def _send(self, code, payload=None):
        body = json.dumps(payload or {}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send(200, {"models": [{"name": "llama3"}]})
        else:
            self._send(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        _MockAPI.last = {
            "path": self.path,
            "key": self.headers.get("x-api-key"),
            "version": self.headers.get("anthropic-version"),
            "body": json.loads(self.rfile.read(n)),
        }
        if _MockAPI.mode == "401":
            self._send(401, {"error": {"type": "authentication_error"}})
            return
        self._send(200, {"content": [
            {"type": "text", "text": "```json\n{\"answer\": 42}\n```"}]})


@contextmanager
def _mock_server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _MockAPI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()


def test_ws_engine_resolution_order():
    cfg = os.path.join(tempfile.mkdtemp(), "config.json")
    # rung 1: FORGE_STUB wins over everything
    with _engine_env(FORGE_STUB="1", FORGE_ENGINE="anthropic", FORGE_CONFIG=cfg):
        assert get_llm().name == "stub"
    # rung 2: FORGE_ENGINE explicit — honored, and hard-fails when unhealthy
    with _engine_env(FORGE_ENGINE="stub", FORGE_CONFIG=cfg):
        assert get_llm().name == "stub"
    with _engine_env(FORGE_ENGINE="anthropic", FORGE_CONFIG=cfg):
        try:
            get_llm()
            raise AssertionError("expected RuntimeError for missing key")
        except RuntimeError as e:
            assert "ANTHROPIC_API_KEY" in str(e)
    with _engine_env(FORGE_ENGINE="ollama", FORGE_CONFIG=cfg):
        try:
            get_llm()
            raise AssertionError("expected RuntimeError for dead ollama")
        except RuntimeError as e:
            assert "ollama serve" in str(e)
    # rung 3: config file engine — same hard behavior, resolves when healthy
    with _engine_env(FORGE_CONFIG=cfg, ANTHROPIC_API_KEY=_FAKE_KEY):
        forge_config.save({"engine": "anthropic", "model": None,
                           "api_key_env": "ANTHROPIC_API_KEY"})
        llm = get_llm()
        assert llm.name == "anthropic"
        assert llm.model == "claude-haiku-4-5-20251001"
    os.remove(cfg)
    # rung 4: auto — Ollama reachable wins
    with _engine_env(FORGE_CONFIG=cfg), _mock_server() as url:
        forge_llm.OLLAMA_URL = url
        llm = get_llm()
        assert llm.name == "ollama" and llm.model == "llama3"
        ok, msg = llm.healthy()
        assert ok, msg
    # rung 5: auto — no Ollama, Anthropic key present
    with _engine_env(FORGE_CONFIG=cfg, ANTHROPIC_API_KEY=_FAKE_KEY):
        assert get_llm().name == "anthropic"
    # rung 6: auto — nothing available -> Stub, never crashes
    with _engine_env(FORGE_CONFIG=cfg):
        llm = get_llm()
        assert llm.name == "stub"
        assert llm.healthy()[0] is True


def test_ws_engine_healthy_messages():
    with _engine_env():
        ok, msg = Ollama("llama3").healthy()
        assert ok is False and "ollama serve" in msg
        ok, msg = AnthropicEngine().healthy()
        assert ok is False and "ANTHROPIC_API_KEY" in msg
    with _engine_env(ANTHROPIC_API_KEY=_FAKE_KEY):
        ok, msg = AnthropicEngine().healthy()
        assert ok is True and _FAKE_KEY not in msg


def test_ws_engine_anthropic_request_shape():
    _MockAPI.mode = "ok"
    with _mock_server() as url, _engine_env(
            ANTHROPIC_API_KEY=_FAKE_KEY, FORGE_ANTHROPIC_URL=url):
        out = AnthropicEngine().ask("hello engine", as_json=True)
        assert out == '{"answer": 42}'  # fences stripped, parses clean
        assert json.loads(out) == {"answer": 42}
        last = _MockAPI.last
        assert last["path"] == "/v1/messages"
        assert last["key"] == _FAKE_KEY
        assert last["version"] == "2023-06-01"
        body = last["body"]
        assert body["model"] == "claude-haiku-4-5-20251001"
        assert body["max_tokens"] == 4096 and body["temperature"] == 0.4
        content = body["messages"][0]["content"]
        assert "hello engine" in content
        assert "ONLY valid JSON" in content


def test_ws_engine_error_is_prescriptive_and_keyless():
    _MockAPI.mode = "401"
    try:
        with _mock_server() as url, _engine_env(
                ANTHROPIC_API_KEY=_FAKE_KEY, FORGE_ANTHROPIC_URL=url):
            try:
                AnthropicEngine().ask("hi")
                raise AssertionError("expected RuntimeError on 401")
            except RuntimeError as e:
                msg = str(e)
                assert "API key rejected" in msg and "forge init" in msg
                assert _FAKE_KEY not in msg
    finally:
        _MockAPI.mode = "ok"


def test_ws_engine_config_save_mode_and_key_lookup():
    cfg = os.path.join(tempfile.mkdtemp(), "config.json")
    with _engine_env(FORGE_CONFIG=cfg, ANTHROPIC_API_KEY="env-key"):
        path = forge_config.save({"engine": "stub", "model": None,
                                  "api_key_env": "ANTHROPIC_API_KEY"})
        assert path == cfg
        assert stat.S_IMODE(os.stat(cfg).st_mode) == 0o600
        loaded = forge_config.load()
        assert loaded["engine"] == "stub"
        # env key via api_key_env
        assert forge_config.api_key() == "env-key"
        # explicit config key wins over env; save preserves it
        forge_config.save({**loaded, "api_key": "cfg-key"})
        assert forge_config.api_key() == "cfg-key"
        assert forge_config.load()["api_key"] == "cfg-key"
    # missing/corrupt file -> {} not crash
    with _engine_env(FORGE_CONFIG=cfg + ".missing"):
        assert forge_config.load() == {}


def test_ws_engine_grader_follows_engine_type():
    with _engine_env(FORGE_STUB="1", FORGE_GRADER="whatever"):
        assert _get_grader().name == "stub"  # stub ignores FORGE_GRADER
    with _engine_env(FORGE_ENGINE="anthropic", ANTHROPIC_API_KEY=_FAKE_KEY,
                     FORGE_GRADER="claude-opus-x"):
        grader = _get_grader()
        assert grader.name == "anthropic" and grader.model == "claude-opus-x"


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"PASS {fn.__name__}")
    print("all tests passed")
