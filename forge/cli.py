"""Text UI for The Forge.

  forge learn "<topic>"   start a mastery-gated session (no exit but mastery)
        --from FILE       ground the curriculum in your own notes/document
        --speak           voice mode: lessons and questions read aloud (macOS)
  forge review            drill everything the scheduler says is due
  forge web               dashboard at http://127.0.0.1:8765
  forge stats             strength, recall %, schedule, streak, 14-day forecast
  forge export            Anki-importable TSV of everything learned
  forge graph             export knowledge graph to forge_graph.html
"""
import argparse
import csv
import hashlib
import json
import os
import plistlib
import re
import shutil
import subprocess
import sys
import time

from .graph import run_review, run_topic
from .graphview import export
from .graphview import layout as graph_layout
from .llm import get_llm
from .memory import DAY, DEFAULT_DB, Memory, retrievability
from .profiles import copy_profile, list_profiles, select_profile
from .sources import SourceError, read_source, source_manifest as build_source_manifest


def _ask(prompt: str) -> str:
    return input(prompt)


def _io(speak: bool, voice: str = "", push_to_talk: bool = False,
        cleanup: bool = False, readback: bool = False):
    """(emit, ask) pair; --speak reads lessons and questions aloud on macOS."""
    def say(text):
        if not (speak and sys.platform == "darwin"):
            return
        t = str(text).strip()
        if t:
            # ponytail: blocking speech paces the session; async queue if it drags
            cmd = ["say", "-r", "200"]
            if voice:
                cmd += ["-v", voice]
            subprocess.run(cmd + [t[:900]])

    def emit(text=""):
        print(text)
        say(text)

    def ask(prompt):
        say(prompt)
        if push_to_talk:
            text = _push_to_talk_answer(prompt)
        else:
            text = input(prompt)
        if cleanup:
            text = cleanup_dictation(text)
        if readback:
            print(f"[Forge] Heard: {text}")
            say(f"Heard: {text}")
        return text

    return emit, ask


def cleanup_dictation(text: str) -> str:
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"\b(um+|uh+|er+|ah+|like|you know)\b", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def _push_to_talk_answer(prompt: str) -> str:
    seeded = os.environ.get("FORGE_DICTATION_TEXT")
    if seeded is not None:
        print(prompt, end="")
        print(seeded)
        return seeded
    wav = os.environ.get("FORGE_DICTATION_WAV")
    cmd = os.environ.get("FORGE_WHISPER_CMD") or shutil.which("whisper-cli") or shutil.which("main")
    if cmd and wav and os.path.exists(wav):
        proc = subprocess.run([cmd, "-f", wav], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, timeout=180)
        if proc.returncode == 0 and proc.stdout.strip():
            text = proc.stdout.strip().splitlines()[-1]
            print(prompt, end="")
            print(text)
            return text
    return input(prompt + "[Forge] push-to-talk unavailable; type answer.\n> ")


def _json(data) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def cmd_learn(args):
    llm = get_llm()
    print(f"[Forge] model: {llm.model}")
    source = ""
    manifest = {}
    if args.source:
        docs = []
        for path in args.source:
            text = read_source(path)
            docs.append((os.path.basename(path), text))
        manifest = build_source_manifest(docs)
        source = "\n\n".join(text for _, text in docs)
        chunks = sum(len(doc["chunks"]) for doc in manifest["docs"])
        print(f"[Forge] curriculum grounded in {len(docs)} source file(s), "
              f"{chunks} chunk(s), {len(source)} chars")
    emit, ask = _io(args.speak, args.voice, args.push_to_talk,
                   args.cleanup_dictation, args.readback)
    final = run_topic(args.topic, ask=ask, emit=emit, llm=llm,
                      max_failed_attempts=args.max_failures, source=source,
                      source_manifest=manifest, correct_lessons=args.correct)
    if not final.get("mastery"):
        print("\n[Forge] Halted before mastery (failure cap hit). "
              "The topic stays on your schedule.")


def cmd_review(args):
    emit, ask = _io(args.speak, args.voice, args.push_to_talk,
                   args.cleanup_dictation, args.readback)
    run_review(ask=ask, emit=emit, limit=args.limit, hard=args.hard,
               confidence=args.confidence, followup=not args.no_followup)


def cmd_practice(args):
    emit, ask = _io(args.speak, args.voice, args.push_to_talk,
                   args.cleanup_dictation, args.readback)
    run_review(ask=ask, emit=emit, limit=args.limit, hard=args.hard,
               confidence=args.confidence, followup=not args.no_followup)


def cmd_cram(args):
    emit, ask = _io(args.speak, args.voice, args.push_to_talk,
                   args.cleanup_dictation, args.readback)
    run_review(ask=ask, emit=emit, limit=args.limit, days=args.days,
               hard=args.hard, confidence=args.confidence,
               followup=not args.no_followup)


def cmd_web(args):
    from .web import serve
    serve(args.port, learner=args.learner)


def cmd_stats(args):
    m = Memory()
    rows = m.stats()
    if not rows:
        if args.json:
            _json([])
            return
        print("[Forge] Empty memory. Run: forge learn \"<topic>\"")
        return
    now = time.time()
    if args.json:
        out = []
        for r in rows:
            d = dict(r)
            d["recall"] = retrievability(r["interval_days"], r["due"], now)
            d["due_in_days"] = (r["due"] - now) / DAY
            out.append(d)
        _json(out)
        return
    print(f"{'topic':<22}{'concept':<34}{'ease':>6}{'ivl(d)':>8}{'due(d)':>8}"
          f"{'reps':>6}{'lapses':>7}{'avg':>6}{'recall':>8}")
    for r in rows:
        rec = retrievability(r["interval_days"], r["due"], now)
        print(f"{r['topic'][:21]:<22}{r['concept'][:33]:<34}{r['ease']:>6.2f}"
              f"{r['interval_days']:>8.1f}{(r['due'] - now) / DAY:>8.1f}"
              f"{r['reps']:>6}{r['lapses']:>7}{(r['avg_score'] or 0):>6.2f}"
              f"{'   new' if rec is None else f'{rec * 100:>7.0f}%'}")
    fc = m.forecast()
    bars = "▁▂▃▄▅▆▇█"
    spark = "".join(bars[min(7, c)] for c in fc)
    print(f"\nstreak: {m.streak()} day(s)   next 14 days: {spark} "
          f"({sum(fc)} reviews queued)")


def cmd_due(args):
    rows = Memory().due_queue(limit=args.limit, days=args.days)
    if args.json:
        _json(rows)
        return
    if not rows:
        print("[Forge] No cards in that due window.")
        return
    print(f"{'risk':>5} {'due':>8} {'topic':<24} {'concept':<36} why")
    for r in rows:
        due = "now" if r["due_in_days"] <= 0 else f"{r['due_in_days']:.1f}d"
        print(f"{r['risk']:>5.2f} {due:>8} {r['topic'][:23]:<24}"
              f"{r['concept'][:35]:<36} {', '.join(r['reasons'])}")


def cmd_weak(args):
    rows = Memory().weak_cards(args.limit)
    if args.json:
        _json(rows)
        return
    if not rows:
        print("[Forge] No cards yet.")
        return
    print(f"{'risk':>5} {'ease':>5} {'laps':>4} {'topic':<24} {'concept':<36} why")
    for r in rows:
        print(f"{r['risk']:>5.2f} {r['ease']:>5.2f} {r['lapses']:>4} "
              f"{r['topic'][:23]:<24}{r['concept'][:35]:<36} {', '.join(r['reasons'])}")


def cmd_topics(args):
    rows = Memory().topic_summaries()
    if args.json:
        _json(rows)
        return
    if not rows:
        print("[Forge] Empty memory.")
        return
    print(f"{'health':>6} {'risk':>5} {'due':>4} {'weak':>4} {'cards':>5} topic")
    for r in rows:
        active = f" (active #{r['active_idx'] + 1})" if r["active_idx"] is not None else ""
        print(f"{r['health']:>6.0%} {r['risk']:>5.2f} {r['due']:>4} {r['weak']:>4} "
              f"{r['concepts']:>5} {r['topic']}{active}")


def cmd_inspect(args):
    d = Memory().card_detail(args.topic, args.concept)
    if not d:
        print("[Forge] Concept not found.")
        return
    rec = "new" if d["recall"] is None else f"{d['recall'] * 100:.0f}%"
    due = "now" if d["due_in_days"] <= 0 else f"{d['due_in_days']:.1f}d"
    print(f"{d['topic']} / {d['concept']}")
    print(f"risk {d['risk']:.2f} ({', '.join(d['reasons'])}); recall {rec}; "
          f"due {due}; ease {d['ease']:.2f}; reps {d['reps']}; lapses {d['lapses']}")
    if d["lesson"]:
        print("\nlesson:")
        print(d["lesson"][:1200])
    if d["attempts"]:
        print("\nrecent attempts:")
        for a in d["attempts"]:
            stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(a["ts"]))
            print(f"  {stamp}  {a['score']:.2f}")


def cmd_search(args):
    rows = Memory().search_lessons(args.query, args.limit)
    if not rows:
        print("[Forge] No lesson matches.")
        return
    for r in rows:
        print(f"\n{r['topic']} / {r['concept']}")
        print(r["excerpt"])


def cmd_timeline(args):
    rows = Memory().attempt_timeline(args.limit, args.topic or "", args.concept or "")
    if not rows:
        print("[Forge] No attempts yet.")
        return
    print(f"{'when':<17} {'score':>5} {'topic':<24} concept")
    for r in rows:
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["ts"]))
        print(f"{stamp:<17} {r['score']:>5.2f} {r['topic'][:23]:<24} {r['concept']}")


def cmd_transcript(args):
    text = Memory().transcript(args.limit, args.topic or "", args.concept or "")
    if args.path:
        with open(args.path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[Forge] wrote transcript to {args.path}")
    else:
        print(text, end="")


def cmd_debt(args):
    d = Memory().review_debt()
    if args.json:
        _json(d)
        return
    print(f"due: {d['due']}   peak risk: {d['peak_risk']:.2f}   "
          f"avg risk: {d['avg_risk']:.2f}   oldest overdue: {d['max_overdue_days']:.1f}d   "
          f"20/day catch-up: {d['review_days_at_20']} day(s)")


def cmd_avalanche(args):
    d = Memory().avalanche_warning(args.days, args.budget)
    status = "warning" if d["warning"] else "steady"
    print(f"{status}: {d['total']} reviews in {d['days']}d; "
          f"peak day {d['peak_day']} has {d['peak_count']} "
          f"(budget {d['daily_budget']}/day)")
    if d["overloaded_days"]:
        print("overloaded days: " + ", ".join(str(x) for x in d["overloaded_days"]))


def cmd_vacation(args):
    n = Memory().shift_due(args.days, args.topic or "", args.concept or "")
    print(f"[Forge] shifted {n} active due date(s) by {args.days:g} day(s)")


def cmd_performance(args):
    rows = (Memory().performance_by_hour() if args.by == "hour"
            else Memory().performance_by_weekday())
    if not rows:
        print("[Forge] No attempts yet.")
        return
    print(f"{args.by:<8} {'attempts':>8} {'avg':>6}")
    for r in rows:
        print(f"{r['bucket']:<8} {r['attempts']:>8} {r['avg_score']:>6.2f}")


def cmd_today(args):
    m = Memory()
    debt = m.review_debt()
    print("[Forge] Today")
    print(f"due {debt['due']} · peak risk {debt['peak_risk']:.2f} · streak {m.streak()}d")
    weak = m.weak_cards(5)
    if weak:
        print("\nTop retrieval targets:")
        for r in weak:
            print(f"  {r['risk']:.2f}  {r['topic']} / {r['concept']} — {', '.join(r['reasons'])}")


def cmd_stuck(args):
    rows = Memory().stuck_cards(args.limit, args.min_lapses)
    if not rows:
        print("[Forge] No repeatedly failed active cards.")
        return
    print(f"{'lapses':>6} {'risk':>5} {'topic':<24} {'concept':<36} why")
    for r in rows:
        print(f"{r['lapses']:>6} {r['risk']:>5.2f} {r['topic'][:23]:<24}"
              f"{r['concept'][:35]:<36} {', '.join(r['reasons'])}")


def cmd_daily5(args):
    emit, ask = _io(args.speak, args.voice, args.push_to_talk,
                   args.cleanup_dictation, args.readback)
    run_review(ask=ask, emit=emit, limit=5, hard=args.hard,
               confidence=args.confidence, followup=not args.no_followup)


def cmd_exam(args):
    m = Memory()
    cards = [r for r in m.stats() if r["topic"] == args.topic and not r["suspended"]]
    if not cards:
        print("[Forge] No active cards for that topic.")
        return
    for r in cards:
        m.due_now(r["topic"], r["concept"])
    emit, ask = _io(args.speak, args.voice, args.push_to_talk,
                   args.cleanup_dictation, args.readback)
    run_review(ask=ask, emit=emit, memory=m, limit=len(cards),
               hard=args.hard, confidence=args.confidence,
               followup=not args.no_followup)


def cmd_weekly(args):
    m = Memory()
    cards = m.weak_cards(args.limit)
    if not cards:
        print("[Forge] No active cards yet.")
        return
    for r in cards:
        m.due_now(r["topic"], r["concept"])
    emit, ask = _io(args.speak, args.voice, args.push_to_talk,
                   args.cleanup_dictation, args.readback)
    run_review(ask=ask, emit=emit, memory=m, limit=len(cards),
               hard=args.hard, confidence=args.confidence,
               followup=not args.no_followup)


def cmd_sprint(args):
    m = Memory()
    cards = m.sprint_cards(args.mode, args.limit, args.topic or "")
    if not cards:
        print("[Forge] No cards match that sprint.")
        return
    for c in cards:
        m.due_now(c["topic"], c["concept"])
    emit, ask = _io(args.speak, args.voice, args.push_to_talk,
                   args.cleanup_dictation, args.readback)
    run_review(ask=ask, emit=emit, memory=m, limit=len(cards),
               hard=args.hard, confidence=args.confidence,
               followup=not args.no_followup)


def cmd_drill(args):
    m = Memory()
    if args.topic and args.concept:
        cards = [m.due_now(args.topic, args.concept)]
    else:
        cards = m.sprint_cards(args.kind, args.limit, args.topic or "")
        for c in cards:
            m.due_now(c["topic"], c["concept"])
    if not cards:
        print("[Forge] No cards match that drill.")
        return
    emit, ask = _io(args.speak, args.voice, args.push_to_talk,
                   args.cleanup_dictation, args.readback)
    run_review(ask=ask, emit=emit, memory=m, limit=len(cards),
               hard=args.hard, confidence=args.confidence,
               followup=not args.no_followup)


def cmd_missed(args):
    rows = Memory().missed_points(args.topic or "", args.concept or "", args.limit)
    if not rows:
        print("[Forge] No missed key points recorded yet.")
        return
    print(f"{'count':>5} {'topic':<24} {'concept':<36} missed point")
    for r in rows:
        print(f"{r['count']:>5} {r['topic'][:23]:<24}{r['concept'][:35]:<36} {r['point']}")


def cmd_calibration(args):
    c = Memory().calibration()
    if not c["n"]:
        print("[Forge] No confidence data yet. Use review --confidence.")
        return
    print(f"attempts: {c['n']}   avg confidence: {c['avg_confidence']:.2f}/5   "
          f"avg score: {c['avg_score']:.2f}   gap: {c['calibration_gap']:.2f}")


def cmd_prompts(args):
    prompts = Memory().retrieval_prompts(args.topic, args.concept)
    for p in prompts:
        print(f"[{p['kind']}] {p['prompt']}")


def cmd_cloze(args):
    rows = Memory().cloze_cards(args.topic or "", args.limit)
    if args.path:
        with open(args.path, "w", encoding="utf-8") as f:
            f.write("#separator:tab\n#html:true\n")
            for r in rows:
                f.write(f"{r['topic']}::{r['concept']}\t{r['cloze']}\n")
        print(f"[Forge] wrote {len(rows)} cloze cards to {args.path}")
    else:
        for r in rows:
            print(f"\n{r['topic']} / {r['concept']}\n{r['cloze']}")


def cmd_note(args):
    m = Memory()
    if args.set is not None:
        m.set_learner_note(args.topic, args.concept, args.set,
                           replace_lesson=args.replace_lesson)
        print("[Forge] learner note stored.")
        return
    note = m.learner_note(args.topic, args.concept)
    print(note or "[Forge] No learner note stored.")


def cmd_promote_answer(args):
    Memory().note_from_best_answer(args.topic, args.concept,
                                   replace_lesson=args.replace_lesson)
    print("[Forge] promoted best passing answer to learner note.")


def cmd_source_checklist(args):
    text = read_source(args.source)
    words = []
    for w in text.replace("-", " ").split():
        clean = "".join(ch for ch in w if ch.isalnum()).lower()
        if len(clean) >= 5 and clean not in words:
            words.append(clean)
        if len(words) >= args.limit:
            break
    for i, w in enumerate(words, 1):
        print(f"{i}. {w}")


def cmd_source_manifest(args):
    manifest = Memory().source_manifest(args.topic)
    if not manifest.get("docs"):
        print("[Forge] No source manifest stored for that topic.")
        return
    print(f"{args.topic}  digest={manifest['digest']}  docs={len(manifest['docs'])}")
    for doc in manifest["docs"]:
        print(f"- {doc['name']}  digest={doc['digest']}  chunks={len(doc['chunks'])}")


def cmd_source_chunks(args):
    m = Memory()
    manifest = m.source_manifest(args.topic)
    if not manifest.get("docs"):
        print("[Forge] No source manifest stored for that topic.")
        return
    rows = (m.source_snippets(args.topic, args.query, args.limit) if args.query
            else _all_source_chunks(manifest, args.limit))
    for r in rows:
        text = r["text"].replace("\n", " ")[:220]
        print(f"[{r['label']}] {r['source']}: {text}")


def cmd_source_coverage(args):
    rows = Memory().source_coverage(args.topic)
    if not rows:
        print("[Forge] No source coverage available for that topic.")
        return
    print(f"{'cover':>5} {'snips':>5} {'concept':<36} unsupported terms")
    for r in rows:
        terms = ", ".join(r["unsupported"][:6]) or "-"
        print(f"{r['coverage']:>5.2f} {r['snippets']:>5} {r['concept'][:35]:<36} {terms}")


def cmd_unsupported(args):
    rows = Memory().unsupported_claims(args.topic, args.concept or "", args.limit)
    if not rows:
        print("[Forge] No source-backed lessons found for that query.")
        return
    for r in rows:
        terms = ", ".join(r["terms"]) or "none"
        print(f"{r['topic']} / {r['concept']}: {terms}")


def cmd_source_glossary(args):
    rows = Memory().source_glossary(args.topic, args.limit)
    if not rows:
        print("[Forge] No source glossary available for that topic.")
        return
    for r in rows:
        print(f"{r['term']}\t{r['count']}")


def cmd_source_report(args):
    report = Memory().source_report(args.topic)
    if args.json:
        _json(report)
        return
    if not report["docs"]:
        print("[Forge] No source manifest stored for that topic.")
        return
    print(f"{report['topic']} digest={report['digest']} docs={len(report['docs'])} chunks={report['chunks']}")
    for d in report["docs"]:
        print(f"- {d['name']} {d['chars']} chars {d['chunks']} chunk(s) {d['digest']}")
    if report["glossary"]:
        print("\nglossary: " + ", ".join(r["term"] for r in report["glossary"][:8]))
    if report["coverage"]:
        print("\ncoverage:")
        for r in report["coverage"][:8]:
            print(f"  {r['coverage']:.2f} {r['concept']} [{r['best_label'] or '-'}]")


def cmd_source_quotes(args):
    rows = Memory().source_snippets(args.topic, args.query, args.limit)
    if not rows:
        print("[Forge] No source quotes found.")
        return
    for r in rows:
        print(f"[{r['label']}] {r['source']} score={r['score']:.2f}")
        print(r["text"].replace("\n", " ")[:500])


def cmd_source_quiz(args):
    rows = Memory().source_quiz(args.topic, args.kind, args.limit)
    if not rows:
        print("[Forge] No source quiz prompts available.")
        return
    for i, r in enumerate(rows, 1):
        print(f"{i}. {r['prompt']}")


def cmd_config(args):
    whisper = os.environ.get("FORGE_WHISPER_CMD") or shutil.which("whisper-cli") or shutil.which("main")
    data = {
        "db": os.environ.get("FORGE_DB", DEFAULT_DB),
        "learner": os.environ.get("FORGE_LEARNER", "default"),
        "model": get_llm().model,
        "ollama": os.environ.get("FORGE_OLLAMA", "http://localhost:11434"),
        "stub": bool(os.environ.get("FORGE_STUB")),
        "pdftotext": shutil.which("pdftotext") or "",
        "say": shutil.which("say") or "",
        "whisper": whisper or "",
    }
    if args.json:
        _json(data)
        return
    for k, v in data.items():
        print(f"{k}: {v if v != '' else 'missing'}")


def cmd_profiles(args):
    rows = list_profiles()
    if args.json:
        _json(rows)
        return
    if not rows:
        print("[Forge] No profiles found yet.")
        return
    print(f"{'current':<8} {'name':<24} {'size':>9} {'modified':<16} path")
    for r in rows:
        mark = "*" if r["current"] else ""
        print(f"{mark:<8} {r['name']:<24} {r['bytes']:>9} {r['modified']:<16} {r['path']}")


def cmd_profile_copy(args):
    path = copy_profile(args.source, args.target, overwrite=args.overwrite)
    print(f"[Forge] copied profile {args.source} -> {args.target} ({path})")


def cmd_completion(args):
    commands = [
        "learn", "review", "due", "weak", "topics", "inspect", "search", "practice",
        "cram", "daily5", "exam", "weekly", "config", "profiles", "profile-copy",
        "notify", "schedule", "semantic", "bridge", "weekly-sheet", "capstone",
    ]
    if args.shell == "zsh":
        print("#compdef forge\n_arguments '1:command:(" + " ".join(commands) + ")'")
    else:
        print("complete -W '" + " ".join(commands) + "' forge")


LAUNCHD_LABEL = "com.forge.notify"


def _launchd_plist_path() -> str:
    return os.path.expanduser(f"~/Library/LaunchAgents/{LAUNCHD_LABEL}.plist")


def cmd_schedule(args):
    if sys.platform != "darwin":
        raise ValueError("forge schedule needs launchd, which is macOS-only")
    plist_path = _launchd_plist_path()
    if args.action == "status":
        installed = os.path.exists(plist_path)
        print(f"[Forge] {'installed' if installed else 'not installed'}: {plist_path}")
        return
    if args.action == "remove":
        subprocess.run(["launchctl", "unload", plist_path], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(plist_path):
            os.remove(plist_path)
        print("[Forge] daily reminder removed")
        return
    forge_bin = shutil.which("forge") or sys.argv[0]
    plist = {
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": [forge_bin, "notify"],
        "StartCalendarInterval": {"Hour": args.hour, "Minute": 0},
        "RunAtLoad": False,
        "StandardOutPath": os.path.expanduser("~/.forge/notify.log"),
        "StandardErrorPath": os.path.expanduser("~/.forge/notify.log"),
    }
    if os.environ.get("FORGE_LEARNER"):
        plist["EnvironmentVariables"] = {"FORGE_LEARNER": os.environ["FORGE_LEARNER"]}
    os.makedirs(os.path.dirname(plist_path), exist_ok=True)
    os.makedirs(os.path.expanduser("~/.forge"), exist_ok=True)
    with open(plist_path, "wb") as f:
        plistlib.dump(plist, f)
    subprocess.run(["launchctl", "unload", plist_path], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["launchctl", "load", plist_path], check=True)
    print(f"[Forge] daily reminder installed: {args.hour:02d}:00 -> {plist_path}")


def cmd_notify(args):
    m = Memory()
    debt = m.review_debt()
    msg = (f"{debt['due']} due; peak risk {debt['peak_risk']:.2f}; "
           f"{m.streak()} day streak")
    if args.dry_run or sys.platform != "darwin" or not shutil.which("osascript"):
        print(f"[Forge] {msg}")
        return
    script = (f'display notification {json.dumps(msg)} '
              f'with title {json.dumps("The Forge")}')
    subprocess.run(["osascript", "-e", script], check=False)
    print("[Forge] notification sent")


def cmd_semantic(args):
    rows = Memory().semantic_recall(args.query, args.limit)
    if args.json:
        _json(rows)
        return
    if not rows:
        print("[Forge] No semantic recall candidates.")
        return
    for r in rows:
        print(f"\n{r['score']:.2f}  {r['topic']} / {r['concept']}")
        print(r["excerpt"])


def cmd_bridge(args):
    m = Memory()
    cards = m.weak_cards(args.limit)
    pair = None
    for a in cards:
        for b in cards:
            if a["topic"] != b["topic"]:
                pair = (a, b)
                break
        if pair:
            break
    if not pair:
        print("[Forge] Need weak cards from at least two topics.")
        return
    a, b = pair
    print(f"# Bridge: {a['concept']} <-> {b['concept']}")
    print(f"Retrieve `{a['concept']}` ({a['topic']}) and `{b['concept']}` ({b['topic']}) together.")
    print("Explain the shared mechanism, the crucial difference, and one example where confusing them would break the answer.")
    print("\nA lesson excerpt:")
    print(m.lesson_for(a["topic"], a["concept"])[:500])
    print("\nB lesson excerpt:")
    print(m.lesson_for(b["topic"], b["concept"])[:500])


def cmd_weekly_sheet(args):
    m = Memory()
    due = m.due_queue(limit=args.limit, days=args.days)
    weak = m.weak_cards(args.limit)
    html = _weekly_sheet_html(due, weak)
    path = args.path or "forge-weekly-review.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[Forge] wrote weekly review sheet to {path}")


def cmd_capstone(args):
    rows = [r for r in Memory().stats() if r["topic"] == args.topic]
    if not rows:
        print("[Forge] No concepts for that topic.")
        return
    concepts = [r["concept"] for r in rows]
    print(f"# Capstone: {args.topic}")
    print("Answer from memory. Use every concept below exactly once as part of one coherent explanation:")
    for i, concept in enumerate(concepts, 1):
        print(f"{i}. {concept}")
    print("\nThen add one failure case where omitting any single concept would make the explanation wrong.")


def cmd_window(args):
    m = Memory()
    if args.which == "next":
        rows = m.due_queue(limit=1, days=3650)
        if rows:
            r = rows[0]
            due = "now" if r["due_in_days"] <= 0 else f"{r['due_in_days']:.1f}d"
            print(f"{due}: {r['topic']} / {r['concept']}")
        else:
            print("[Forge] No scheduled reviews.")
        return
    days = 1 if args.which == "tomorrow" else 3
    rows = m.due_queue(days=days)
    print(f"{args.which}: {len(rows)} review(s)")
    for r in rows[:args.limit]:
        print(f"  {r['risk']:.2f}  {r['topic']} / {r['concept']}")


def cmd_daily_stats(args):
    since = time.time() - DAY
    attempts = [a for a in Memory().attempt_timeline(100000) if a["ts"] >= since]
    learned = {(a["topic"], a["concept"]) for a in attempts if a["score"] >= 0.8}
    failed = {(a["topic"], a["concept"]) for a in attempts if a["score"] < 0.8}
    avg = sum(a["score"] for a in attempts) / len(attempts) if attempts else 0.0
    if args.which == "learned":
        for topic, concept in sorted(learned):
            print(f"{topic} / {concept}")
    elif args.which == "failed":
        for topic, concept in sorted(failed):
            print(f"{topic} / {concept}")
    else:
        print(f"attempts: {len(attempts)}   average score: {avg:.2f}")


def cmd_progress(args):
    rows = Memory().progress_rows()
    if args.json:
        _json(rows)
        return
    if not rows:
        print("[Forge] No active progress.")
        return
    for r in rows:
        print(f"{r['topic']}: {r['concept_idx'] + 1}/{r['total']} next={r['next_concept']}")


def cmd_resume_progress(args):
    rows = Memory().progress_rows()
    if not rows:
        print("[Forge] No active progress.")
        return
    pick = sorted(rows, key=lambda r: r["topic"], reverse=args.newest)[0]
    print(f"[Forge] resume with: forge learn {json.dumps(pick['topic'])}")


def cmd_clear_progress(args):
    Memory().clear_progress(args.topic)
    print(f"[Forge] cleared progress for {args.topic}")


def cmd_analytics(args):
    data = Memory().analytics_summary()
    if args.json:
        _json(data)
        return
    print(f"attempt days: {len(data['attempts_per_day'])}   "
          f"avg score: {data['average_score']:.2f}   "
          f"due this week: {data['reviews_due_this_week']}   "
          f"done this week: {data['reviews_completed_this_week']}")
    print("\nLapses by topic:")
    for topic, n in sorted(data["lapses_by_topic"].items(), key=lambda x: (-x[1], x[0]))[:10]:
        print(f"  {n:>3}  {topic}")
    print("\nMost expensive:")
    for c in data["most_expensive"][:5]:
        print(f"  {c['n_attempts']:>3} attempts  {c['topic']} / {c['concept']}")


def cmd_maintain(args):
    m = Memory()
    report = m.maintenance_report()
    if args.json:
        _json(report)
        return
    print(f"integrity: {report['integrity']}")
    print(f"cards: {report['cards']}   attempts: {report['attempts']}   notes: {report['notes']}   db: {report['db_bytes']} bytes")
    for key in ("missing_lessons", "duplicates", "orphan_attempts", "orphan_notes",
                "future_due_outliers", "negative_intervals", "bad_scores"):
        print(f"{key}: {len(report[key])}")
    if args.vacuum and m.path != ":memory:":
        m.db.execute("VACUUM")
        print("[Forge] vacuum complete")


def cmd_rebuild_fts(args):
    n = Memory().rebuild_fts()
    print(f"[Forge] rebuilt FTS notes index ({n} lesson(s))")


def cmd_csv_export(args):
    rows = [dict(r) for r in Memory().stats()]
    with open(args.path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0]) if rows else ["topic", "concept"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Forge] exported CSV to {args.path}")


def cmd_jsonl_export(args):
    rows = Memory().attempt_timeline(100000)
    with open(args.path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"[Forge] exported JSONL attempts to {args.path}")


def cmd_markdown_export(args):
    m = Memory()
    rows = m.db.execute("SELECT topic, concept, lesson FROM notes ORDER BY topic, concept").fetchall()
    with open(args.path, "w", encoding="utf-8") as f:
        f.write("# Forge Topic Export\n\n")
        for r in rows:
            f.write(f"## {r['topic']} / {r['concept']}\n\n{r['lesson']}\n\n")
    print(f"[Forge] exported Markdown to {args.path}")


def cmd_source_export(args):
    manifest = Memory().source_manifest(args.topic)
    with open(args.path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"[Forge] exported source manifest to {args.path}")


def cmd_graph_json(args):
    with open(args.path, "w", encoding="utf-8") as f:
        json.dump(graph_layout(Memory()), f, indent=2, sort_keys=True)
    print(f"[Forge] exported graph JSON to {args.path}")


def cmd_forecast_export(args):
    data = {"forecast": Memory().forecast(args.days)}
    with open(args.path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    print(f"[Forge] exported forecast to {args.path}")


def cmd_export_checksum(args):
    h = hashlib.sha256()
    with open(args.path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    print(h.hexdigest())


def cmd_doctor(args):
    m = Memory()
    ok = True
    try:
        integrity = m.db.execute("PRAGMA integrity_check").fetchone()[0]
    except Exception as e:
        integrity, ok = str(e), False
    fts = "ok"
    try:
        m.db.execute("SELECT COUNT(*) FROM notes").fetchone()
    except Exception as e:
        fts, ok = str(e), False
    pdf = shutil.which("pdftotext") or "missing"
    model = get_llm().model
    print(f"db: {os.environ.get('FORGE_DB', '~/.forge/forge.db')}")
    print(f"integrity: {integrity}")
    print(f"fts: {fts}")
    print(f"llm: {model}")
    print(f"pdftotext: {pdf}")
    if m.missing_lessons():
        print(f"missing lessons: {len(m.missing_lessons())}")
    if m.duplicate_concepts():
        print(f"duplicate concept names: {len(m.duplicate_concepts())}")
    if args.fix:
        result = m.fix()
        print(f"[Forge] fix: removed {result['orphan_attempts_removed']} orphan attempt(s), "
              f"{result['orphan_notes_removed']} orphan note(s); "
              f"rebuilt FTS ({result['fts_rows_rebuilt']} row(s))")
    if not ok:
        sys.exit(1)


def cmd_rename_topic(args):
    out = Memory().rename_topic(args.old, args.new)
    print(f"[Forge] renamed topic to {out['topic']} ({out['cards']} card(s))")


def cmd_rename_concept(args):
    out = Memory().rename_concept(args.topic, args.old, args.new)
    print(f"[Forge] renamed concept to {out['topic']} / {out['concept']}")


def cmd_merge_topic(args):
    out = Memory().merge_topic(args.old, args.target)
    print(f"[Forge] merged {args.old} -> {out['topic']} "
          f"({out['moved']} moved, {out['merged']} merged)")


def cmd_merge_concept(args):
    out = Memory().merge_concept(args.topic, args.old, args.target)
    print(f"[Forge] merged concept into {out['topic']} / {out['concept']}")


def cmd_suspend(args):
    card = Memory().suspend_card(args.topic, args.concept)
    print(f"[Forge] suspended {card['topic']} / {card['concept']}")


def cmd_unsuspend(args):
    card = Memory().unsuspend_card(args.topic, args.concept)
    print(f"[Forge] unsuspended {card['topic']} / {card['concept']}")


def cmd_due_now(args):
    d = Memory().due_now(args.topic, args.concept)
    print(f"[Forge] due now: {d['topic']} / {d['concept']}")


def cmd_retire(args):
    card = Memory().retire_card(args.topic, args.concept)
    print(f"[Forge] retired {card['topic']} / {card['concept']}")


def cmd_unretire(args):
    card = Memory().unretire_card(args.topic, args.concept)
    print(f"[Forge] unretired {card['topic']} / {card['concept']}")


def cmd_pin(args):
    card = Memory().pin_card(args.topic, args.concept)
    print(f"[Forge] pinned {card['topic']} / {card['concept']}")


def cmd_unpin(args):
    card = Memory().unpin_card(args.topic, args.concept)
    print(f"[Forge] unpinned {card['topic']} / {card['concept']}")


def cmd_delete_topic(args):
    n = Memory().delete_topic(args.topic, args.confirm or "")
    print(f"[Forge] deleted topic {args.topic} ({n} card(s))")


def cmd_delete_concept(args):
    Memory().delete_concept(args.topic, args.concept, args.confirm or "")
    print(f"[Forge] deleted concept {args.topic} / {args.concept}")


def cmd_backup(args):
    m = Memory()
    if m.path == ":memory:":
        raise ValueError("cannot back up an in-memory database")
    dest = args.path or time.strftime("forge-backup-%Y%m%d-%H%M%S.db")
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    out = sqlite_backup(m, dest)
    print(f"[Forge] backed up {m.path} -> {out}")


def cmd_restore(args):
    m = Memory()
    if m.path == ":memory:":
        raise ValueError("cannot restore into an in-memory database")
    src = os.path.abspath(args.path)
    if not os.path.exists(src):
        raise ValueError(f"backup not found: {args.path}")
    if os.path.abspath(args.confirm or "") != src:
        raise ValueError("restore requires --confirm with the exact backup path")
    m.db.close()
    shutil.copy2(src, m.path)
    print(f"[Forge] restored {m.path} from {src}")


def cmd_json_export(args):
    data = Memory().export_json()
    with open(args.path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    print(f"[Forge] exported JSON to {args.path}")


def cmd_json_import(args):
    with open(args.path, encoding="utf-8") as f:
        data = json.load(f)
    out = Memory().import_json(data, merge=args.merge)
    print(f"[Forge] imported {out['cards']} cards and {out['attempts']} attempts")


def sqlite_backup(memory: Memory, path: str) -> str:
    import sqlite3
    path = os.path.abspath(path)
    dest = sqlite3.connect(path)
    try:
        memory.db.backup(dest)
    finally:
        dest.close()
    return path


def _all_source_chunks(manifest: dict, limit: int) -> list[dict]:
    rows = []
    for doc in manifest.get("docs", []):
        for chunk in doc.get("chunks", []):
            rows.append({"source": doc.get("name", "source"),
                         "label": chunk.get("label", ""),
                         "text": chunk.get("text", "")})
            if len(rows) >= limit:
                return rows
    return rows


def _weekly_sheet_html(due: list[dict], weak: list[dict]) -> str:
    def items(rows):
        if not rows:
            return "<li>Clear.</li>"
        out = []
        for r in rows:
            prompt = (f"Reconstruct {esc_html(r['concept'])} from memory, then explain why "
                      f"it is fragile: {esc_html(', '.join(r['reasons']))}.")
            out.append(f"<li><b>{esc_html(r['topic'])} / {esc_html(r['concept'])}</b>"
                       f"<p>{prompt}</p></li>")
        return "\n".join(out)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Forge Weekly Review</title>
<style>body{{font:14px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;}}
h1,h2{{font-family:Georgia,serif;}} li{{break-inside:avoid;margin:0 0 14px;}}
@media print{{button{{display:none}}}}</style></head>
<body><button onclick="print()">Print</button><h1>Forge Weekly Review</h1>
<h2>Due Queue</h2><ol>{items(due)}</ol>
<h2>Weak Concepts</h2><ol>{items(weak)}</ol></body></html>"""


def esc_html(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _tag(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")).lower() or "untagged"


def add_review_options(parser):
    parser.add_argument("--speak", action="store_true")
    parser.add_argument("--voice", default="",
                        help="macOS say voice name for --speak/readback")
    parser.add_argument("--push-to-talk", action="store_true",
                        help="use local whisper.cpp when configured; typed fallback otherwise")
    parser.add_argument("--cleanup-dictation", action="store_true",
                        help="remove common filler words before grading")
    parser.add_argument("--readback", action="store_true",
                        help="read/display each answer before grading")
    parser.add_argument("--hard", action="store_true",
                        help="hide topic names during review")
    parser.add_argument("--confidence", action="store_true",
                        help="ask for 0-5 confidence before grading")
    parser.add_argument("--no-followup", action="store_true",
                        help="disable close-but-incomplete follow-up prompts")


def cmd_export(args):
    m = Memory()
    rows = m.db.execute("""
        SELECT c.topic, c.concept, c.ease, c.lapses, n.lesson
        FROM notes n JOIN cards c ON c.topic = n.topic AND c.concept = n.concept
        ORDER BY c.topic, c.concept""").fetchall()
    if not rows:
        print("[Forge] Nothing to export yet.")
        return
    with open(args.path, "w", encoding="utf-8") as f:
        f.write("#separator:tab\n#html:true\n")
        for r in rows:
            front = f"{r['concept']} — reconstruct from memory ({r['topic']})"
            back = r["lesson"].replace("\t", " ").replace("\n", "<br>")
            if args.source_citations:
                snippets = m.source_snippets(r["topic"], f"{r['concept']} {r['lesson']}", 2)
                if snippets:
                    cites = "<br>".join(
                        f"[{s['label']}] {s['source']}: "
                        f"{s['text'][:220].replace(chr(10), ' ')}"
                        for s in snippets)
                    back += "<br><br>Sources:<br>" + cites
            if args.tags:
                tags = " ".join([
                    "forge", f"topic_{_tag(r['topic'])}",
                    f"ease_{int(r['ease'] * 10)}", f"lapses_{r['lapses']}",
                ])
                f.write(f"{front}\t{back}\t{tags}\n")
            else:
                f.write(f"{front}\t{back}\n")
    print(f"[Forge] exported {len(rows)} cards to {args.path} (Anki: File > Import)")


def cmd_graph(args):
    print(f"[Forge] wrote {export()}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="forge", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", action="store_true",
                   help="quiet JSON output for script-friendly read-only commands")
    profile = argparse.ArgumentParser(add_help=False)
    profile.add_argument("--learner", metavar="NAME",
                         help="use ~/.forge/NAME.db instead of the default memory")
    sub = p.add_subparsers(dest="cmd", required=True)
    learn = sub.add_parser("learn", parents=[profile])
    learn.add_argument("topic")
    learn.add_argument("--from", dest="source", metavar="FILE", action="append",
                       default=[],
                       help="ground the curriculum in a local text/markdown/PDF file; repeatable")
    learn.add_argument("--speak", action="store_true",
                       help="read lessons and questions aloud (macOS)")
    learn.add_argument("--voice", default="",
                       help="macOS say voice name for --speak/readback")
    learn.add_argument("--push-to-talk", action="store_true",
                       help="use local whisper.cpp when configured; typed fallback otherwise")
    learn.add_argument("--cleanup-dictation", action="store_true",
                       help="remove common filler words before grading")
    learn.add_argument("--readback", action="store_true",
                       help="read/display each answer before grading")
    learn.add_argument("--max-failures", type=int, default=0,
                       help="0 = never give up (default)")
    learn.add_argument("--correct", action="store_true",
                       help="ask after each pass for a learner reconstruction to store")
    learn.set_defaults(fn=cmd_learn)
    review = sub.add_parser("review", parents=[profile])
    add_review_options(review)
    review.add_argument("--limit", type=int, default=None,
                        help="review only the highest-risk N due cards")
    review.set_defaults(fn=cmd_review)
    practice = sub.add_parser("practice", parents=[profile])
    add_review_options(practice)
    practice.add_argument("--limit", type=int, default=10)
    practice.set_defaults(fn=cmd_practice)
    cram = sub.add_parser("cram", parents=[profile])
    cram.add_argument("--days", type=int, default=7,
                      help="pull forward cards due within N days")
    cram.add_argument("--limit", type=int, default=20)
    add_review_options(cram)
    cram.set_defaults(fn=cmd_cram)
    due = sub.add_parser("due", parents=[profile])
    due.add_argument("--limit", type=int, default=20)
    due.add_argument("--days", type=int, default=None,
                     help="include cards due within N days, not only overdue")
    due.set_defaults(fn=cmd_due)
    queue = sub.add_parser("queue", parents=[profile])
    queue.add_argument("--limit", type=int, default=20)
    queue.add_argument("--days", type=int, default=None)
    queue.set_defaults(fn=cmd_due)
    weak = sub.add_parser("weak", parents=[profile])
    weak.add_argument("--limit", type=int, default=10)
    weak.set_defaults(fn=cmd_weak)
    sub.add_parser("topics", parents=[profile]).set_defaults(fn=cmd_topics)
    sub.add_parser("health", parents=[profile]).set_defaults(fn=cmd_topics)
    inspect = sub.add_parser("inspect", parents=[profile])
    inspect.add_argument("topic")
    inspect.add_argument("concept")
    inspect.set_defaults(fn=cmd_inspect)
    search = sub.add_parser("search", parents=[profile])
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=10)
    search.set_defaults(fn=cmd_search)
    timeline = sub.add_parser("timeline", parents=[profile])
    timeline.add_argument("--limit", type=int, default=20)
    timeline.add_argument("--topic")
    timeline.add_argument("--concept")
    timeline.set_defaults(fn=cmd_timeline)
    transcript = sub.add_parser("transcript", parents=[profile])
    transcript.add_argument("path", nargs="?")
    transcript.add_argument("--limit", type=int, default=100)
    transcript.add_argument("--topic")
    transcript.add_argument("--concept")
    transcript.set_defaults(fn=cmd_transcript)
    sub.add_parser("debt", parents=[profile]).set_defaults(fn=cmd_debt)
    avalanche = sub.add_parser("avalanche", parents=[profile])
    avalanche.add_argument("--days", type=int, default=14)
    avalanche.add_argument("--budget", type=int, default=20)
    avalanche.set_defaults(fn=cmd_avalanche)
    vacation = sub.add_parser("vacation", parents=[profile])
    vacation.add_argument("days", type=float)
    vacation.add_argument("--topic")
    vacation.add_argument("--concept")
    vacation.set_defaults(fn=cmd_vacation)
    perf = sub.add_parser("performance", parents=[profile])
    perf.add_argument("--by", choices=["weekday", "hour"], default="weekday")
    perf.set_defaults(fn=cmd_performance)
    sub.add_parser("today", parents=[profile]).set_defaults(fn=cmd_today)
    daily5 = sub.add_parser("daily5", parents=[profile])
    add_review_options(daily5)
    daily5.set_defaults(fn=cmd_daily5)
    exam = sub.add_parser("exam", parents=[profile])
    exam.add_argument("topic")
    add_review_options(exam)
    exam.set_defaults(fn=cmd_exam)
    weekly = sub.add_parser("weekly", parents=[profile])
    weekly.add_argument("--limit", type=int, default=10)
    add_review_options(weekly)
    weekly.set_defaults(fn=cmd_weekly)
    sprint = sub.add_parser("sprint", parents=[profile])
    sprint.add_argument("--mode", choices=["mixed", "weak", "newest", "oldest", "overdue", "lapse"],
                        default="mixed")
    sprint.add_argument("--limit", type=int, default=10)
    sprint.add_argument("--topic")
    add_review_options(sprint)
    sprint.set_defaults(fn=cmd_sprint)
    drill = sub.add_parser("drill", parents=[profile])
    drill.add_argument("--kind", choices=["mixed", "weak", "overdue", "lapse"],
                       default="mixed")
    drill.add_argument("--limit", type=int, default=1)
    drill.add_argument("--topic")
    drill.add_argument("--concept")
    add_review_options(drill)
    drill.set_defaults(fn=cmd_drill)
    stuck = sub.add_parser("stuck", parents=[profile])
    stuck.add_argument("--limit", type=int, default=10)
    stuck.add_argument("--min-lapses", type=int, default=1)
    stuck.set_defaults(fn=cmd_stuck)
    missed = sub.add_parser("missed", parents=[profile])
    missed.add_argument("--topic")
    missed.add_argument("--concept")
    missed.add_argument("--limit", type=int, default=20)
    missed.set_defaults(fn=cmd_missed)
    sub.add_parser("calibration", parents=[profile]).set_defaults(fn=cmd_calibration)
    prompts = sub.add_parser("prompts", parents=[profile])
    prompts.add_argument("topic")
    prompts.add_argument("concept")
    prompts.set_defaults(fn=cmd_prompts)
    cloze = sub.add_parser("cloze", parents=[profile])
    cloze.add_argument("path", nargs="?")
    cloze.add_argument("--topic")
    cloze.add_argument("--limit", type=int, default=50)
    cloze.set_defaults(fn=cmd_cloze)
    note = sub.add_parser("note", parents=[profile])
    note.add_argument("topic")
    note.add_argument("concept")
    note.add_argument("--set")
    note.add_argument("--replace-lesson", action="store_true")
    note.set_defaults(fn=cmd_note)
    promote = sub.add_parser("promote-answer", parents=[profile])
    promote.add_argument("topic")
    promote.add_argument("concept")
    promote.add_argument("--replace-lesson", action="store_true")
    promote.set_defaults(fn=cmd_promote_answer)
    checklist = sub.add_parser("source-checklist", parents=[profile])
    checklist.add_argument("source")
    checklist.add_argument("--limit", type=int, default=12)
    checklist.set_defaults(fn=cmd_source_checklist)
    sm = sub.add_parser("source-manifest", parents=[profile])
    sm.add_argument("topic")
    sm.set_defaults(fn=cmd_source_manifest)
    sc = sub.add_parser("source-chunks", parents=[profile])
    sc.add_argument("topic")
    sc.add_argument("--query", default="")
    sc.add_argument("--limit", type=int, default=8)
    sc.set_defaults(fn=cmd_source_chunks)
    cov = sub.add_parser("source-coverage", parents=[profile])
    cov.add_argument("topic")
    cov.set_defaults(fn=cmd_source_coverage)
    uns = sub.add_parser("unsupported", parents=[profile])
    uns.add_argument("topic")
    uns.add_argument("concept", nargs="?")
    uns.add_argument("--limit", type=int, default=12)
    uns.set_defaults(fn=cmd_unsupported)
    glossary = sub.add_parser("source-glossary", parents=[profile])
    glossary.add_argument("topic")
    glossary.add_argument("--limit", type=int, default=20)
    glossary.set_defaults(fn=cmd_source_glossary)
    srep = sub.add_parser("source-report", parents=[profile])
    srep.add_argument("topic")
    srep.set_defaults(fn=cmd_source_report)
    squote = sub.add_parser("source-quotes", parents=[profile])
    squote.add_argument("topic")
    squote.add_argument("query")
    squote.add_argument("--limit", type=int, default=5)
    squote.set_defaults(fn=cmd_source_quotes)
    squiz = sub.add_parser("source-quiz", parents=[profile])
    squiz.add_argument("topic")
    squiz.add_argument("--kind", choices=["glossary", "section", "quote"],
                       default="glossary")
    squiz.add_argument("--limit", type=int, default=8)
    squiz.set_defaults(fn=cmd_source_quiz)
    sub.add_parser("config", parents=[profile]).set_defaults(fn=cmd_config)
    sub.add_parser("profiles", parents=[profile]).set_defaults(fn=cmd_profiles)
    pc = sub.add_parser("profile-copy", parents=[profile])
    pc.add_argument("source")
    pc.add_argument("target")
    pc.add_argument("--overwrite", action="store_true")
    pc.set_defaults(fn=cmd_profile_copy)
    comp = sub.add_parser("completion", parents=[profile])
    comp.add_argument("--shell", choices=["bash", "zsh"], default="bash")
    comp.set_defaults(fn=cmd_completion)
    notify = sub.add_parser("notify", parents=[profile])
    notify.add_argument("--dry-run", action="store_true")
    notify.set_defaults(fn=cmd_notify)
    schedule = sub.add_parser("schedule", parents=[profile])
    schedule.add_argument("action", choices=["install", "remove", "status"])
    schedule.add_argument("--hour", type=int, default=9,
                          help="hour (0-23, local time) for the daily reminder")
    schedule.set_defaults(fn=cmd_schedule)
    semantic = sub.add_parser("semantic", parents=[profile])
    semantic.add_argument("query")
    semantic.add_argument("--limit", type=int, default=10)
    semantic.set_defaults(fn=cmd_semantic)
    bridge = sub.add_parser("bridge", parents=[profile])
    bridge.add_argument("--limit", type=int, default=10)
    bridge.set_defaults(fn=cmd_bridge)
    sheet = sub.add_parser("weekly-sheet", parents=[profile])
    sheet.add_argument("path", nargs="?")
    sheet.add_argument("--limit", type=int, default=12)
    sheet.add_argument("--days", type=int, default=7)
    sheet.set_defaults(fn=cmd_weekly_sheet)
    capstone = sub.add_parser("capstone", parents=[profile])
    capstone.add_argument("topic")
    capstone.set_defaults(fn=cmd_capstone)
    nxt = sub.add_parser("next", parents=[profile])
    nxt.add_argument("--limit", type=int, default=10)
    nxt.set_defaults(fn=cmd_window, which="next")
    tomorrow = sub.add_parser("tomorrow", parents=[profile])
    tomorrow.add_argument("--limit", type=int, default=10)
    tomorrow.set_defaults(fn=cmd_window, which="tomorrow")
    weekend = sub.add_parser("weekend", parents=[profile])
    weekend.add_argument("--limit", type=int, default=10)
    weekend.set_defaults(fn=cmd_window, which="weekend")
    learned = sub.add_parser("learned-today", parents=[profile])
    learned.set_defaults(fn=cmd_daily_stats, which="learned")
    failed = sub.add_parser("failed-today", parents=[profile])
    failed.set_defaults(fn=cmd_daily_stats, which="failed")
    avg = sub.add_parser("score-today", parents=[profile])
    avg.set_defaults(fn=cmd_daily_stats, which="score")
    sub.add_parser("progress", parents=[profile]).set_defaults(fn=cmd_progress)
    ro = sub.add_parser("resume-oldest", parents=[profile])
    ro.set_defaults(fn=cmd_resume_progress, newest=False)
    rn = sub.add_parser("resume-newest", parents=[profile])
    rn.set_defaults(fn=cmd_resume_progress, newest=True)
    cp = sub.add_parser("clear-progress", parents=[profile])
    cp.add_argument("topic")
    cp.set_defaults(fn=cmd_clear_progress)
    sub.add_parser("analytics", parents=[profile]).set_defaults(fn=cmd_analytics)
    maintain = sub.add_parser("maintain", parents=[profile])
    maintain.add_argument("--vacuum", action="store_true")
    maintain.set_defaults(fn=cmd_maintain)
    vacuum = sub.add_parser("vacuum", parents=[profile])
    vacuum.set_defaults(fn=lambda args: cmd_maintain(argparse.Namespace(json=args.json, vacuum=True)))
    sub.add_parser("rebuild-fts", parents=[profile]).set_defaults(fn=cmd_rebuild_fts)
    doctor = sub.add_parser("doctor", parents=[profile])
    doctor.add_argument("--fix", action="store_true",
                        help="remove orphan rows and rebuild the FTS index")
    doctor.set_defaults(fn=cmd_doctor)
    rt = sub.add_parser("rename-topic", parents=[profile])
    rt.add_argument("old")
    rt.add_argument("new")
    rt.set_defaults(fn=cmd_rename_topic)
    rc = sub.add_parser("rename-concept", parents=[profile])
    rc.add_argument("topic")
    rc.add_argument("old")
    rc.add_argument("new")
    rc.set_defaults(fn=cmd_rename_concept)
    mt = sub.add_parser("merge-topic", parents=[profile])
    mt.add_argument("old")
    mt.add_argument("target")
    mt.set_defaults(fn=cmd_merge_topic)
    mc = sub.add_parser("merge-concept", parents=[profile])
    mc.add_argument("topic")
    mc.add_argument("old")
    mc.add_argument("target")
    mc.set_defaults(fn=cmd_merge_concept)
    suspend = sub.add_parser("suspend", parents=[profile])
    suspend.add_argument("topic")
    suspend.add_argument("concept")
    suspend.set_defaults(fn=cmd_suspend)
    unsuspend = sub.add_parser("unsuspend", parents=[profile])
    unsuspend.add_argument("topic")
    unsuspend.add_argument("concept")
    unsuspend.set_defaults(fn=cmd_unsuspend)
    dn = sub.add_parser("due-now", parents=[profile])
    dn.add_argument("topic")
    dn.add_argument("concept")
    dn.set_defaults(fn=cmd_due_now)
    retire = sub.add_parser("retire", parents=[profile])
    retire.add_argument("topic")
    retire.add_argument("concept")
    retire.set_defaults(fn=cmd_retire)
    unretire = sub.add_parser("unretire", parents=[profile])
    unretire.add_argument("topic")
    unretire.add_argument("concept")
    unretire.set_defaults(fn=cmd_unretire)
    pin = sub.add_parser("pin", parents=[profile])
    pin.add_argument("topic")
    pin.add_argument("concept")
    pin.set_defaults(fn=cmd_pin)
    unpin = sub.add_parser("unpin", parents=[profile])
    unpin.add_argument("topic")
    unpin.add_argument("concept")
    unpin.set_defaults(fn=cmd_unpin)
    dt = sub.add_parser("delete-topic", parents=[profile])
    dt.add_argument("topic")
    dt.add_argument("--confirm")
    dt.set_defaults(fn=cmd_delete_topic)
    dc = sub.add_parser("delete-concept", parents=[profile])
    dc.add_argument("topic")
    dc.add_argument("concept")
    dc.add_argument("--confirm")
    dc.set_defaults(fn=cmd_delete_concept)
    backup = sub.add_parser("backup", parents=[profile])
    backup.add_argument("path", nargs="?")
    backup.set_defaults(fn=cmd_backup)
    restore = sub.add_parser("restore", parents=[profile])
    restore.add_argument("path")
    restore.add_argument("--confirm")
    restore.set_defaults(fn=cmd_restore)
    jexp = sub.add_parser("json-export", parents=[profile])
    jexp.add_argument("path")
    jexp.set_defaults(fn=cmd_json_export)
    jimp = sub.add_parser("json-import", parents=[profile])
    jimp.add_argument("path")
    jimp.add_argument("--merge", action="store_true")
    jimp.set_defaults(fn=cmd_json_import)
    csvp = sub.add_parser("csv-export", parents=[profile])
    csvp.add_argument("path")
    csvp.set_defaults(fn=cmd_csv_export)
    jsonl = sub.add_parser("jsonl-export", parents=[profile])
    jsonl.add_argument("path")
    jsonl.set_defaults(fn=cmd_jsonl_export)
    md = sub.add_parser("markdown-export", parents=[profile])
    md.add_argument("path")
    md.set_defaults(fn=cmd_markdown_export)
    sx = sub.add_parser("source-export", parents=[profile])
    sx.add_argument("topic")
    sx.add_argument("path")
    sx.set_defaults(fn=cmd_source_export)
    gj = sub.add_parser("graph-json", parents=[profile])
    gj.add_argument("path")
    gj.set_defaults(fn=cmd_graph_json)
    fx = sub.add_parser("forecast-export", parents=[profile])
    fx.add_argument("path")
    fx.add_argument("--days", type=int, default=14)
    fx.set_defaults(fn=cmd_forecast_export)
    ch = sub.add_parser("checksum", parents=[profile])
    ch.add_argument("path")
    ch.set_defaults(fn=cmd_export_checksum)
    exp = sub.add_parser("export", parents=[profile])
    exp.add_argument("path", nargs="?", default="forge_anki.tsv")
    exp.add_argument("--tags", action="store_true",
                     help="include Anki tags for topic/ease/lapses")
    exp.add_argument("--source-citations", action="store_true",
                     help="append source snippets to card backs when available")
    exp.set_defaults(fn=cmd_export)
    web = sub.add_parser("web", parents=[profile])
    web.add_argument("--port", type=int, default=8765)
    web.set_defaults(fn=cmd_web)
    sub.add_parser("stats", parents=[profile]).set_defaults(fn=cmd_stats)
    sub.add_parser("graph", parents=[profile]).set_defaults(fn=cmd_graph)
    args = p.parse_args(argv)
    try:
        select_profile(args.learner)
        args.fn(args)
    except (SourceError, ValueError) as e:
        print(f"[Forge] {e}", file=sys.stderr)
        sys.exit(2)
    except (KeyboardInterrupt, EOFError):
        print("\n[Forge] Session paused. Your schedule survives in ~/.forge/forge.db")
        sys.exit(130)


if __name__ == "__main__":
    main()
