"""B1 teaching-quality eval harness.

Golden-session records live in evals/golden/*.json; each is a captured stub
run plus deterministic checks (curriculum coverage, lesson markers, quiz
answerability, heuristic-grading bands). The LLM judge is scaffolded but
reports "—" until human calibration labels exist (H6 — human-only; this code
never creates them).
"""
import json
import re
import sys
from pathlib import Path

from .agents import ANGLES, heuristic_grade
from .graph import run_topic
from .llm import Stub
from .memory import Memory

EVALS_DIR = Path(__file__).resolve().parent.parent / "evals"
GOLDEN_DIR = EVALS_DIR / "golden"
HUMAN_LABELS = EVALS_DIR / "calibration" / "human_labels.json"
PASS_ANSWER = "remember alpha beta gamma"
UNCALIBRATED = "uncalibrated: awaiting human labels (H6)"

JUDGE = (
    'Judge the teaching quality of this lesson on the concept "{name}" '
    "(topic: {topic}) from 0.0 to 1.0. Criteria: concrete, precise, "
    "answerable-from-lesson quiz support, no filler.\n---\n{lesson}\n---\n"
    'Reply as JSON: {{"score": 0.0, "reasons": "one sentence"}}'
)

# topic -> optional source text; class is encoded in the record name
CORPUS = [
    ("stem-linear-algebra", "linear algebra", None),
    ("stem-thermodynamics", "thermodynamics", None),
    ("stem-graph-theory", "graph theory", None),
    ("stem-cell-biology", "cell biology", None),
    ("humanities-stoic-philosophy", "stoic philosophy", None),
    ("humanities-french-revolution", "the french revolution", None),
    ("humanities-modernist-poetry", "modernist poetry", None),
    ("skill-touch-typing", "touch typing", None),
    ("skill-knife-sharpening", "knife sharpening", None),
    ("skill-negotiation", "negotiation basics", None),
    ("source-tide-windows", "tide window planning",
     ("Tide windows constrain marine work. Slack water is the safe interval; "
      "current speed and datum offsets set the working window each day.")),
    ("source-epoxy-prep", "epoxy surface prep",
     ("Surface profile and dew point govern epoxy adhesion. Blast to the "
      "specified profile, verify substrate temperature 3C above dew point.")),
    ("source-easement-value", "easement valuation",
     ("An easement's value is the difference between the property value "
      "before and after the encumbrance, supported by paired sales.")),
]

GRADING_BANDS = [
    {"answer": "remember alpha beta gamma", "expected_band": [1.0, 1.0]},
    {"answer": "only alpha here", "expected_band": [0.32, 0.35]},
    {"answer": "alpha and beta together", "expected_band": [0.65, 0.68]},
    {"answer": "no idea at all", "expected_band": [0.0, 0.0]},
]

CHECKS = {
    "curriculum": {
        "required_concepts": ["foundations", "mechanics", "applications"],
        "max_concepts": 5,
    },
    "lesson": {"required_markers": ["LESSON[", "alpha beta gamma"]},
    "quiz": {"min_term_overlap": 1.0},
    "grading": GRADING_BANDS,
}

CONCEPT_LINE = re.compile(r"^\s+\d+\. (.+?) — ")


class _Capture(Stub):
    """Stub that also keeps the parsed artifacts of each agent call."""

    def __init__(self):
        self.concepts, self.lessons, self.quizzes = [], [], []

    def ask(self, prompt, as_json=False):
        out = super().ask(prompt, as_json)
        if '"concepts"' in prompt:
            self.concepts = json.loads(out)["concepts"]
        elif '"questions"' in prompt:
            self.quizzes.append(json.loads(out)["questions"])
        elif '"score"' not in prompt:
            self.lessons.append(out)
        return out


def replay(record: dict) -> tuple[_Capture, list[str]]:
    """Re-run the deterministic stub loop for one golden record."""
    cap, events = _Capture(), []
    run_topic(record["topic"], ask=lambda _p: PASS_ANSWER, emit=events.append,
              llm=cap, grader=cap, memory=Memory(":memory:"),
              source=record.get("source") or "")
    return cap, events


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


def check_record(record: dict, cap: _Capture | None = None) -> dict[str, tuple[bool, str]]:
    """All four deterministic checks; {check: (ok, detail)}."""
    if cap is None:
        cap, _events = replay(record)
    checks, results = record["checks"], {}

    # curriculum: coverage by name containment, count cap, stable ordering
    names = [c["name"] for c in cap.concepts]
    cur = checks["curriculum"]
    missing = [r for r in cur["required_concepts"]
               if not any(r.lower() in n.lower() for n in names)]
    golden_names = [m.group(1) for line in record["session"]
                    if (m := CONCEPT_LINE.match(line))]
    ok = (not missing and len(names) <= cur["max_concepts"]
          and names == golden_names)
    detail = ("ok" if ok else
              f"missing={missing} count={len(names)} order_stable="
              f"{names == golden_names}")
    results["curriculum"] = (ok, detail)

    # lesson: angle marker (first pass always angle 0) + required markers
    angle_word = re.match(r"\w+", ANGLES[0]).group(0)
    bad = [i for i, lesson in enumerate(cap.lessons)
           if f"|{angle_word}]" not in lesson
           or any(mk not in lesson for mk in checks["lesson"]["required_markers"])]
    results["lesson"] = (not bad, "ok" if not bad else f"lessons missing markers: {bad}")

    # quiz: every key point is answerable from its lesson
    floor = checks["quiz"]["min_term_overlap"]
    low = []
    for i, quiz in enumerate(cap.quizzes):
        lesson_terms = _terms(cap.lessons[i]) if i < len(cap.lessons) else set()
        for q in quiz:
            for kp in q["key_points"]:
                t = _terms(kp)
                overlap = len(t & lesson_terms) / len(t) if t else 0.0
                if overlap < floor:
                    low.append(f"c{i}:{kp}={overlap:.2f}")
    results["quiz"] = (not low, "ok" if not low else f"below floor: {low[:4]}")

    # grading bands: deterministic floor calibration of heuristic_grade
    key_points = cap.quizzes[0][0]["key_points"] if cap.quizzes else []
    out_of_band = []
    for g in checks["grading"]:
        score = heuristic_grade(g["answer"], key_points)
        lo, hi = g["expected_band"]
        if not (lo - 1e-9 <= score <= hi + 1e-9):
            out_of_band.append(f"{g['answer']!r}->{score:.2f} not in [{lo},{hi}]")
    results["grading"] = (not out_of_band,
                          "ok" if not out_of_band else "; ".join(out_of_band))
    return results


def judge(llm, record: dict, lessons: list[str]) -> dict:
    """LLM-judge scaffold. Without human calibration labels every metric is
    the literal "—" — a judge score is never fabricated. A stub can never
    judge: its canned output is not a measurement."""
    if not HUMAN_LABELS.exists():
        return {"score": "—", "note": UNCALIBRATED}
    if getattr(llm, "name", "") == "stub":
        return {"score": "—",
                "note": "judge requires a real engine — run with Ollama or "
                        "an Anthropic key, not FORGE_STUB"}
    try:  # H6 done by a human: labels exist, the judge may run
        raw = llm.ask(JUDGE.format(name=record["topic"], topic=record["topic"],
                                   lesson=lessons[0] if lessons else ""),
                      as_json=True)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return {"score": float(json.loads(m.group(0))["score"]),
                "note": "calibrated judge run"}
    except (RuntimeError, ValueError, KeyError, AttributeError):
        return {"score": "—",
                "note": "judge output unparseable — no number reported"}


def load_golden() -> list[dict]:
    records = []
    for path in sorted(GOLDEN_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        rec["_name"] = path.stem
        records.append(rec)
    return records


def run_evals(records: list[dict] | None = None) -> tuple[bool, list[dict]]:
    """(all_deterministic_checks_pass, per-record results)."""
    results = []
    for rec in records if records is not None else load_golden():
        cap, _ = replay(rec)
        checks = check_record(rec, cap)
        results.append({
            "record": rec.get("_name", rec["topic"]),
            "checks": checks,
            "judge": judge(_judge_llm(), rec, cap.lessons),
        })
    ok = all(all(v[0] for v in r["checks"].values()) for r in results)
    return ok, results


def _judge_llm():
    from .llm import get_llm
    return get_llm()


def judge_agreement(judged: list[dict]) -> dict:
    """Agreement between judge scores and human labels (H6 output).

    Labels file: {"<record name>": {"score": float}, ...}. Returns mean
    absolute difference and within-0.15 agreement rate over records where
    both a numeric judge score and a human label exist; "—" otherwise.
    """
    if not HUMAN_LABELS.exists():
        return {"mad": "—", "agree_rate": "—", "n": 0, "note": UNCALIBRATED}
    with open(HUMAN_LABELS, encoding="utf-8") as f:
        labels = json.load(f)
    pairs = [(r["judge"]["score"], labels[r["record"]]["score"])
             for r in judged
             if isinstance(r["judge"].get("score"), float) and r["record"] in labels]
    if not pairs:
        return {"mad": "—", "agree_rate": "—", "n": 0,
                "note": "no overlapping numeric judge scores and labels"}
    diffs = [abs(j - h) for j, h in pairs]
    return {"mad": round(sum(diffs) / len(diffs), 3),
            "agree_rate": round(sum(d <= 0.15 for d in diffs) / len(diffs), 3),
            "n": len(pairs), "note": "judge vs human labels"}


def regenerate() -> int:
    """Rebuild evals/golden/*.json by running the stub loop — reproducible."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for name, topic, source in CORPUS:
        record = {"topic": topic, "source": source, "session": [],
                  "checks": CHECKS}
        _cap, events = replay(record)
        record["session"] = events
        with open(GOLDEN_DIR / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, sort_keys=True)
            f.write("\n")
    return len(CORPUS)


def _clean(s) -> str:
    # terminal-escape spoofing guard: golden fields print raw otherwise
    return str(s).replace("\x1b", "\\x1b")


def cmd_eval(args) -> None:
    if getattr(args, "regenerate", False):
        print(f"[Forge] regenerated {regenerate()} golden record(s) in {GOLDEN_DIR}")
        print("[Forge] review the diff before trusting a green run — "
              "evaluating freshly regenerated goldens can baptize a regression:"
              "\n  git diff --stat evals/golden")
    ok, results = run_evals()
    if getattr(args, "json", False):
        print(json.dumps({"pass": ok, "results": [
            {"record": r["record"],
             "checks": {k: {"ok": v[0], "detail": v[1]}
                        for k, v in r["checks"].items()},
             "judge": r["judge"]} for r in results]}, indent=2))
    else:
        header = f"{'record':<30} {'curriculum':<11} {'lesson':<7} {'quiz':<5} {'grading':<8} judge"
        print(header)
        print("-" * len(header))
        for r in results:
            cells = [("PASS" if r["checks"][k][0] else "FAIL")
                     for k in ("curriculum", "lesson", "quiz", "grading")]
            print(f"{_clean(r['record']):<30} {cells[0]:<11} {cells[1]:<7} "
                  f"{cells[2]:<5} {cells[3]:<8} {_clean(r['judge']['score'])}")
            for k, (okc, detail) in r["checks"].items():
                if not okc:
                    print(f"  !! {k}: {_clean(detail)}")
        note = results[0]["judge"]["note"] if results else UNCALIBRATED
        print(f"\njudge: {_clean(note)}")
        agree = judge_agreement(results)
        if agree["n"]:
            print(f"judge agreement vs human labels: mad={agree['mad']} "
                  f"agree@0.15={agree['agree_rate']} n={agree['n']}")
        print(f"[Forge] eval {'PASS' if ok else 'FAIL'} "
              f"({len(results)} golden record(s))")
    if not ok:
        sys.exit(1)
